import os
import cv2
import torch
import torch.multiprocessing as mp
import shutil
from typing import *
from PIL import Image
import torch.nn as nn
import time
import imageio
import numpy as np
import open3d as o3d
from torchvision import transforms
import torch.nn.functional as F

import utils3d
import json
from tqdm import tqdm
from openai import OpenAI
import t2v_metrics
from collections import deque
from scipy.optimize import linear_sum_assignment

from transformers import CLIPProcessor, CLIPModel
import lpips
import matplotlib.pyplot as plt
import pandas as pd
from render_multiviews_blender_mp import get_multi_view_from_3d_assets_mp
from dotenv import load_dotenv
load_dotenv()
from DSG.query_utils import generate_dsg
from DSG.parse_utils import parse_tuple_output, parse_dependency_output, parse_question_output
from text_gen_utils import describe_objs_from_views, filter_questions
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trellis.pipelines import TrellisImageTo3DAttributeSliderPipeline
from trellis.utils import render_utils, postprocessing_utils
os.environ['SPCONV_ALGO'] = 'native'
from text_gen_utils import quality_and_difference_control, quality_control_only, quality_and_change_control
from point_sampling import get_next_batch_to_run, suggest_batch_by_y_spread

def openai_completion(
	prompt: str,
	model='gpt-5.2',
	temperature=0,
	return_response=False,
	max_completion_tokens=500,
	):
	client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
	resp = client.chat.completions.create(
			model=model,
			messages=[{"role": "user", "content": prompt}],
			temperature=temperature,
			max_completion_tokens=max_completion_tokens,
		)
	
	if return_response:
		return resp

	return resp.choices[0].message.content

def get_questions_VQA(input_text_prompt: str):
    id2prompts = {
        'custom_0': {
            'input': input_text_prompt,
        }
    }

    id2tuple_outputs, id2question_outputs, id2dependency_outputs = generate_dsg(
        id2prompts,
        # you can change this method with any method that takes prompt as input and outputs LLM generation result.
        generate_fn=openai_completion
    )

    qid2dependency = parse_dependency_output(id2dependency_outputs['custom_0']['output'])
    qid2question = parse_question_output(id2question_outputs['custom_0']['output'])
    return qid2dependency, qid2question

def get_views_for_eval(asset_path: str, out_dir: str):
    """Get 4 views for evaluation."""
    num_views = 4
    get_multi_view_from_3d_assets_mp(asset_path, num_views=num_views, out_dir=out_dir, max_workers=num_views, eval_mode=True)

def get_lpips_model(device="cuda"):
    """Load LPIPS model once and reuse."""
    loss_fn = lpips.LPIPS(net='alex').to(device)
    loss_fn.eval()
    return loss_fn

def preprocess_images_for_lpips(images: List[Image.Image], size=64, device="cuda"):
    """Batch preprocess images for LPIPS: resize, normalize to [-1, 1]."""
    # Convert RGBA to RGB if needed
    rgb_images = [img.convert('RGB') if img.mode == 'RGBA' else img for img in images]
    
    # Convert PIL to tensors and stack [N, C, H, W]
    to_tensor = transforms.ToTensor()
    batch = torch.stack([to_tensor(img) for img in rgb_images])  # [N, 3, H, W], range [0, 1]
    
    # Batch resize using F.interpolate (vectorized)
    batch = F.interpolate(batch, size=(size, size), mode='bilinear', align_corners=False)
    
    # Normalize to [-1, 1]
    batch = (batch - 0.5) * 2
    return batch.to(device, dtype=torch.float32)

def calc_LPIPS_similarity(loss_fn, images_1: List[Image.Image], images_2: List[Image.Image], mean: bool = True, device=None):
    """
    Calculate mean LPIPS distance between two sets of corresponding images.
    Lower LPIPS = more similar.
    """
    assert len(images_1) == len(images_2), "Image lists must have same length"
    if device is None:
        device = next(loss_fn.parameters()).device
    
    batch_1 = preprocess_images_for_lpips(images_1, device=device)  # [N, C, H, W]
    batch_2 = preprocess_images_for_lpips(images_2, device=device)  # [N, C, H, W]
    
    with torch.no_grad():
        # LPIPS returns [N, 1, 1, 1] distances for each pair
        distances = loss_fn(batch_1, batch_2)
    
    if mean:
        return distances.mean().item()
    else:
        return distances.view(-1).cpu().numpy()

def get_image_to_3d_single_attribute_slider_pipeline(device="cuda"):
    pipeline = TrellisImageTo3DAttributeSliderPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.to(device)
    return pipeline

def get_llava_score():
    pipe_llava = t2v_metrics.VQAScore(model='llava-v1.5-13b', device="cuda")
    return pipe_llava

def get_clip_flant5_score():
    pipe_clip_flant5 = t2v_metrics.VQAScore(model='clip-flant5-xxl', device="cuda")
    return pipe_clip_flant5

def get_slider_conditions(pipeline: TrellisImageTo3DAttributeSliderPipeline, view_images_dir: str, edited_images_dir: str):
    view_images = [Image.open(os.path.join(view_images_dir, f)) for f in sorted(os.listdir(view_images_dir)) if f.endswith(".png")]
    positive_image_paths = [os.path.join(edited_images_dir, f) for f in sorted(os.listdir(edited_images_dir)) if f.endswith("_pos.png")]
    negative_image_paths = [os.path.join(edited_images_dir, f) for f in sorted(os.listdir(edited_images_dir)) if f.endswith("_neg.png")]
    positive_image_paths.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))
    negative_image_paths.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))
    positive_images = [Image.open(path) for path in positive_image_paths]
    negative_images = [Image.open(path) for path in negative_image_paths]
    cond = pipeline.get_reconstruct_edit_cond(view_images, positive_images, negative_images)
    return cond

def edit_asset_ori_slider_value(
        pipeline: TrellisImageTo3DAttributeSliderPipeline, 
        cond: dict,
        slider_value: float,
        seed: int = 42,
        cfg_strength: float = 3,
    ):
    outputs = pipeline.run_reconstruct_edit_parallel(
        cond, 
        seed=seed,
        slider_scale=slider_value,
        # Optional parameters
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": cfg_strength,
        },
        slat_sampler_params={
            "steps": 12,
            "cfg_strength": cfg_strength,
        },
        ss_parallel='batch',
        slat_parallel='batch',
        gpu_ids=[0, 1, 2, 3],
        ss_batch_size_per_gpu=40,
        slat_batch_size_per_gpu=10,
    )
    return outputs


def compute_subtree_weights(
    questions: Dict[str, str],
    dependencies: Dict[str, List[int]],
    filtered_questions: List[str],
) -> Tuple[List[float], List[int]]:
    """
    Compute importance weights for filtered questions based on subtree size
    in the dependency tree.

    Each question's raw weight equals its subtree size (the question itself
    plus all of its descendants).  Weights are then re-normalised over the
    *filtered* subset so they sum to 1.

    Args:
        questions:          Full ``{qid_str: question_text}`` mapping.
        dependencies:       Full ``{qid_str: [parent_qid_int, ...]}`` mapping.
                            Parent ID 0 means the question is a root.
        filtered_questions: Ordered list of question texts to weight
                            (subset of ``questions`` values).

    Returns:
        (weights, question_ids):
            weights      – list of floats aligned with *filtered_questions*,
                           summing to 1.
            question_ids – list of integer question IDs aligned with
                           *filtered_questions* (needed for conditional eval).
    """
    qids = sorted(int(q) for q in questions.keys())

    children: Dict[int, List[int]] = {qid: [] for qid in qids}
    for qid_str, parents in dependencies.items():
        qid = int(qid_str)
        for parent in parents:
            if parent in children:
                children[parent].append(qid)

    subtree_size: Dict[int, int] = {}
    def _size(qid: int) -> int:
        if qid in subtree_size:
            return subtree_size[qid]
        s = 1 + sum(_size(c) for c in children[qid])
        subtree_size[qid] = s
        return s

    for qid in qids:
        _size(qid)

    text_to_qid = {text.replace("\u2019", "'"): int(qid) for qid, text in questions.items()}

    raw_weights: List[float] = []
    question_ids: List[int] = []
    for q_text in filtered_questions:
        qid = text_to_qid.get(q_text.replace("\u2019", "'"))
        if qid is not None:
            raw_weights.append(float(subtree_size[qid]))
            question_ids.append(qid) # the qid for the question in the filtered_questions list
        else:
            raw_weights.append(1.0)
            question_ids.append(-1)
            print(f"!!! Question {q_text} not found in questions")

    total = sum(raw_weights)
    weights = [w / total for w in raw_weights] if total > 0 else [1.0 / len(raw_weights)] * len(raw_weights)

    return weights, question_ids


def check_boundary(
    outputs,
    unaffected_questions: List[str],
    unaffected_dependencies: Dict[str, List[int]],
    vqa_score: t2v_metrics.VQAScore,
    threshold: float = 0.5,
    temp_dir: str = None,
    weights: Optional[List[float]] = None,
    question_ids: Optional[List[int]] = None,
    boundary_threshold: float = 0.0,
) -> Tuple[List[str], float]:
    """
    Evaluate whether a slider value crosses a quality boundary.

    Args:
        outputs:                 Pipeline outputs containing gaussians to render.
        unaffected_questions:    List of question texts to evaluate.
        unaffected_dependencies: ``{qid_str: [parent_qid_int, ...]}`` dep map.
        vqa_score:             VQA scoring model.
        threshold:               Per-question VQA score below which a question
                                 is considered *directly failed*.
        temp_dir:                If set, key-frames and failure logs are saved.
        weights:                 Importance weight per question (sums to 1).
                                 When ``None``, all questions are equal weight.
        question_ids:            Integer question IDs aligned with
                                 *unaffected_questions* (needed for conditional
                                 failure propagation).
        boundary_threshold:      Weighted failure score above which the slider
                                 value is considered out of bounds.

    Returns:
        (failed_question_texts, weighted_failure_score)
    """
    key_frames = render_utils.render_key_frames(
        outputs['gaussian'][0], bg_color=(.5, .5, .5))['color']
    frame_paths = []
    if temp_dir is not None:
        for f in range(len(key_frames)):
            f_path = os.path.join(temp_dir, f"key_frame_{f}.png")
            imageio.imwrite(f_path, key_frames[f])
            frame_paths.append(f_path)

    n_questions = len(unaffected_questions)
    scores = vqa_score(
        images=frame_paths,
        texts=unaffected_questions,
        question_template='"{}" Please answer yes or no.',
        answer_template='Yes',
    )
    max_scores = scores.max(dim=0).values  # (N,)

    directly_failed = set(
        i for i in range(n_questions) if max_scores[i].item() < threshold
    )

    # --- conditional evaluation: propagate parent failures to children ---
    all_failed = set(directly_failed)
    if question_ids is not None and unaffected_dependencies is not None:
        qid_to_idx = {qid: i for i, qid in enumerate(question_ids) if qid >= 0}

        filtered_children: Dict[int, List[int]] = {
            qid: [] for qid in qid_to_idx
        }
        in_degree: Dict[int, int] = {qid: 0 for qid in qid_to_idx}
        for qid in qid_to_idx:
            parents = unaffected_dependencies.get(str(qid), [])
            for parent in parents: # 0 is not a key in filtered_children
                if parent in filtered_children:
                    filtered_children[parent].append(qid)
                    in_degree[qid] += 1

        
        queue = deque(qid for qid, deg in in_degree.items() if deg == 0)
        topo_order: List[int] = []
        while queue:
            qid = queue.popleft()
            topo_order.append(qid)
            for child in filtered_children[qid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        for qid in topo_order:
            idx = qid_to_idx[qid]
            if idx in all_failed:
                for child in filtered_children[qid]:
                    all_failed.add(qid_to_idx[child])

    # --- weighted failure score ---
    if weights is not None:
        weighted_failure = sum(weights[i] for i in all_failed)
    else:
        weighted_failure = 0.5 # len(all_failed) / n_questions if n_questions > 0 else 0.0

    failed_question_texts = [unaffected_questions[i] for i in sorted(all_failed)]
    failed_question_scores = [max_scores[i].item() for i in sorted(all_failed)]

    if temp_dir is not None:
        with open(os.path.join(temp_dir, "failed_questions.txt"), "w") as f:
            for question, score in zip(failed_question_texts, failed_question_scores):
                f.write(f"{question} {score:.4f}")
                if weights is not None:
                    idx = unaffected_questions.index(question)
                    f.write(f"  [weight={weights[idx]:.4f}]")
                f.write("\n")
            f.write(f"\nboundary_threshold     = {boundary_threshold:.4f}\n")
            if weights is not None:
                f.write(f"weighted_failure_score = {weighted_failure:.4f}\n")
                f.write(f"is_boundary            = {weighted_failure > boundary_threshold}\n")
            else:
                f.write(f"is_boundary            = {len(failed_question_texts) > 0}\n")

    return failed_question_texts, weighted_failure

def _repair_views_impl(lpips_model: nn.Module, reference_view_images: List[Image.Image], edited_view_images: List[Image.Image]) -> List[Image.Image]:
    # cross-product pairing: e.g., 1,2,3,4 -> 1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4
    n1, n2 = len(reference_view_images), len(edited_view_images) # n1 = n2 = 4
    reference_view_images_expanded = [img for img in reference_view_images for _ in range(n2)]
    edited_view_images_expanded = edited_view_images * n1
    lpips_similarity = calc_LPIPS_similarity(lpips_model, reference_view_images_expanded, edited_view_images_expanded, mean=False)
    cost_matrix = lpips_similarity.reshape(n1, n2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    repaired_view_images = [edited_view_images[col_ind[i]] for i in range(n1)]
    return repaired_view_images

def repair_views(lpips_model: nn.Module, reference_view_images_paths: List[str], edited_view_image_paths: List[str]) -> List[Image.Image]:
    reference_view_images = [Image.open(path) for path in reference_view_images_paths]
    edited_view_images = [Image.open(path) for path in edited_view_image_paths]
    return _repair_views_impl(lpips_model, reference_view_images, edited_view_images)

def repair_views_from_images(lpips_model: nn.Module, reference_view_images: List[Image.Image], edited_view_images: List[Image.Image]) -> List[Image.Image]:
    return _repair_views_impl(lpips_model, reference_view_images, edited_view_images)

def check_boundary_simple(
    outputs, # edited views
    unaffected_questions: List[str],
    reference_view_images_paths: List[str],
    editing_prompt: str,
    vqa_score: t2v_metrics.VQAScore,
    threshold: float = 0.5,
    temp_dir: str = None,
    lpips_model: nn.Module = None,
) -> Tuple[List[str], float]:
    """
    Evaluate whether a slider value crosses a quality boundary.

    Args:
        outputs:                 Pipeline outputs containing gaussians to render.
        unaffected_questions:    List of question texts to evaluate.
        unaffected_dependencies: ``{qid_str: [parent_qid_int, ...]}`` dep map.
        vqa_score:             VQA scoring model.
        threshold:               Per-question VQA score below which a question
                                 is considered *directly failed*.
        temp_dir:                If set, key-frames and failure logs are saved.

    Returns:
        (failed_question_texts, weighted_failure_score)
    """
    if outputs is None: # meaning the key frames are already generated
        frame_paths = [os.path.join(temp_dir, f) for f in sorted(os.listdir(temp_dir)) if f.endswith('.png')]
    else:
        key_frames = render_utils.render_key_frames(
            outputs['gaussian'][0], bg_color=(.5, .5, .5))['color']
        frame_paths = []
        if temp_dir is not None:
            for f in range(len(key_frames)):
                f_path = os.path.join(temp_dir, f"key_frame_{f}.png")
                imageio.imwrite(f_path, key_frames[f])
                frame_paths.append(f_path)

    scores = vqa_score(
        images=frame_paths,
        texts=unaffected_questions,
        question_template='"{}" Please answer yes or no.',
        answer_template='Yes',
    )
    max_scores = scores.max(dim=0).values  # (N,)
    failed_questions = [unaffected_questions[i] for i in range(len(unaffected_questions)) if max_scores[i].item() < threshold]
    # --- weighted failure score ---
    failed_question_scores = [max_scores[i].item() for i in range(len(unaffected_questions)) if max_scores[i].item() < threshold]
    # repair images:
    adjusted_dir = os.path.join(temp_dir, "adjusted_frames")
    if os.path.exists(adjusted_dir):
        repaired_view_image_paths = [os.path.join(adjusted_dir, f) for f in sorted(os.listdir(adjusted_dir)) if f.endswith('.png')]
    else:
        repaired_view_images = repair_views(lpips_model, reference_view_images_paths, frame_paths)
        os.makedirs(adjusted_dir, exist_ok=True)
        for i, image in enumerate(repaired_view_images):
            adjusted_path = os.path.join(adjusted_dir, f"key_frame_{i}.png")
            imageio.imwrite(adjusted_path, image)
        repaired_view_image_paths = [os.path.join(adjusted_dir, f) for f in sorted(os.listdir(adjusted_dir)) if f.endswith('.png')]

    # posthoc filter:
    # final_iso_passed = True
    final_integrity_passed = True
    # final_iso_reasoning = []
    final_integrity_reasoning = []
    for ori_img, edited_img in zip(reference_view_images_paths, repaired_view_image_paths):
        result = quality_control_only(ori_img, edited_img, editing_prompt)
        quality_passed = result["quality_passed"]
        quality_reasoning = result["quality_reasoning"]
        if not quality_passed:
            final_integrity_passed = False
            final_integrity_reasoning.append(quality_reasoning)
        # result = quality_and_difference_control(ori_img, edited_img, editing_prompt)
        # iso_passed = result["isolation_passed"]
        # iso_reasoning = result["isolation_reasoning"]
        # integrity_passed = result["integrity_passed"]
        # integrity_reasoning = result["integrity_reasoning"]
        # if not iso_passed:
        #     final_iso_passed = False
        #     final_iso_reasoning.append(iso_reasoning)
        # if not integrity_passed:
        #     final_integrity_passed = False
        #     final_integrity_reasoning.append(integrity_reasoning)
        
    if temp_dir is not None:
        with open(os.path.join(temp_dir, "all_questions.txt"), "w") as f:
            for question, score in zip(unaffected_questions, max_scores):
                f.write(f"{question} {score:.4f}\n")
            # f.write(f"final_iso_passed: {final_iso_passed}\n")
            f.write(f"final_integrity_passed: {final_integrity_passed}\n")
            # f.write(f"final_iso_reasoning: {final_iso_reasoning}\n")
            f.write(f"final_integrity_reasoning: {final_integrity_reasoning}\n")

    return failed_questions, failed_question_scores, final_integrity_passed, final_integrity_reasoning

def _search_one_direction(
    pipeline: TrellisImageTo3DAttributeSliderPipeline,
    vqa_score: t2v_metrics.VQAScore,
    cond: dict,
    unaffected_questions: List[str],
    unaffected_dependencies: Dict[str, List[int]],
    sign: int,
    initial_step: float,
    explore_growth: float,
    min_step: float,
    max_range: float,
    threshold: float,
    out_dir: str,
    reference_view_images_paths: List[str] = None,
    editing_prompt: str = None,
    lpips_model = None,
    weights: Optional[List[float]] = None,
    question_ids: Optional[List[int]] = None,
    boundary_threshold: float = 0.0,
) -> float:
    """Run explore + refine for one direction (sign = +1 or -1)."""
    _cache: Dict[float, bool] = {}

    direction = "pos" if sign == 1 else "neg"
    device = str(next(iter(cond.values())).device) if isinstance(next(iter(cond.values())), torch.Tensor) else "?"

    def _is_boundary(slider_value: float) -> bool:
        key = round(slider_value, 6)
        if key in _cache:
            print(f"  [{direction}|{device}] scale={slider_value:.5f} → cached={'FAIL' if _cache[key] else 'PASS'}", flush=True)
            return _cache[key]
        print(f"  [{direction}|{device}] scale={slider_value:.5f} → generating...", flush=True)
        temp_dir = os.path.join(out_dir, f"search_at_{slider_value:.5f}")
        if os.path.exists(temp_dir) and any(f.endswith('.png') for f in os.listdir(temp_dir)):
            t0 = time.time()
            print(f"  [{direction}|{device}] scale={slider_value:.5f} → cached PASS", flush=True)
            outputs = None
            t_gen = time.time() - t0
        else:
            os.makedirs(temp_dir, exist_ok=True)
            t0 = time.time()
            outputs = edit_asset_ori_slider_value(pipeline, cond, slider_value)
            t_gen = time.time() - t0
        # failed_questions, weighted_failure = check_boundary(
        #     outputs, unaffected_questions, unaffected_dependencies,
        #     vqa_score, threshold, temp_dir,
        #     weights=weights, question_ids=question_ids,
        #     boundary_threshold=boundary_threshold,
        # )
        failed_questions, failed_question_scores, integrity_passed, integrity_reasoning = check_boundary_simple(
            outputs, unaffected_questions, reference_view_images_paths, editing_prompt,
            vqa_score, threshold, temp_dir, lpips_model,
        )
        result = len(failed_questions) > 0 or not integrity_passed
        _cache[key] = result
        verdict = "FAIL" if result else "PASS"
        print(f"  [{direction}|{device}] scale={slider_value:.5f} → {verdict} "
              f"({len(failed_questions)} questions failed, integrity={integrity_passed}) "
              f"[{t_gen:.1f}s gen]", flush=True)
        if failed_questions:
            for q, score in zip(failed_questions, failed_question_scores):
                print(f"    ✗ {q} {score:.4f}", flush=True)
        if not integrity_passed:
            for r in integrity_reasoning:
                print(f"    ✗ [integrity] {r}", flush=True)
        return result # return True if this slider value is a boundary (exclusive)
 
    # Phase 1: exponential exploration
    print(f"[{direction}] Phase 1: exploring outward (step={initial_step}, growth={explore_growth})", flush=True)
    safe = 0.0
    step = initial_step
    while abs(safe + sign * step) <= max_range:
        candidate = safe + sign * step
        if _is_boundary(candidate):
            fail = candidate
            print(f"[{direction}] Boundary found between safe={safe:.4f} and fail={fail:.4f}", flush=True)
            break
        safe = candidate
        step *= explore_growth
    else:
        print(f"[{direction}] No boundary found within ±{max_range}. Returning safe={safe:.4f}", flush=True)
        return safe

    # Phase 2: binary search between safe and fail
    print(f"[{direction}] Phase 2: refining [{safe:.4f}, {fail:.4f}] (min_step={min_step})", flush=True)
    iteration = 0
    while abs(fail - safe) >= min_step:
        iteration += 1
        mid = (safe + fail) / 2.0
        print(f"[{direction}] Refine #{iteration}: [{safe:.4f}, {fail:.4f}] → mid={mid:.4f}", flush=True)
        if _is_boundary(mid):
            fail = mid
        else:
            safe = mid

    print(f"[{direction}] Final boundary: {safe:.4f} (interval [{safe:.4f}, {fail:.4f}])", flush=True)
    return safe


def _boundary_worker(
    gpu_id: int,
    sign: int,
    view_images_dir: str,
    edited_images_dir: str,
    unaffected_questions: List[str],
    unaffected_dependencies: Dict[str, List[int]],
    search_kwargs: dict,
    result_dict: dict,
    out_dir: str,
    startup_barrier=None,
    reference_view_images_paths: List[str] = None,
    editing_prompt: str = None,
    weights: Optional[List[float]] = None,
    question_ids: Optional[List[int]] = None,
    boundary_threshold: float = 0.0,
):
    """Subprocess entry point: creates its own pipeline + VQA model on the given GPU."""
    direction = "positive" if sign == 1 else "negative"
    device = f'cuda:{gpu_id}'

    torch.cuda.set_device(gpu_id)

    print(f"[{direction}|GPU {gpu_id}] Loading pipeline on {device}...", flush=True)
    pipeline = get_image_to_3d_single_attribute_slider_pipeline(device)
    print(f"[{direction}|GPU {gpu_id}] Building conditions...", flush=True)
    cond = get_slider_conditions(pipeline, view_images_dir, edited_images_dir)
    print(f"[{direction}|GPU {gpu_id}] Loading VQA model...", flush=True)
    vqa_score = get_llava_score() #get_clip_flant5_score()
    print(f"[{direction}|GPU {gpu_id}] Loading LPIPS model...", flush=True)
    lpips_model = get_lpips_model()
    print(f"[{direction}|GPU {gpu_id}] All models loaded. Waiting for other worker...", flush=True)

    if startup_barrier is not None:
        startup_barrier.wait()

    print(f"[{direction}|GPU {gpu_id}] Starting boundary search...", flush=True)

    boundary = _search_one_direction(
        pipeline, vqa_score, cond,
        unaffected_questions, unaffected_dependencies,
        sign=sign, **search_kwargs,
        out_dir=out_dir,
        reference_view_images_paths=reference_view_images_paths,
        editing_prompt=editing_prompt,
        lpips_model=lpips_model,
        weights=weights, question_ids=question_ids,
        boundary_threshold=boundary_threshold,
    )
    print(f"[{direction}|GPU {gpu_id}] Done. Boundary = {boundary}", flush=True)
    result_dict[sign] = boundary


def find_boundary_binary_search(
    view_images_dir: str,
    edited_images_dir: str,
    unaffected_questions: List[str],
    unaffected_dependencies: Dict[str, List[int]],
    gpu_ids: List[int] = [0, 1],
    initial_step: float = 5.0,
    explore_growth: float = 2.0,
    min_step: float = 0.25,
    max_range: float = 50.0,
    threshold: float = 0.5,
    out_dir: str = None,
    reference_view_images_paths: List[str] = None,
    editing_prompt_pair: Tuple[str, str] = None,
    weights: Optional[List[float]] = None,
    question_ids: Optional[List[int]] = None,
    boundary_threshold: float = 0.0,
):
    """
    Find the positive and negative slider boundaries via two-phase search.

    Phase 1 — Explore: walk outward from 0 with exponentially growing steps
    (initial_step, initial_step * explore_growth, ...) until a boundary is
    hit or *max_range* is reached.

    Phase 2 — Refine: binary-search between the last safe value and the
    first failing value, halving the interval each iteration until the
    interval width drops below *min_step*.

    When two ``gpu_ids`` are provided, the positive and negative directions
    run as separate processes on separate GPUs (each loads its own pipeline
    and VQA model).  With a single GPU the two directions run sequentially.

    Args:
        view_images_dir: Path to the directory of original multi-view images.
        edited_images_dir: Path to the directory of edited (pos/neg) images.
        unaffected_questions: Questions that should remain valid.
        unaffected_dependencies: Dependencies for the unaffected questions.
        gpu_ids: GPU device ids. Two ids → parallel; one → sequential.
        initial_step: First exploration probe distance from 0.
        explore_growth: Multiplier for each successive exploration step.
        min_step: Stop refining when the interval is narrower than this.
        max_range: Give up exploring beyond this absolute slider value.
        threshold: Per-question VQA score threshold.
        weights: Subtree-size importance weights (sum to 1), aligned with
                 *unaffected_questions*.  When ``None``, all questions are
                 weighted equally.
        question_ids: Integer question IDs aligned with
                      *unaffected_questions* (needed for conditional eval).
        boundary_threshold: Weighted failure score above which a slider value
                            is considered out of bounds.

    Returns:
        (pos_boundary, neg_boundary)
    """
    search_kwargs = dict(
        initial_step=initial_step,
        explore_growth=explore_growth,
        min_step=min_step,
        max_range=max_range,
        threshold=threshold,
    )
    neg_editing_prompt, pos_editing_prompt = editing_prompt_pair
    
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
    if len(gpu_ids) >= 2:
        ctx = mp.get_context('spawn')
        manager = ctx.Manager()
        result_dict = manager.dict()
        barrier = ctx.Barrier(2)

        procs = []
        for sign, gid in zip([+1, -1], gpu_ids[:2]):
            p = ctx.Process(
                target=_boundary_worker,
                args=(
                    gid, sign,
                    view_images_dir, edited_images_dir,
                    unaffected_questions, unaffected_dependencies,
                    search_kwargs, result_dict,
                    out_dir, barrier,
                ),
                kwargs=dict(
                    reference_view_images_paths=reference_view_images_paths,
                    editing_prompt=pos_editing_prompt if sign == 1 else neg_editing_prompt,
                    weights=weights,
                    question_ids=question_ids,
                    boundary_threshold=boundary_threshold,
                ),
            )
            procs.append(p)

        for p in procs:
            p.start()
        for p in procs:
            p.join()

        pos_boundary = result_dict[+1]
        neg_boundary = result_dict[-1]
    else:
        device = f'cuda:{gpu_ids[0]}'
        pipeline = get_image_to_3d_single_attribute_slider_pipeline(device)
        cond = get_slider_conditions(pipeline, view_images_dir, edited_images_dir)
        vqa_score = get_llava_score() #get_clip_flant5_score()
        lpips_model = get_lpips_model()
        pos_boundary = _search_one_direction(
            pipeline, vqa_score, cond,
            unaffected_questions, unaffected_dependencies,
            sign=+1, **search_kwargs,
            out_dir=out_dir,
            reference_view_images_paths=reference_view_images_paths,
            editing_prompt=pos_editing_prompt,
            lpips_model=lpips_model,
            weights=weights, question_ids=question_ids,
            boundary_threshold=boundary_threshold,
        )
        neg_boundary = _search_one_direction(
            pipeline, vqa_score, cond,
            unaffected_questions, unaffected_dependencies,
            sign=-1, **search_kwargs,
            out_dir=out_dir,
            reference_view_images_paths=reference_view_images_paths,
            editing_prompt=neg_editing_prompt,
            lpips_model=lpips_model,
            weights=weights, question_ids=question_ids,
            boundary_threshold=boundary_threshold,
        )
    # write pos_boundary and neg_boundary to one file
    with open(os.path.join(out_dir, "boundaries.txt"), "w") as f:
        f.write(f"pos_boundary: {pos_boundary:.5f}\n")
        f.write(f"neg_boundary: {neg_boundary:.5f}\n")
    return pos_boundary, neg_boundary

def load_pipelines(gpu_ids: List[int]) -> Dict[int, object]:
    """Pre-load pipeline on each GPU.

    Returns:
        Dict mapping gpu_id to pipeline.
    """
    pipelines = {}
    for gid in gpu_ids:
        device = f'cuda:{gid}'
        print(f"[load] Loading pipeline on {device}...", flush=True)
        pipelines[gid] = get_image_to_3d_single_attribute_slider_pipeline(device)
    print(f"[load] All {len(gpu_ids)} pipelines ready.", flush=True)
    return pipelines

def load_conds(pipelines: Dict[int, object], view_images_dir: str, edited_images_dir: str) -> Dict[int, dict]:
    """Compute conditions for each pre-loaded pipeline.

    Returns:
        Dict mapping gpu_id to cond.
    """
    conds = {}
    for gid, pipeline in pipelines.items():
        print(f"[load] Building conditions on cuda:{gid}...", flush=True)
        conds[gid] = get_slider_conditions(pipeline, view_images_dir, edited_images_dir)
    print(f"[load] All {len(pipelines)} conditions ready.", flush=True)
    return conds

def _pool_worker_loop(gpu_id, worker_id, task_queue, result_queue):
    """Main loop for a persistent GPU worker process."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    torch.cuda.set_device(gpu_id)
    print(f"[pool] Loading pipeline on cuda:{gpu_id} worker {worker_id}...", flush=True)
    pipeline = get_image_to_3d_single_attribute_slider_pipeline(f'cuda:{gpu_id}')
    print(f"[pool] Pipeline loaded on cuda:{gpu_id} worker {worker_id}. Waiting for other workers...", flush=True)
    cond = None
    result_queue.put(('ready', worker_id))
    while True:
        task = task_queue.get()
        if task is None:
            break
        cmd = task[0]
        if cmd == 'set_cond':
            _, view_dir, edited_dir = task
            cond = get_slider_conditions(pipeline, view_dir, edited_dir)
            result_queue.put(('cond_ready', worker_id))
        elif cmd == 'generate':
            _, slider_values, view_dir, edited_dir, out_dir = task
            _generate_frames_worker(
                gpu_id, slider_values, view_dir, edited_dir, out_dir,
                pipeline=pipeline, cond=cond,
            )
            result_queue.put(('done', worker_id))


class GPUWorkerPool:
    """Persistent worker processes for frame generation.

    Spawns *workers_per_gpu* processes on each GPU. Each process loads its
    own pipeline and has its own task queue, so there is no GIL contention.
    Ctrl+C in the parent terminates all workers cleanly.

    Args:
        gpu_ids: CUDA device ids to use.
        workers_per_gpu: How many worker processes to place on each GPU.
            Each worker uses ~15 GB VRAM; A100-80GB can hold up to 4–5.
    """

    def __init__(self, gpu_ids: List[int], workers_per_gpu: int = 1):
        import signal
        ctx = mp.get_context('spawn')
        self.gpu_ids = list(gpu_ids)
        self.workers_per_gpu = workers_per_gpu
        self._worker_ids: List[int] = []
        self._task_queues: Dict[int, mp.Queue] = {}
        self._result_queue = ctx.Queue()
        self._processes: Dict[int, mp.Process] = {}
        self._alive = True
        wid = 0
        for gid in self.gpu_ids:
            for _ in range(workers_per_gpu):
                q = ctx.Queue()
                p = ctx.Process(
                    target=_pool_worker_loop,
                    args=(gid, wid, q, self._result_queue),
                )
                p.start()
                self._task_queues[wid] = q
                self._processes[wid] = p
                self._worker_ids.append(wid)
                wid += 1
        self._wait_results(len(self._worker_ids), tag='ready')
        n = len(self._worker_ids)
        self._orig_sigint = signal.signal(signal.SIGINT, self._handle_sigint)
        print(f"[GPUWorkerPool] {n} workers ready "
              f"({workers_per_gpu}/gpu x {len(gpu_ids)} gpus).", flush=True)

    def _handle_sigint(self, signum, frame):
        import signal
        print("\n[GPUWorkerPool] Ctrl+C received, killing workers...", flush=True)
        self.shutdown()
        signal.signal(signal.SIGINT, self._orig_sigint or signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGINT)

    def _wait_results(self, n: int, tag: str = None):
        """Wait for *n* results, raising KeyboardInterrupt promptly on Ctrl+C."""
        received = 0
        while received < n:
            if not self._alive:
                raise KeyboardInterrupt
            try:
                msg = self._result_queue.get(timeout=1.0)
            except Exception:
                continue
            if tag is not None:
                assert msg[0] == tag, f"Expected '{tag}', got: {msg}"
            received += 1

    def set_conds(self, view_images_dir: str, edited_images_dir: str):
        """Send new conditions to all workers (call once per asset)."""
        for wid in self._worker_ids:
            self._task_queues[wid].put(('set_cond', view_images_dir, edited_images_dir))
        self._wait_results(len(self._worker_ids))

    def generate(
        self,
        view_images_dir: str,
        edited_images_dir: str,
        slider_values: List[float],
        out_dir: str,
    ):
        """Distribute slider values round-robin across workers and wait."""
        n_workers = len(self._worker_ids)
        worker_chunks: Dict[int, List[float]] = {wid: [] for wid in self._worker_ids}
        for i, sv in enumerate(slider_values):
            wid = self._worker_ids[i % n_workers]
            worker_chunks[wid].append(sv)
        n_submitted = 0
        for wid, chunk in worker_chunks.items():
            if chunk:
                self._task_queues[wid].put(
                    ('generate', chunk, view_images_dir, edited_images_dir, out_dir))
                n_submitted += 1
        self._wait_results(n_submitted)

    def shutdown(self):
        """Immediately terminate all workers (SIGTERM then SIGKILL)."""
        if not self._alive:
            return
        self._alive = False
        import signal
        if hasattr(self, '_orig_sigint'):
            signal.signal(signal.SIGINT, self._orig_sigint or signal.SIG_DFL)
        for p in self._processes.values():
            if p.is_alive():
                p.terminate()
        for p in self._processes.values():
            p.join(timeout=2)
        for p in self._processes.values():
            if p.is_alive():
                try:
                    os.kill(p.pid, signal.SIGKILL)
                except OSError:
                    pass
                p.join(timeout=2)
        self._processes.clear()
        print("[GPUWorkerPool] All workers stopped.", flush=True)

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False


def _generate_frames_worker(
    gpu_id: int,
    slider_values: List[float],
    view_images_dir: str,
    edited_images_dir: str,
    out_dir: str,
    barrier=None,
    pipeline=None,
    cond=None,
):
    """Generate and save key-frames for assigned slider values.

    If pipeline/cond are provided, uses them directly (thread-safe path).
    Otherwise loads its own models (subprocess path).
    """
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)

    if pipeline is None or cond is None:
        print(f"[lpips|GPU {gpu_id}] Loading pipeline on {device}...", flush=True)
        pipeline = get_image_to_3d_single_attribute_slider_pipeline(device)
        cond = get_slider_conditions(pipeline, view_images_dir, edited_images_dir)
        print(f"[lpips|GPU {gpu_id}] Ready. Waiting for other workers...", flush=True)

    if barrier is not None:
        barrier.wait()

    for slider_value in slider_values:
        frames_dir = os.path.join(out_dir, f"search_at_{slider_value:.5f}")
        has_frames = os.path.exists(frames_dir) and any(f.endswith('.png') for f in os.listdir(frames_dir))
        has_glb = os.path.exists(os.path.join(frames_dir, "output.glb")) if has_frames else False
        if has_frames and has_glb:
            print(f"[lpips|GPU {gpu_id}] scale={slider_value:.5f} → cached", flush=True)
            continue
        print(f"[lpips|GPU {gpu_id}] scale={slider_value:.5f} → generating...", flush=True)
        t0 = time.time()
        outputs = edit_asset_ori_slider_value(pipeline, cond, slider_value)
        os.makedirs(frames_dir, exist_ok=True)
        if not has_glb:
            
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0], outputs['mesh'][0],
                simplify=0.95, texture_size=1024,
            )
            glb.export(os.path.join(frames_dir, "output.glb"))
        if not has_frames:
            key_frames = render_utils.render_key_frames(
                outputs['gaussian'][0], bg_color=(.5, .5, .5))['color']
            for f in range(len(key_frames)):
                imageio.imwrite(os.path.join(frames_dir, f"key_frame_{f}.png"), key_frames[f])
        print(f"[lpips|GPU {gpu_id}] scale={slider_value:.5f} → done [{time.time() - t0:.1f}s]", flush=True)

def run_generation_multiprocess(
    view_images_dir: str,
    edited_images_dir: str,
    values_to_generate: List[float],
    out_dir: str = None,
    gpu_ids: List[int] = [0],
    workers_per_gpu: int = 1,
    pipelines: Dict[int, object] = None,
    pool: GPUWorkerPool = None,
    conds: Dict[int, dict] = None,
):
    """Generate frames across GPUs.

    Args:
        pipelines: Optional dict from load_pipelines(). When provided with
                   conds, uses threads (shared memory) instead of spawn
                   processes, avoiding redundant model loading.
        conds: Optional dict from load_conds().
        pool: Optional GPUWorkerPool for persistent-process parallelism.
    """
    if pool is not None:
        pool.generate(view_images_dir, edited_images_dir, values_to_generate, out_dir)
        return

    gid = gpu_ids[0]
    _generate_frames_worker(
        gid, values_to_generate, view_images_dir, edited_images_dir, out_dir,
        pipeline=pipelines[gid] if pipelines else None,
        cond=conds[gid] if conds else None,
    )

def get_LPIPS_score_relative_to_reference(
    lpips_model: nn.Module,
    reference_sv_value: float,
    sv_values: List[float],
    out_dir: str,
):
    lpips_scores = []
    reference_frames_dir = os.path.join(out_dir, f"search_at_{reference_sv_value:.5f}")
    reference_key_frames = [Image.open(os.path.join(reference_frames_dir, f))
                    for f in sorted(os.listdir(reference_frames_dir)) if f.endswith(".png") and f.startswith('key_frame_')]
    for i, slider_value in enumerate(sv_values):
        if slider_value == reference_sv_value:
            lpips_score = 0.0
            lpips_scores.append(lpips_score)
            continue
        frames_dir = os.path.join(out_dir, f"search_at_{slider_value:.5f}")
        adjusted_frames_dir = os.path.join(frames_dir, f"adjusted_views")
        if os.path.exists(adjusted_frames_dir) and any(f.endswith('.png') for f in os.listdir(adjusted_frames_dir)):
            key_frames = [Image.open(os.path.join(adjusted_frames_dir, f))
                    for f in sorted(os.listdir(adjusted_frames_dir)) if f.endswith(".png") and f.startswith('key_frame_')]
        else:
            key_frames = [Image.open(os.path.join(frames_dir, f))
                        for f in sorted(os.listdir(frames_dir)) if f.endswith(".png") and f.startswith('key_frame_')]
            # repair key_frames to match reference key frames.
            key_frames = repair_views_from_images(lpips_model, reference_key_frames, key_frames)
            # save to adjusted view images
            os.makedirs(adjusted_frames_dir, exist_ok=True)
            for f in range(len(key_frames)):
                imageio.imwrite(os.path.join(adjusted_frames_dir, f"key_frame_{f}.png"), key_frames[f])
        # if slider_value > reference_sv_value, +, otherwise -
        sign = 1 if slider_value > reference_sv_value else -1
        lpips_score = calc_LPIPS_similarity(lpips_model, reference_key_frames, key_frames) * sign
        print(f"LPIPS score for slider value {slider_value:.5f} relative to reference {reference_sv_value:.5f}: {lpips_score:.5f}")
        lpips_scores.append(lpips_score)
    return lpips_scores

def get_quality_change_verification(
    editing_prompt_pair: Tuple[str, str],
    reference_sv_value: float,
    sv_values: List[float],
    out_dir: str,
):
    passed_results = []
    reasoning_results = []
    reference_frames_dir = os.path.join(out_dir, f"search_at_{reference_sv_value:.5f}")
    if not os.path.exists(os.path.join(reference_frames_dir, 'concat.png')):
        reference_key_frames = [cv2.imread(os.path.join(reference_frames_dir, f))
                        for f in sorted(os.listdir(reference_frames_dir)) if f.endswith(".png") and f.startswith('key_frame_')]
        reference_key_frames_concat = cv2.hconcat(reference_key_frames)
        cv2.imwrite(os.path.join(reference_frames_dir, 'concat.png'), reference_key_frames_concat)
    reference_concat_path = os.path.join(reference_frames_dir, 'concat.png')
    for i, slider_value in enumerate(sv_values):
        if slider_value == reference_sv_value:
            passed_results.append(True)
            reasoning_results.append("Reference value -- skipped verification.")
            continue
        adjusted_frames_dir = os.path.join(out_dir, f"search_at_{slider_value:.5f}", f"adjusted_views")
        if not os.path.exists(os.path.join(adjusted_frames_dir, 'concat.png')):
            adjusted_key_frames = [cv2.imread(os.path.join(adjusted_frames_dir, f))
                            for f in sorted(os.listdir(adjusted_frames_dir)) if f.endswith(".png") and f.startswith('key_frame_')]
            adjusted_key_frames_concat = cv2.hconcat(adjusted_key_frames)
            cv2.imwrite(os.path.join(adjusted_frames_dir, 'concat.png'), adjusted_key_frames_concat)
        adjusted_concat_path = os.path.join(adjusted_frames_dir, 'concat.png')
        editing_prompt = editing_prompt_pair[0] if slider_value < reference_sv_value else editing_prompt_pair[1]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = quality_and_change_control(reference_concat_path, adjusted_concat_path, editing_prompt)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"VLM API error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"VLM API failed after {max_retries} attempts for slider {slider_value:.5f}. Defaulting to passed=True.")
                    res = {"passed": True, "evaluation_reasoning": f"API error after {max_retries} retries: {e}"}
        passed_results.append(res['passed'])
        reasoning_results.append(res['evaluation_reasoning'])
        if not res['passed']:
            print(f"🚫 Failed quality change verification for slider value {slider_value:.5f}: {res['evaluation_reasoning']}")
            with open(os.path.join(out_dir, f"search_at_{slider_value:.5f}", f"failed_reasoning.txt"), "w") as f:
                f.write(res['evaluation_reasoning'] + '\n')
        else:
            print(f"✅ Passed quality change verification for slider value {slider_value:.5f}")
    return passed_results, reasoning_results

def get_lpips_curve_gradient_based(
    view_images_dir: str,
    edited_images_dir: str,
    editing_prompt_pair: Tuple[str, str],
    boundaries: Tuple[float, float],
    out_dir: str = None,
    gpu_ids: List[int] = [0],
    n_samples: int = 9,
    workers_per_gpu: int = 1,
    pipelines: Dict[int, object] = None,
    conds: Dict[int, dict] = None,
    pool: GPUWorkerPool = None,
):
    lpips_model = get_lpips_model(f'cuda:{gpu_ids[0]}')
    if pool is None:
        if pipelines is None:
            pipelines = load_pipelines(gpu_ids)
        if conds is None:
            conds = load_conds(pipelines, view_images_dir, edited_images_dir)
    pos_boundary, neg_boundary = boundaries
    ori_slider_values = np.linspace(neg_boundary, pos_boundary, n_samples).tolist()
    # add reference slider value to the list if not already in the list
    if 0 not in ori_slider_values:
        ori_slider_values.append(0)
    # initial observations
    values_to_generate = [
        sv for sv in ori_slider_values
        if not os.path.exists(os.path.join(out_dir, f"search_at_{sv:.5f}"))
        or not any(f.endswith('.png') for f in os.listdir(os.path.join(out_dir, f"search_at_{sv:.5f}")))
    ]
    T0 = time.time()
    time_to_generate_initial_frames = time.time()
    # step 1. generate key-frames for initial observations if not generated already
    if values_to_generate:
        run_generation_multiprocess(
            view_images_dir, edited_images_dir, values_to_generate, out_dir, gpu_ids, workers_per_gpu,
            pipelines=pipelines, pool=pool, conds=conds)
    time_to_generate_initial_frames = time.time() - time_to_generate_initial_frames
    # step 2. compute LPIPS from saved frames
    lpips_scores = get_LPIPS_score_relative_to_reference(lpips_model, 0, ori_slider_values, out_dir)
    passed_results, reasoning_results = get_quality_change_verification(editing_prompt_pair, 0, ori_slider_values, out_dir)

    banned_slider_values = []
    banned_lpips_scores = []
    observation_slider_values = np.array([sv for sv, p in zip(ori_slider_values, passed_results) if p])
    observation_lpips_scores = np.array([ls for ls, p in zip(lpips_scores, passed_results) if p])
    for sv, ls, p in zip(ori_slider_values, lpips_scores, passed_results):
        if not p:
            banned_slider_values.append(sv)
            banned_lpips_scores.append(ls)
    print(f"Initial: {len(observation_slider_values)} valid, {len(banned_slider_values)} banned")

    # boundary fallback: if an outermost initial value was banned, probe inward
    sorted_initial = sorted(ori_slider_values)
    banned_initial_set = {sv for sv, p in zip(ori_slider_values, passed_results) if not p}
    boundary_step = 0.25
    for boundary_sv, next_initial, direction in [
        (neg_boundary, sorted_initial[1], 1),
        (pos_boundary, sorted_initial[-2], -1),
    ]:
        if boundary_sv not in banned_initial_set:
            continue
        probe = boundary_sv + direction * boundary_step
        while abs(probe - next_initial) > 0.1:
            probe_to_gen = [probe] if (
                not os.path.exists(os.path.join(out_dir, f"search_at_{probe:.5f}"))
                or not any(f.endswith('.png') for f in os.listdir(os.path.join(out_dir, f"search_at_{probe:.5f}")))
            ) else []
            if probe_to_gen:
                run_generation_multiprocess(
                    view_images_dir, edited_images_dir, probe_to_gen, out_dir, gpu_ids, 1,
                    pipelines=pipelines, pool=pool, conds=conds)
            probe_lpips = get_LPIPS_score_relative_to_reference(lpips_model, 0, [probe], out_dir)
            probe_passed, _ = get_quality_change_verification(editing_prompt_pair, 0, [probe], out_dir)
            if probe_passed[0]:
                observation_slider_values = np.append(observation_slider_values, probe)
                observation_lpips_scores = np.append(observation_lpips_scores, probe_lpips[0])
                print(f"Boundary recovery: found valid replacement {probe:.5f} for banned boundary {boundary_sv:.5f}")
                break
            else:
                banned_slider_values.append(probe)
                banned_lpips_scores.append(probe_lpips[0])
                print(f"Boundary recovery: {probe:.5f} also failed, probing further...")
            probe += direction * boundary_step

    sorted_indices = np.argsort(observation_slider_values)
    observation_slider_values = observation_slider_values[sorted_indices]
    observation_lpips_scores = observation_lpips_scores[sorted_indices]

    plot_lpips_curve(observation_slider_values, observation_lpips_scores, out_dir, "gradient_rnd_0",
                     banned_slider_values=np.array(banned_slider_values),
                     banned_lpips_scores=np.array(banned_lpips_scores))

    round = 0
    time_to_generate_new_frames = time.time()
    max_rounds = 2
    while round < max_rounds:
        if len(observation_slider_values) < 2:
            print("Not enough valid observations for curve fitting. Stopping iteration.")
            break
        print('searching for next batch to run...')
        next_sv_batch = suggest_batch_by_y_spread(
            observation_slider_values, observation_lpips_scores, n_samples,
            x_range=(neg_boundary, pos_boundary),
            banned_x=np.array(banned_slider_values) if banned_slider_values else None)
        next_sv_batch = [sv for sv in next_sv_batch
                         if np.min(np.abs(observation_slider_values - sv)) > 0.1]
        print(f"Round {round+1} | Next slider values to generate: {next_sv_batch}")
        if not next_sv_batch:
            print('No new slider values to generate. Stopping...')
            break
        new_svs_to_generate = [
            sv for sv in next_sv_batch
            if not os.path.exists(os.path.join(out_dir, f"search_at_{sv:.5f}"))
            or not any(f.endswith('.png') for f in os.listdir(os.path.join(out_dir, f"search_at_{sv:.5f}")))
        ]
        if new_svs_to_generate:
            run_generation_multiprocess(
                view_images_dir, edited_images_dir, new_svs_to_generate, out_dir, gpu_ids, 1,
                pipelines=pipelines, pool=pool, conds=conds)
        new_lpips_scores = get_LPIPS_score_relative_to_reference(lpips_model, 0, next_sv_batch, out_dir)
        new_passed, _ = get_quality_change_verification(editing_prompt_pair, 0, next_sv_batch, out_dir)
        for sv, ls, p in zip(next_sv_batch, new_lpips_scores, new_passed):
            if p:
                observation_slider_values = np.append(observation_slider_values, sv)
                observation_lpips_scores = np.append(observation_lpips_scores, ls)
            else:
                banned_slider_values.append(sv)
                banned_lpips_scores.append(ls)
        sorted_indices = np.argsort(observation_slider_values)
        observation_slider_values = observation_slider_values[sorted_indices]
        observation_lpips_scores = observation_lpips_scores[sorted_indices]
        plot_lpips_curve(observation_slider_values, observation_lpips_scores, out_dir, f"gradient_rnd_{round+1}",
                         banned_slider_values=np.array(banned_slider_values),
                         banned_lpips_scores=np.array(banned_lpips_scores))
        print(f"Round {round+1}: {len(observation_slider_values)} valid, {len(banned_slider_values)} banned")
        round += 1
    time_to_generate_new_frames = time.time() - time_to_generate_new_frames
    T1 = time.time()
    print(f"Time taken: {T1 - T0:.2f} seconds")
    all_sv = np.concatenate([observation_slider_values, np.array(banned_slider_values)])
    all_lp = np.concatenate([observation_lpips_scores, np.array(banned_lpips_scores)])
    all_valid = np.concatenate([np.ones(len(observation_slider_values)),
                                np.zeros(len(banned_slider_values))])
    sorted_idx = np.argsort(all_sv)
    with open(os.path.join(out_dir, f"lpips_curve_gradient_{n_samples}.txt"), "w") as f:
        for sv, ls, v in zip(all_sv[sorted_idx], all_lp[sorted_idx], all_valid[sorted_idx]):
            f.write(f"{sv:.5f},{ls:.5f},{int(v)}\n")
        f.write("\n")
        f.write(f"Time taken to generate initial frames: {time_to_generate_initial_frames:.2f} seconds\n")
        f.write(f"Time taken to generate new frames: {time_to_generate_new_frames:.2f} seconds\n")
        f.write(f"Total time taken: {T1 - T0:.2f} seconds\n")
        f.write(f"Number of rounds: {round}\n")
        f.write(f"Valid points: {len(observation_slider_values)}, Banned points: {len(banned_slider_values)}\n")

    return observation_slider_values, observation_lpips_scores

def plot_lpips_curve(observation_slider_values, observation_lpips_scores, out_dir, tag,
                     banned_slider_values=None, banned_lpips_scores=None):
    lowest_lpips_score_index = np.argmin(observation_lpips_scores)
    highest_lpips_score_index = np.argmax(observation_lpips_scores)
    plt.plot([observation_slider_values[lowest_lpips_score_index], observation_slider_values[highest_lpips_score_index]], 
        [observation_lpips_scores[lowest_lpips_score_index], observation_lpips_scores[highest_lpips_score_index]], '--', color='gray')
    plt.plot(observation_slider_values, observation_lpips_scores, '-o', markersize=5, label='valid')
    if banned_slider_values is not None and len(banned_slider_values) > 0:
        plt.scatter(banned_slider_values, banned_lpips_scores,
                    marker='x', color='red', s=100, zorder=5, label='banned')
    plt.xlabel('Slider Value')
    plt.ylabel('LPIPS Score')
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"lpips_curve_{tag}.png"))
    plt.close()

def get_lpips_curve(
    view_images_dir: str,
    edited_images_dir: str,
    boundaries: Tuple[float, float],
    out_dir: str = None,
    gpu_ids: List[int] = [0],
    n_samples: int = 11,
    workers_per_gpu: int = 1,
    pipelines: Dict[int, object] = None,
    conds: Dict[int, dict] = None,
    pool: GPUWorkerPool = None,
):
    pos_boundary, neg_boundary = boundaries
    ori_slider_values = np.linspace(neg_boundary, pos_boundary, n_samples).tolist()

    # --- Phase 1: generate key-frames (parallel across GPUs) ---
    values_to_generate = [
        sv for sv in ori_slider_values
        if not os.path.exists(os.path.join(out_dir, f"search_at_{sv:.5f}"))
        or not any(f.endswith('.png') for f in os.listdir(os.path.join(out_dir, f"search_at_{sv:.5f}")))
    ]

    if values_to_generate:
        run_generation_multiprocess(
            view_images_dir, edited_images_dir, values_to_generate, out_dir,
            gpu_ids, workers_per_gpu, pipelines=pipelines, pool=pool, conds=conds)

    # --- Phase 2: compute LPIPS from saved frames (cheap, sequential) ---
    lpips_model = get_lpips_model(f'cuda:{gpu_ids[0]}')
    lpips_scores: List[float] = []
    reference_key_frames = None
    for i, slider_value in enumerate(ori_slider_values):
        frames_dir = os.path.join(out_dir, f"search_at_{slider_value:.5f}")
        key_frames = [Image.open(os.path.join(frames_dir, f))
                      for f in sorted(os.listdir(frames_dir)) if f.endswith(".png")]
        if i == 0:
            lpips_score = 0.0
            reference_key_frames = key_frames
        else:
            lpips_score = calc_LPIPS_similarity(lpips_model, reference_key_frames, key_frames)
        lpips_scores.append(lpips_score)

    with open(os.path.join(out_dir, "lpips_curve.csv"), "w") as f:
        for ori_sv, ls in zip(ori_slider_values, lpips_scores):
            f.write(f"{ori_sv:.5f},{ls:.5f}\n")
    return ori_slider_values, lpips_scores

def make_non_decreasing_y(arr):
    res = []
    max_value = arr[0, 1]
    for i in range(arr.shape[0]):
        x, y = arr[i]
        if y >= max_value:
            res.append((x, y))
            max_value = y
    return np.array(res)

def generate_new_sliders_from_lpips_curve(
    view_images_dir: str,
    edited_images_dir: str,
    lpips_curve_path: str,
    out_dir: str = None,
    gpu_ids: List[int] = [0],
    n_samples: int = 11,
    workers_per_gpu: int = 5,
    pipelines: Dict[int, object] = None,
    conds: Dict[int, dict] = None,
    pool: GPUWorkerPool = None,
):
    with open(lpips_curve_path, "r") as f:
        lpips_curve = [line.strip().split(',') for line in f.readlines()]
        lpips_curve = np.array(lpips_curve, dtype=float)
    # remove unmonotonic points

    min_lpips_score = np.min(lpips_curve[:, 1])
    max_lpips_score = np.max(lpips_curve[:, 1])
    new_sliders_lpips_projected = np.linspace(min_lpips_score, max_lpips_score, n_samples)
    # use lpips_curve interpolation to find original slider value that can result in desired lpips score
    new_sliders = []
    monotonic_lpips_curve = make_non_decreasing_y(lpips_curve)
    for new_slider_lpips in new_sliders_lpips_projected:
        new_slider = np.interp(new_slider_lpips, monotonic_lpips_curve[:, 1], monotonic_lpips_curve[:, 0])
        new_sliders.append(new_slider)

    all_sliders = new_sliders.copy()
    print(f'Also adding original slider values...')
    ori_slider_values = np.linspace(-5, 5, n_samples).tolist()
    for ori_slider_value in ori_slider_values:
        if ori_slider_value not in new_sliders:
            all_sliders.append(ori_slider_value)
    print(f"All slider values: {all_sliders}")
    values_to_generate = [
        sv for sv in all_sliders
        if not os.path.exists(os.path.join(out_dir, f"search_at_{sv:.5f}"))
        or not any(f.endswith('.png') for f in os.listdir(os.path.join(out_dir, f"search_at_{sv:.5f}")))
    ]

    if values_to_generate:
        run_generation_multiprocess(
            view_images_dir, edited_images_dir, values_to_generate, out_dir,
            gpu_ids, workers_per_gpu, pipelines=pipelines, pool=pool, conds=conds)
    # --- Phase 2: compute LPIPS from saved frames (cheap, sequential) ---
    lpips_model = get_lpips_model(f'cuda:{gpu_ids[0]}')
    actual_lpips_scores: List[float] = []
    reference_key_frames = None
    for i, slider_value in enumerate(new_sliders):
        frames_dir = os.path.join(out_dir, f"search_at_{slider_value:.5f}")
        key_frames = [Image.open(os.path.join(frames_dir, f))
                      for f in sorted(os.listdir(frames_dir)) if f.endswith(".png")]
        if i == 0:
            lpips_score = 0.0
            reference_key_frames = key_frames
        else:
            lpips_score = calc_LPIPS_similarity(lpips_model, reference_key_frames, key_frames)
        actual_lpips_scores.append(lpips_score)
    with open(os.path.join(out_dir, "new_sliders_lpips_scores.csv"), "w") as f:
        # write new_sliders, new_sliders_lpips_projected, and actual_lpips_scores to file
        for sv, lpips_proj, lpips_score in zip(new_sliders, new_sliders_lpips_projected, actual_lpips_scores):
            f.write(f"{sv:.5f},{lpips_proj:.5f},{lpips_score:.5f}\n")
    return new_sliders


def _parse_lpips_curve(lpips_curve_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse LPIPS curve file.

    Args:
        lpips_curve_path: Path to LPIPS curve file

    Returns:
        Tuple of (slider_values, lpips_scores, validities)
    """
    with open(lpips_curve_path, "r") as f:
        lines = f.read().split('\n\n')[0].strip().split('\n')

    slider_values = []
    lpips_scores = []
    validities = []

    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split(',')
        if len(parts) == 3:
            slider_values.append(float(parts[0]))
            lpips_scores.append(float(parts[1]))
            validities.append(int(parts[2]))

    return (np.array(slider_values),
            np.array(lpips_scores),
            np.array(validities))


def _is_too_close_to_invalid(
    candidate: float,
    invalid_slider_values: np.ndarray,
    threshold: float
) -> bool:
    """
    Check if candidate is too close to any invalid observation.

    Args:
        candidate: Candidate slider value
        invalid_slider_values: Array of invalid slider values from observations
        threshold: Minimum distance threshold

    Returns:
        True if candidate is within threshold of any invalid point
    """
    if len(invalid_slider_values) == 0:
        return False
    min_distance = np.min(np.abs(invalid_slider_values - candidate))
    return min_distance < threshold


def _generate_candidates(
    target_lpips: float,
    all_slider_values: np.ndarray,
    all_lpips_scores: np.ndarray,
    all_validities: np.ndarray,
    num_nearest: int = 2,
) -> List[float]:
    """
    Generate candidate slider values via nearest-neighbor lookup and local
    linear interpolation, using all observations (valid and invalid).

    Algorithm:
        1. Sort ALL observations by slider value for neighbor lookups.
        2. Find the ``num_nearest`` VALID observations whose LPIPS scores are
           closest to ``target_lpips``.
        3. For each nearest valid point, check its immediate left/right
           neighbors in the sorted order:
           - If the neighbor is VALID and the target LPIPS falls between the
             point and the neighbor, linearly interpolate to get a precise
             candidate slider value.
           - If the neighbor is INVALID, skip interpolation (the interpolated
             value would land in known-bad territory). Use the nearest valid
             point itself instead.
        4. Always include the nearest valid observation as a fallback.
        5. Deduplicate and sort candidates by estimated LPIPS distance.

    Returns:
        List of candidate slider values, sorted by estimated distance to target.
    """
    order = np.argsort(all_slider_values)
    sv_sorted = all_slider_values[order]
    lp_sorted = all_lpips_scores[order]
    vl_sorted = all_validities[order]

    valid_mask = (vl_sorted == 1)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return []

    # Find top-k nearest VALID points by LPIPS distance
    valid_distances = np.abs(lp_sorted[valid_indices] - target_lpips)
    nearest_valid = valid_indices[np.argsort(valid_distances)[:num_nearest]]

    candidates_with_estimates = []
    for idx in nearest_valid:
        for neighbor_idx in [idx - 1, idx + 1]:
            if not (0 <= neighbor_idx < len(sv_sorted)):
                continue
            if vl_sorted[neighbor_idx] != 1:
                continue
            lp_a, lp_b = lp_sorted[idx], lp_sorted[neighbor_idx]
            if lp_a == lp_b:
                continue
            if min(lp_a, lp_b) <= target_lpips <= max(lp_a, lp_b):
                t = (target_lpips - lp_a) / (lp_b - lp_a)
                interp_slider = sv_sorted[idx] + t * (sv_sorted[neighbor_idx] - sv_sorted[idx])
                candidates_with_estimates.append((interp_slider, target_lpips))
        # Always include the nearest valid observation itself
        candidates_with_estimates.append((sv_sorted[idx], lp_sorted[idx]))

    # Deduplicate within tolerance
    unique_candidates = []
    seen_sliders = []
    for slider_val, lpips_est in candidates_with_estimates:
        if not any(abs(slider_val - existing) < 0.01 for existing in seen_sliders):
            unique_candidates.append((slider_val, lpips_est))
            seen_sliders.append(slider_val)

    unique_candidates.sort(key=lambda x: abs(x[1] - target_lpips))
    return [slider_val for slider_val, _ in unique_candidates]


def get_slider_value_from_lpips_percentage(
    lpips_curve_path: str,
    percentage: float,
    view_images_dir: str,
    edited_images_dir: str,
    editing_prompt_pair: Tuple[str, str],
    out_dir: str,
    gpu_ids: List[int] = [0],
    pipelines: Dict[int, object] = None,
    conds: Dict[int, dict] = None,
    pool: GPUWorkerPool = None,
    max_candidates: int = 5,
    invalid_proximity_threshold: float = 0.05,
) -> Tuple[float, float, bool]:
    """
    Find the best valid slider value for a single LPIPS percentage.
    Thin wrapper around ``get_slider_values_from_lpips_percentages``.

    Returns:
        Tuple of (slider_value, actual_lpips_score, is_valid)
    """
    results = get_slider_values_from_lpips_percentages(
        lpips_curve_path=lpips_curve_path,
        percentages=[percentage],
        view_images_dir=view_images_dir,
        edited_images_dir=edited_images_dir,
        editing_prompt_pair=editing_prompt_pair,
        out_dir=out_dir,
        gpu_ids=gpu_ids,
        pipelines=pipelines,
        conds=conds,
        pool=pool,
        min_slider_distance=0.0,
        max_candidates_per_percentage=max_candidates,
        invalid_proximity_threshold=invalid_proximity_threshold,
    )
    return results[0]


def get_slider_values_from_lpips_percentages(
    lpips_curve_path: str,
    percentages: List[float],
    view_images_dir: str,
    edited_images_dir: str,
    editing_prompt_pair: Tuple[str, str],
    out_dir: str,
    gpu_ids: List[int] = [0],
    pipelines: Dict[int, object] = None,
    conds: Dict[int, dict] = None,
    pool: GPUWorkerPool = None,
    min_slider_distance: float = 0.05,
    max_candidates_per_percentage: int = 5,
    invalid_proximity_threshold: float = 0.05,
) -> List[Tuple[float, float, bool]]:
    """
    Find best valid slider values for multiple LPIPS percentages with uniqueness guarantee.

    Args:
        lpips_curve_path: Path to LPIPS curve file
        percentages: List of target percentages (0-100) in LPIPS range
        view_images_dir: Directory containing original view images
        edited_images_dir: Directory containing edited view images
        editing_prompt_pair: (negative_prompt, positive_prompt) for quality verification
        out_dir: Output directory for generated frames
        gpu_ids: GPU device IDs to use
        pipelines: Pre-loaded pipelines (optional)
        conds: Pre-loaded conditions (optional)
        pool: GPU worker pool (optional)
        min_slider_distance: Minimum distance between returned slider values
        max_candidates_per_percentage: Maximum candidates to try per percentage
        invalid_proximity_threshold: Avoid candidates within this distance of invalid points

    Returns:
        List of (slider_value, actual_lpips_score, is_valid) tuples, one per percentage.
        All slider values are guaranteed to be unique (differ by at least min_slider_distance).
    """
    import sys, io

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "slider_selection.log")

    class _Tee:
        """Write to both a file and the original stream."""
        def __init__(self, file, stream):
            self._file = file
            self._stream = stream
        def write(self, msg):
            self._stream.write(msg)
            self._file.write(msg)
        def flush(self):
            self._stream.flush()
            self._file.flush()

    _log_file = open(log_path, "w")
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_log_file, _orig_stdout)

    print(f"\n{'='*80}")
    print(f"Processing {len(percentages)} percentages with uniqueness constraint")
    print(f"Min slider distance: {min_slider_distance}")
    print(f"{'='*80}\n")

    # Parse curve once
    all_slider_vals, all_lpips, all_validities = _parse_lpips_curve(lpips_curve_path)

    if len(all_slider_vals) == 0:
        raise ValueError(f"Empty LPIPS curve file: {lpips_curve_path}")

    # Separate valid and invalid
    valid_mask = (all_validities == 1)
    valid_slider_values = all_slider_vals[valid_mask]
    valid_lpips_scores = all_lpips[valid_mask]
    invalid_slider_values = all_slider_vals[~valid_mask]

    if len(valid_slider_values) == 0:
        raise ValueError("No valid observations in LPIPS curve")

    # Calculate LPIPS range
    min_lpips = np.min(valid_lpips_scores)
    max_lpips = np.max(valid_lpips_scores)

    print(f"LPIPS range: [{min_lpips:.5f}, {max_lpips:.5f}]")
    print(f"Valid observations: {len(valid_slider_values)}, Invalid: {len(invalid_slider_values)}\n")

    # Load models once
    lpips_model = get_lpips_model(f'cuda:{gpu_ids[0]}')

    # Ensure reference frames (slider=0) exist
    ref_dir = os.path.join(out_dir, "search_at_0.00000")
    if not os.path.exists(ref_dir) or not any(
        f.endswith('.png') for f in os.listdir(ref_dir)
    ):
        print(f"Generating reference frames (slider=0)...")
        run_generation_multiprocess(
            view_images_dir, edited_images_dir, [0.0],
            out_dir, gpu_ids, 1,
            pipelines=pipelines, pool=pool, conds=conds
        )

    # Track results and assigned values
    assigned_slider_values = []
    percentage_to_result = {}

    # Sort percentages for orderly processing
    sorted_percentages = sorted(percentages)

    for pct_idx, pct in enumerate(sorted_percentages):
        print(f"\n{'='*80}")
        print(f"[{pct_idx+1}/{len(sorted_percentages)}] Processing percentage: {pct:.2f}%")
        print(f"{'='*80}")

        target_lpips = min_lpips + (max_lpips - min_lpips) * (pct / 100.0)
        print(f"Target LPIPS: {target_lpips:.5f}")

        # Generate candidates
        raw_candidates = _generate_candidates(target_lpips, all_slider_vals, all_lpips, all_validities)

        # Filter by invalid proximity
        candidates = [
            c for c in raw_candidates
            if not _is_too_close_to_invalid(c, invalid_slider_values, invalid_proximity_threshold)
        ]

        # Filter by uniqueness constraint
        available_candidates = [
            c for c in candidates
            if all(abs(c - assigned) >= min_slider_distance for assigned in assigned_slider_values)
        ]

        # If no available candidates, generate perturbations
        if len(available_candidates) == 0 and len(raw_candidates) > 0:
            print(f"⚠️ All candidates conflict with assigned/invalid values, generating perturbations...")
            original = raw_candidates[0]
            perturbations = []
            for delta in np.arange(0.05, 1.0, 0.05):
                for sign in [-1, 1]:
                    perturbed = original + sign * delta
                    if (all(abs(perturbed - assigned) >= min_slider_distance
                            for assigned in assigned_slider_values)
                        and not _is_too_close_to_invalid(
                            perturbed, invalid_slider_values, invalid_proximity_threshold)):
                        perturbations.append(perturbed)
                        if len(perturbations) >= max_candidates_per_percentage * 2:
                            break
                if len(perturbations) >= max_candidates_per_percentage * 2:
                    break
            available_candidates = perturbations[:max_candidates_per_percentage]

        # Limit candidates
        available_candidates = available_candidates[:max_candidates_per_percentage]

        if len(available_candidates) == 0:
            print(f"❌ No available candidates after filtering!")
        else:
            print(f"Available candidates: {[f'{c:.5f}' for c in available_candidates]}")

        # Try to validate candidates
        found_valid = False
        for cand_idx, candidate in enumerate(available_candidates):
            print(f"\n  [{cand_idx+1}/{len(available_candidates)}] Testing candidate: {candidate:.5f}")

            # Generate frames and GLB if needed
            frames_dir = os.path.join(out_dir, f"search_at_{candidate:.5f}")
            has_frames = os.path.exists(frames_dir) and any(f.endswith('.png') for f in os.listdir(frames_dir))
            has_glb = os.path.exists(os.path.join(frames_dir, "output.glb")) if has_frames else False
            if not has_frames or not has_glb:
                print(f"    Generating frames...")
                run_generation_multiprocess(
                    view_images_dir, edited_images_dir, [candidate],
                    out_dir, gpu_ids, 1,
                    pipelines=pipelines, pool=pool, conds=conds
                )
            else:
                print(f"    Frames already exist")

            # Compute LPIPS
            actual_lpips = get_LPIPS_score_relative_to_reference(
                lpips_model, 0, [candidate], out_dir
            )[0]
            print(f"    Actual LPIPS: {actual_lpips:.5f} (target: {target_lpips:.5f})")

            # Verify quality
            passed, reasoning = get_quality_change_verification(
                editing_prompt_pair, 0, [candidate], out_dir
            )

            if passed[0]:
                print(f"    ✅ Validation PASSED")
                percentage_to_result[pct] = (candidate, actual_lpips, True)
                assigned_slider_values.append(candidate)
                found_valid = True
                break
            else:
                print(f"    ❌ Validation FAILED: {reasoning[0][:100]}...")

        # Fallback if no candidate validated
        if not found_valid:
            print(f"\n  ⚠️ All candidates failed, using fallback...")
            # Find closest valid observation not yet assigned
            valid_not_assigned = [
                (sv, ls) for sv, ls in zip(valid_slider_values, valid_lpips_scores)
                if all(abs(sv - assigned) >= min_slider_distance for assigned in assigned_slider_values)
            ]

            if len(valid_not_assigned) > 0:
                # Find closest by LPIPS
                lpips_diffs = [abs(ls - target_lpips) for _, ls in valid_not_assigned]
                best_idx = np.argmin(lpips_diffs)
                fallback_slider, fallback_lpips = valid_not_assigned[best_idx]
                percentage_to_result[pct] = (fallback_slider, fallback_lpips, False)
                assigned_slider_values.append(fallback_slider)
                print(f"  Fallback: {fallback_slider:.5f} (LPIPS: {fallback_lpips:.5f})")
            else:
                print(f"  ❌ No unique value available!")
                percentage_to_result[pct] = (None, None, False)

    # Return results in original order
    results = [percentage_to_result[pct] for pct in percentages]

    # Final uniqueness check
    assigned_vals = [r[0] for r in results if r[0] is not None]
    unique_vals = set()
    for v in assigned_vals:
        # Check minimum distance constraint
        if any(abs(v - existing) < min_slider_distance for existing in unique_vals):
            print(f"\n⚠️ WARNING: Uniqueness constraint violated for value {v:.5f}")
        unique_vals.add(v)

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    for pct, (sv, lpips, valid) in zip(percentages, results):
        status = "✅" if valid else "⚠️"
        if sv is not None:
            print(f"{status} {pct:6.2f}% → slider: {sv:8.5f}, LPIPS: {lpips:8.5f}")
        else:
            print(f"❌ {pct:6.2f}% → FAILED (no unique value)")
    print(f"{'='*80}\n")

    with open(os.path.join(out_dir, "final_slider_values.csv"), "w") as f:
        for sv, lpips, valid in results:
            f.write(f"{sv:.5f},{lpips:.5f},{valid}\n")

    sys.stdout = _orig_stdout
    _log_file.close()
    print(f"Log saved to {log_path}")

    return results


if __name__ == "__main__":
    # view_images_dir = "/home/rwang/TRELLIS/validation/multi_views_new_dataset_40/obv_man_963f1b5f11994709b06d5330cb6baafd_views"
    # edited_images_dir = "/home/rwang/TRELLIS/validation/new_val_edited_views_RM_edprp_mv40_verify_gpt52/obv_man_963f1b5f11994709b06d5330cb6baafd_edited_views_15"
    # out_dir = '/data/ru_data/results/trellis_output/validation/test_slider_bound_search'
    # info_dir = '/data/ru_data/results/trellis_output/validation/test_slider_mapping/obv_man_963f1b5f11994709b06d5330cb6baafd'
    # out_dir = os.path.join(out_dir, os.path.basename(info_dir) + "_15")
    # os.makedirs(out_dir, exist_ok=True)
    # generate_new_sliders_from_lpips_curve(
    #     view_images_dir,
    #     edited_images_dir,
    #     os.path.join(out_dir, "lpips_curve.csv"),
    #     out_dir,
    #     gpu_ids=[0, 1, 2, 3],
    #     n_samples=9,
    #     workers_per_gpu=2,
    # )

    # get_lpips_curve(
    #     view_images_dir,
    #     edited_images_dir,
    #     [3.59375, -5.46875],
    #     out_dir,
    #     gpu_ids=[0, 1, 2, 3],
    #     n_samples=20,
    #     workers_per_gpu=5,
    # )

    # # read unaffected_questions and unaffected_dependencies from file
    # json_info = json.load(open(os.path.join(info_dir, "info_15.json")))
    # unaffected_questions = json_info["filtered_questions"]
    # # clean \u2019 from unaffected_questions
    # unaffected_questions = [q.replace("\u2019", "'") for q in unaffected_questions]
    # unaffected_dependencies = json_info["dependencies"]

    # # weights, question_ids = compute_subtree_weights(
    # #     json_info["questions"], unaffected_dependencies, unaffected_questions,
    # # )

    # # print("Subtree weights:")
    # # for q, w, qid in zip(unaffected_questions, weights, question_ids):
    # #     print(f"  [{qid:>3}] w={w:.4f}  {q}")

    # pos_boundary, neg_boundary = find_boundary_binary_search(
    #     view_images_dir, edited_images_dir, unaffected_questions, unaffected_dependencies,
    #     gpu_ids=[0,1],
    #     explore_growth=1, # uniform explore at the beginning
    #     threshold=0.45,
    #     out_dir=out_dir,
    #     # boundary_threshold=1/len(unaffected_questions),
    # )
    # print(f"Positive boundary: {pos_boundary}, Negative boundary: {neg_boundary}")

    percentages = np.linspace(0, 100, 9)

    results = get_slider_values_from_lpips_percentages(
        lpips_curve_path="/data/ru_data/results/trellis_output/validation/test_lpips_curve/obv_toy_0278450d1e324c33904e160e115f6fbd_2/lpips_curve_gradient_9.txt",
        percentages=percentages,
        view_images_dir="/home/rwang/TRELLIS/validation/multi_views_new_dataset_40/obv_toy_0278450d1e324c33904e160e115f6fbd_views",
        edited_images_dir="/home/rwang/TRELLIS/validation/new_val_edited_views_RM_edppv2_mv40_verify_gpt52/obv_toy_0278450d1e324c33904e160e115f6fbd_edited_views_2",
        editing_prompt_pair=("Make the lego figure's head extremely small and compact.", "Make the lego figure's head extremely large and oversized."),
        out_dir="/data/ru_data/results/trellis_output/validation/test_lpips_curve/obv_toy_0278450d1e324c33904e160e115f6fbd_2",
        gpu_ids=[1,2,3],
        min_slider_distance=0.05,  # All values will differ by at least 0.05
    )

  # Results: [(slider_value, actual_lpips, is_valid), ...]