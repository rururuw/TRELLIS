import os
import torch
import shutil
from typing import *
from PIL import Image
import csv
from dotenv import load_dotenv
import base64
import io
import time
load_dotenv()
import requests
# Use default flash_attn (don't set xformers - it's incompatible with this version)
# os.environ['ATTN_BACKEND'] = 'xformers'     # xformers missing BlockDiagonalMask
os.environ['SPCONV_ALGO'] = 'native'          # REQUIRED: 'auto' causes cudaErrorIllegalAddress
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import numpy as np
import imageio
import open3d as o3d
from diffusers import StableDiffusionXLPipeline, Flux2Pipeline, DiffusionPipeline, Flux2KleinPipeline
from diffsynth_engine import fetch_model, QwenImagePipeline, QwenImagePipelineConfig

from openai import OpenAI
from pydantic import TypeAdapter, ValidationError
from trellis.pipelines import TrellisTextTo3DPipeline, TrellisAttributeSliderPipeline, TrellisImageTo3DAttributeSliderPipeline
from trellis.pipelines.base import Pipeline
from trellis.utils import render_utils, postprocessing_utils
import trellis.models as models
import trellis.modules.sparse as sp
from torchvision import transforms
import torch.nn.functional as F
import utils3d
import json
from subprocess import DEVNULL, call
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
from transformers import CLIPProcessor, CLIPModel


def get_sdxl_pipeline():
    _sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    )
    _sdxl_pipeline.to("cuda")
    return _sdxl_pipeline

def get_flux2_pipeline():
    _flux2_pipeline = Flux2Pipeline.from_pretrained(
        "black-forest-labs/FLUX.2-dev",
        torch_dtype=torch.bfloat16
    )
    _flux2_pipeline.to("cuda")
    _flux2_pipeline.load_lora_weights(
        "fal/FLUX.2-dev-Turbo", 
        weight_name="flux.2-turbo-lora.safetensors"
    )
    return _flux2_pipeline

def get_qwen_pipeline():
    _qwen_pipeline = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image-2512",
        torch_dtype=torch.bfloat16
    )
    _qwen_pipeline.to("cuda")
    return _qwen_pipeline

def get_flux2_klein_pipeline():
    _flux2_klein_pipeline = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B",
        torch_dtype=torch.bfloat16
    )
    _flux2_klein_pipeline.enable_model_cpu_offload()
    # _flux2_klein_pipeline.to("cuda")
    return _flux2_klein_pipeline

def get_qwen_lora_pipeline():
    # loading models:
    model_path=fetch_model("Qwen/Qwen-Image-2512", path="transformer/*.safetensors")
    encoder_path=fetch_model("Qwen/Qwen-Image-2512", path="text_encoder/*.safetensors")
    vae_path=fetch_model("Qwen/Qwen-Image-2512", path="vae/*.safetensors")
    # print("model_path:", model_path)
    # print("encoder_path:", encoder_path)
    # print("vae_path:", vae_path)
    config = QwenImagePipelineConfig.basic_config(
        model_path=model_path,
        encoder_path=encoder_path,
        vae_path=vae_path,
        offload_mode="cpu_offload",
        device="cuda",
    )
    pipe = QwenImagePipeline.from_pretrained(config)
    # Load our turbo LoRA
    pipe.load_lora(
        path=fetch_model("Wuli-Art/Qwen-Image-2512-Turbo-LoRA", path="Wuli-Qwen-Image-2512-Turbo-LoRA-4steps-V1.0-bf16.safetensors"),
        scale=1.0,
        fused=True,
    )

    # Change scheduler config
    scheduler_config = {
        "exponential_shift_mu": math.log(2.5),
        "use_dynamic_shifting": True,
        "shift_terminal": None
    }
    pipe.apply_scheduler_config(scheduler_config)

    return pipe

def get_openai_client():
    client = OpenAI() # defaults to os.environ.get("OPENAI_API_KEY")
    return client

def generate_image_sdxl(sdxl_pipeline: StableDiffusionXLPipeline, prompt: str):
    # use SDXL to generate image, 512x512 resolution, 4k quality
    image = sdxl_pipeline(
        prompt,
        num_inference_steps=50,
        guidance_scale=4,
        width=1024,
        height=1024,
    ).images[0]
    return image

def edit_image_flux2(flux2_pipeline: Flux2Pipeline, input_image: Image, edit_prompt: str):
    # Pre-shifted custom sigmas for 8-step turbo inference
    TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]
    image = flux2_pipeline(
        prompt=edit_prompt,
        image=input_image,
        sigmas=TURBO_SIGMAS,
        num_inference_steps=8,
        guidance_scale=2.5,
        width=1024,
        height=1024,
    ).images[0]
    return image

def generate_image_qwen(qwen_pipeline: DiffusionPipeline, prompt: str):
    negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1104),
        "3:4": (1104, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }

    width, height = aspect_ratios["1:1"]

    image = qwen_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cuda").manual_seed(42)
    ).images[0]

    return image

def generate_image_qwen_lora(qwen_lora_pipeline: QwenImagePipeline, prompt: str):
    image = qwen_lora_pipeline(
        prompt=prompt,
        cfg_scale=1,
        num_inference_steps=8,  # 8 is also recommended
        seed=42,
        width=1328,
        height=1328
    )
    return image

def edit_image_flux2_klein(flux2_klein_pipeline: Flux2KleinPipeline, input_image: Image, edit_prompt: str):
    # Generate edited image
    image = flux2_klein_pipeline(
        prompt=edit_prompt,
        image=input_image,  # Your input image
        height=1024,
        width=1024,
        guidance_scale=1.0,  # Lower guidance for editing
        num_inference_steps=4,
        generator=torch.Generator(device="cuda").manual_seed(0)
    ).images[0]
    return image

def edit_image_gpt_image(client: OpenAI, input_image_path: str, edit_prompt: str):
    # Edit the image using GPT-Image
    result = client.images.edit(
        model="gpt-image-1.5",
        image=[
            open(input_image_path, "rb"),
        ],
        prompt=edit_prompt
    )
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    # convert image_bytes to Image
    image = Image.open(io.BytesIO(image_bytes))
    return image

def edit_image_gemini(input_image_path: str, edit_prompt: str):
    
    ENDPOINT = "https://api.ai-service.global.fujitsu.com/ai-foundation/chat-ai/gemini/flash:generateContent"
    # get key from env
    API_KEY = os.environ.get("GEMINI_API_KEY") 
    headers = {
        "Content-type": "application/json",
        "api-key": API_KEY
    }
    messages = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    # {
                    #     "inlineData": {
                    #         "mimeType": "image/jpeg",
                    #         "data": base64.b64encode(open(input_image_path, "rb").read()).decode("utf-8")
                    #     }
                    # },
                    # {
                    #     "text": 'Can you' + edit_prompt + "?"
                    # }
                    {
                        "text": "Can you do image editing if i send you a image and a prompt?"
                    }
                ]
            }
        ],
        # "generationConfig": {
        #     "responseMimeType": "image/jpeg"
        # }
    }
    response = requests.post(ENDPOINT, headers=headers, json=messages)
    print(response.json())

def get_CLIP_model_and_processor(device="cuda"):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    model = model.to(device).eval()
    return model, processor

def calc_CLIP_similarity_text_vs_images(model, processor, text_prompt, images: List[Image.Image], device="cuda"):
    # Batch process all images at once
    inputs = processor(text=[text_prompt], images=images, return_tensors="pt", padding=True)
    # Move inputs to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # logits_per_image shape: [num_images, 1] - similarity of each image to the single text
    clip_scores = outputs.logits_per_image[:, 0].detach().cpu().numpy()
    return np.mean(clip_scores)

def select_best_views(model, processor, view_image_dir: str, edit_prompt: str, device="cuda"):
    view_image_paths = [os.path.join(view_image_dir, f) for f in os.listdir(view_image_dir) if f.endswith(".png")]
    view_images = [Image.open(f) for f in view_image_paths]
    inputs = processor(text=[edit_prompt], images=view_images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # logits_per_image shape: [num_images, 1] - similarity of each image to the single text
    clip_scores = outputs.logits_per_image[:, 0].detach().cpu().numpy()
    # return the index of views ranked by the clip scores
    best_views_indices = np.argsort(clip_scores)[::-1]
    # for i in best_views_indices:
    #     print(f"View {view_image_paths[i]} similarity: {clip_scores[i]}")
    # select the top 10 best views
    best_views = [view_image_paths[i] for i in best_views_indices]
    return best_views

def _safe_filename(prefix: str, prompt: str, ext: str = ".png", max_len: int = 200) -> str:
    sanitized = prompt.replace(" ", "_").replace(",", "-")
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    base = f"{prefix}_{sanitized}_{digest}{ext}"
    if len(base) <= max_len:
        return base
    truncated = sanitized[: max(0, max_len - len(prefix) - len(digest) - len(ext) - 2)]
    return f"{prefix}_{truncated}_{digest}{ext}"

def _parse_direction_groups(output_text: str, expected_count: int) -> List[List[str]]:
    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Expected JSON output, got: {output_text}") from exc

    if not isinstance(payload, dict) or "groups" not in payload:
        raise ValueError(f"Expected JSON object with 'groups', got: {payload}")

    try:
        adapter = TypeAdapter(List[Tuple[str, str, str]])
        groups = adapter.validate_python(payload["groups"])
    except ValidationError as exc:
        raise ValueError(f"Invalid groups format: {payload}") from exc

    if len(groups) != expected_count:
        raise ValueError(f"Expected {expected_count} groups, got {len(groups)}")

    return [list(item) for item in groups]

def affected_view_prompt_generator(basic_object: str, image_editing_command: str,):
    # use openai api to generate direction
    client = OpenAI() # defaults to os.environ.get("OPENAI_API_KEY")
    sys_prompt = (
        "You are a helpful assistant to generate prompts to describe the affected part of the object in the image, given a 3D model of object A and an image editing command B. "
        "your goal is to generate a description of the part of the object A that is to be affected by the image editing command B. \n\n"
        "For example, if the object A is 'a blue sofa made of leather' and the image editing command B is 'very modern and minimalist style', "
        "The description should be 'a blue sofa made of leather in the middle of the image', "
        "because in order to edit the image with command B, we need to change the style of the sofa to be very modern and minimalist style, "
        "and the entire sofa should be edited to be very modern and minimalist style.\n\n"
        "Another example, if the object A is 'a woman, full body' and the image editing command B is 'very small and closed eyes', "
        "The description should be 'the woman's eyes in the middle of the image', "
        "because in order to edit the image with command B, we need to change the size of the woman's eyes to be very small and closed, "
        "and the woman's eyes are the only part of the object A that can be edited to be very small and closed.\n\n"
        "Another example, if the object A is 'a man, full body' and the image editing command B is 'wearing very warm and thick clothes', "
        "The description should be 'the man's body in the middle of the image', "
        "because in order to edit the image with command B, we need to change the clothes of the man to be very warm and thick, "
        "and the entire body of the man should be edited to be very warm and thick.\n\n"
        "Return the description as a string."
    )
    user_prompt = (
        f"Object: {basic_object} \n"
        f"Image editing command: {image_editing_command} \n"
    )
    description = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
    ).choices[0].message.content
    return description

def direction_generator_edit(basic_object: str, basic_attribute: str,):
    # use openai api to generate direction
    client = OpenAI() # defaults to os.environ.get("OPENAI_API_KEY")
    sys_prompt = (
        "You are a helpful assistant to generate prompts for image editing. Given an object A and an attribute B, "
        "your goal is to generate a negative image editing prmopt and a positive image editing prompt that can be used to edit the image of the object A with the attribute B.\n\n"
        "The negative image editing prompt should be editing the object A such that it presents the negative extreme of attribute B, "
        "and the positive image editing prompt should be editing the object A such that it presents the positive extreme of attribute B. \n\n"
        "Note that attribute B can itself be negative, neutral, or positive, and this should not affect the generation of the negative and positive image editing prompts.\n\n"
        "For example, if the object A is 'a blue sofa made of leather' and the attribute B is 'modern style', "
        "the negative image editing prompt should be 'make the sofa very traditional and ornate style', "
        "and the positive image editing prompt should be 'make the sofa very modern and minimalist style'.\n"
        "Another example, if the object A is 'a woman, full body' and the attribute B is 'eye size', "
        "the negative image editing prompt should be 'make the woman's eyes very small and closed', "
        "and the positive image editing prompt should be 'make the woman's eyes very large and open'.\n"
        "Another example, if the object A is 'a woman' and the attribute B is 'wearing very warm and thick clothes', "
        "the negative image editing prompt should be 'make the woman wear very thin and light clothes', "
        "and the positive image editing prompt should be 'make the woman wear very warm and thick clothes'.\n\n"
        "Return JSON with the schema: {\"prompts\": [negative_prompt, positive_prompt]} where negative_prompt and positive_prompt are strings."
    )
    user_prompt = (
        f"Object: {basic_object} \n"
        f"Attribute: {basic_attribute} \n"
    )
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.3,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "direction_edit_prompts",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "prompts": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["prompts"],
                },
            },
        },
    )
    payload = json.loads(response.choices[0].message.content)
    prompts = payload["prompts"]
    assert len(prompts) == 2, f"Expected 2 prompts, got {len(prompts)}"
    return prompts  # [negative_prompt, positive_prompt]

def direction_generator(basic_object: str, basic_positive_prompt: str, num_groups: int, for_text: bool = False):
    # use openai api to generate direction
    client = OpenAI() # defaults to os.environ.get("OPENAI_API_KEY")

    # v5 prompt
    sys_prompt = "You are a helpful assistant to generate prompts for image editing. Given an object and a basic positive attribute (e.g., 'a wooden chair, very modern style'), " + \
        "your goal is to generate a list of prompt groups. To generate each group, you need to follow the instruction below:\n" + \
        "1. identify the basic object O (e.g., 'a chair') by removing all discriptives (e.g., color, material, style, etc.) from the given object. \n" + \
        "2. based on the basic positive attribute, generate a simple but concrete, description of the positive attribute to the maximum extent specific to the basic object O using less than 10 words (P). \n" + \
        "3. based on the basic positive attribute, generate a simple but concrete, description of the completely opposite attribute of that positive attribute to the maximum extent specific to the basic object O using less than 10 words (N). \n" + \
        "4. generate two attributes A1 and A2 in different dimensions that can describe the basic object, but perpendicular and irrelevant to the positive attribute P and the negative attribute N. \n\n" + \
        f"Following the above steps, generate {num_groups} groups of prompts, each group containing a neutral prompt T0, a positive prompt P and a negative prompt N. T0 consists of the basic object O, A1, and A2. \n" + \
        ("the elements in the neutral prompt T0 should be organized such that the resulting image will be a picture with only the basic object (with corresponding attributes) in the middle with a black background. \n\n" if not for_text else "\n") + \
        "For example, if the basic object is 'a blue sofa made of leather' and the positive attribute is 'modern style', a possible output group can be: \n" + \
        ("['a picture with only a sofa in the middle with a black background, the sofa is yellow, tall and narrow', 'the sofa is very modern and minimalist style', 'the sofa is very traditional and ornate style']\n\n" if not for_text else \
        "['a yellow, tall and narrow sofa', 'the sofa is very modern and minimalist style', 'the sofa is very traditional and ornate style']\n\n")  + \
        "Explanation: \n" + \
        "1. 'a sofa' is O, the basic object by removing the discriptives from the given object. \n" + \
        "1. 'the sofa is very modern and minimalist style' is P, the positive attribute description to the maximum extent specific to the basic object O. \n" + \
        "2. 'the sofa is very traditional and ornate style' is N, the negative attribute description to the maximum extent specific to the basic object O. \n" + \
        "3. 'yellow color' and 'tall and narrow' are A1 and A2, the attributes that are perpendicular and irrelevant to the positive/negative attribute. \n\n" + \
        "Return JSON with the schema: {\"groups\": [[T0, P, N], ...]} where T0, P, and N are strings.\n"



    results = []
    user_prompt = (
        "Please generate the list of prompt groups for the following basic object and basic positive attribute: \n"
        f"Basic object: '{basic_object}' \n"
        f"Basic positive attribute: '{basic_positive_prompt}' \n"
    )
    print("sys_prompt:", sys_prompt)
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.3,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "direction_groups",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "groups": {
                            "type": "array",
                            "minItems": num_groups,
                            "maxItems": num_groups,
                            "items": {
                                "type": "array",
                                "minItems": 3,
                                "maxItems": 3,
                                "items": {"type": "string"},
                            },
                        }
                    },
                    "required": ["groups"],
                },
            },
        },
    )
    groups = _parse_direction_groups(response.choices[0].message.content, num_groups)
    results.extend(groups)
    return results

def get_image_pairs_from_object_and_pos_attributes(img_gen_pipeline, img_gen_method: Callable, obj_name: str, pos_attribute: str, num_pairs: int, out_dir: str, for_text: bool = False):
    # use sdxl to generate image for the object and the attributes
    subfolder = f"{img_gen_method.__name__}_{obj_name.replace(' ', '_')}_{pos_attribute.replace(' ', '_')}"
    os.makedirs(os.path.join(out_dir, subfolder), exist_ok=True)
    file_prompt_map = {}
    slider_prompts = direction_generator(obj_name, pos_attribute, num_pairs, for_text=for_text)
    for i, slider_prompt in enumerate(slider_prompts):
        img_gen_neutral_prompt, pos_prompt, neg_prompt = slider_prompt
        # print('Generating image pairs for:', slider_prompt)
        img_gen_pos_prompt = f"{img_gen_neutral_prompt}, {pos_prompt}, 4k"
        img_gen_neg_prompt = f"{img_gen_neutral_prompt}," + (f" {neg_prompt}, 4k" if neg_prompt else " 4k")
        print('Generating positive image for:', img_gen_pos_prompt)
        img_gen_pos_img = img_gen_method(img_gen_pipeline, img_gen_pos_prompt)
        img_gen_pos_img.save(os.path.join(out_dir, subfolder, f"{i}_pos.png"))
        print('Generating negative image for:', img_gen_neg_prompt)
        img_gen_neg_img = img_gen_method(img_gen_pipeline, img_gen_neg_prompt)
        img_gen_neg_img.save(os.path.join(out_dir, subfolder, f"{i}_neg.png"))
        file_prompt_map[f"{i}_pos.png"] = img_gen_pos_prompt
        file_prompt_map[f"{i}_neg.png"] = img_gen_neg_prompt
    
    # output the file_prompt_map to a tsv file
    with open(os.path.join(out_dir, subfolder, "img_gen_prompt_map.tsv"), "w", encoding='utf-8') as f:
        for file, prompt in file_prompt_map.items():
            f.write(f"{file}\t{prompt}\n")

def batch_get_image_pairs_from_object_and_pos_attributes(img_gen_pipeline: Callable, img_gen_method: Callable, obj_name_pos_attribute_pairs: List[Tuple[str, str]], num_pairs: int, out_dir: str, for_text: bool = False):
    for obj_name, pos_attribute in obj_name_pos_attribute_pairs:
        get_image_pairs_from_object_and_pos_attributes(img_gen_pipeline, img_gen_method, obj_name, pos_attribute, num_pairs, out_dir, for_text=for_text)

def get_image_to_3d_single_attribute_slider_pipeline():
    pipeline = TrellisImageTo3DAttributeSliderPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    return pipeline

def get_text_to_3d_single_attribute_slider_pipeline():
    """Load pipeline on default GPU (cuda:0)."""
    pipeline = TrellisAttributeSliderPipeline.from_pretrained("microsoft/TRELLIS-text-large")
    pipeline.cuda()  # Uses default GPU (cuda:0)
    # Keep FP16 (default) - flash_attn requires FP16
    return pipeline

def merge_videos_and_save(videos, out_path):
    slider_values = videos.keys()
    # save videos for negatives and positives separately
    neg_values = [slider_value for slider_value in slider_values if slider_value <= 0]
    neg_values.sort(reverse=True) # 0, -1, -2, -3, -4, -5
    pos_values = [slider_value for slider_value in slider_values if slider_value >= 0]
    pos_values.sort() # 0, 1, 2, 3, 4, 5
    # merge videos for negatives and positives
    videos_row_one = [videos[slider_value] for slider_value in neg_values]
    merged_row_one = np.concatenate(videos_row_one, axis=2)
    videos_row_two = [videos[slider_value] for slider_value in pos_values]
    merged_row_two = np.concatenate(videos_row_two, axis=2)
    merged_videos = np.concatenate([merged_row_one, merged_row_two], axis=1)
    imageio.mimsave(out_path, merged_videos, fps=25)

def edit_3d_assets_w_text(
        pipeline: TrellisAttributeSliderPipeline, 
        cond: dict,
        seed: int = 42,
        out_dir: str = "outputs/test_img_slat",
        cfg_strength: float = 3,
    ):
    os.makedirs(out_dir, exist_ok=True)
    # Device is now set when loading the pipeline - don't change it here

    # slider_values = np.linspace(-10, 10, 11)
    # extreme = 20
    # slider_values = np.linspace(-extreme, extreme, extreme * 2 + 1)
    # slider_values = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
    slider_values = np.linspace(-5, 5, 11)
    for slider_value in slider_values:
        outputs = pipeline.run_variant_condition_to_ss_slat(
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
        )
        # Render the outputs
        video_gs = render_utils.render_video(outputs['gaussian'][0], bg_color=(.5, .5, .5))['color']
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
        imageio.mimsave(os.path.join(out_dir, f"text_slat_editing_video_{slider_value}.mp4"), video, fps=25)


def edit_3d_assets_w_images(
        pipeline: TrellisImageTo3DAttributeSliderPipeline, 
        cond: dict,
        seed: int = 42,
        out_dir: str = "outputs/test_img_slat",
        cfg_strength: float = 3,
    ):
    os.makedirs(out_dir, exist_ok=True)
    # slider_values = np.linspace(-10, 10, 11)
    extreme = 5
    slider_values = np.linspace(-extreme, extreme, extreme * 2 + 1)
    # slider_values = [-5, 0, 5]
    videos = {}
    for slider_value in slider_values:
        outputs = pipeline.run_reconstruct_edit(
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
        )
        # Render the outputs
        video_gs = render_utils.render_video(outputs['gaussian'][0], bg_color=(.5, .5, .5))['color']
        videos[slider_value] = video_gs
        # video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        # video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
        # imageio.mimsave(os.path.join(out_dir, f"img_slat_editing_video_{slider_value}.mp4"), video, fps=25)
    print('Merging videos and saving...')
    merge_videos_and_save(videos, os.path.join(out_dir, "img_slat_editing_video_merged.mp4"))

def batch_processing_validation_dataset():
    pipeline = get_image_to_3d_single_attribute_slider_pipeline()
    task_seq = []
    with open("../3d-interior-val/new_validation.csv", "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip the header
        for row in reader:
            basic_description, attributes, asset_name = row
            print('adding task: object:', basic_description, 'with attributes:', attributes, 'asset name:', asset_name)
            task_seq.append((basic_description, attributes, asset_name))
    out_dir = "validation/editing_videos_new_dataset"
    views_dir = "validation/multi_views_new_dataset"
    image_pairs_dir = "outputs/test_img_slat/img_direction_pairs_new_dataset"
    for basic_description, attributes, asset_name in task_seq:
        print('processing task: object:', basic_description, 'with attributes:', attributes, 'asset name:', asset_name)
        asset_views_dir = os.path.join(views_dir, asset_name[:-4] + "_views")
        view_images = [Image.open(os.path.join(asset_views_dir, f)) for f in os.listdir(asset_views_dir) if f.endswith(".png")]
        direction_images_dir = os.path.join(image_pairs_dir, 'generate_image_qwen_lora_' + basic_description.replace(' ', '_') + "_" + attributes.replace(' ', '_'))
        positive_image_paths = [os.path.join(direction_images_dir, f) for f in os.listdir(direction_images_dir) if f.endswith("_pos.png")]
        negative_image_paths = [os.path.join(direction_images_dir, f) for f in os.listdir(direction_images_dir) if f.endswith("_neg.png")]
        
        positive_image_paths.sort(key=lambda x: int(os.path.basename(x).split("_")[0]))
        negative_image_paths.sort(key=lambda x: int(os.path.basename(x).split("_")[0]))
        positive_images = [Image.open(path) for path in positive_image_paths]
        negative_images = [Image.open(path) for path in negative_image_paths]
        print('Editing 3D assets with images for:', basic_description, 'with attributes:', attributes, 'asset name:', asset_name)
        cond = pipeline.get_reconstruct_edit_cond(view_images, positive_images, negative_images)
        edit_3d_assets_w_images(pipeline, cond, out_dir=os.path.join(out_dir, os.path.basename(direction_images_dir)), cfg_strength=3)
    print('Done processing all tasks')

def test_gemini_api():
    edit_prompt = "make the cat very chubby and round"
    concluding_prompt = ', without changing any other irrelevant attributes of the object and its background'
    input_image_path = "/home/rwang/TRELLIS/validation/multi_views_new_dataset/obv_cat_cc71d870acf84ca382545962fc0773c3_views/017.png"
    edit_image_gemini(input_image_path, edit_prompt + concluding_prompt)

def test_select_best_views_and_edit():
    model, processor = get_CLIP_model_and_processor()
    # pipe = get_flux2_klein_pipeline()
    pipe = get_flux2_pipeline()
    out_dir = "/home/rwang/TRELLIS/validation/test_img_edit"
    view_image_dirs = [
        "/home/rwang/TRELLIS/validation/multi_views_new_dataset/obv_cat_cc71d870acf84ca382545962fc0773c3_views", 
        "/home/rwang/TRELLIS/validation/multi_views_new_dataset/obv_piano_294ad9f9f0bb4223b8a71c2aec6f7104_views",
        "/home/rwang/TRELLIS/validation/multi_views_new_dataset/obv_dog_be6aa66db20943fcad3af3d200607870_views",
        "/home/rwang/TRELLIS/validation/multi_views_new_dataset/obv_woman_head_7961481ba1324707be7fe34c9d8ca5f7_views"        
    ]

    edit_prompts = [
        "make the cat very chubby and round",
        "make the piano very futuristic and sleek",
        "make the dog's tail very long",
        "make the woman's face very happy and smiling"
    ]

    concluding_prompt = ', without changing any other irrelevant attributes of the object and its background'
    client = get_openai_client()
    # total_time_f2k9 = 0
    # total_time_gpt = 0
    total_time_flux2 = 0
    num_imgs_processed = 0
    for view_image_dir, edit_prompt in zip(view_image_dirs, edit_prompts):
        print('Selecting best views for:', edit_prompt)
        best_views = select_best_views(model, processor, view_image_dir, edit_prompt)
        best_views = best_views[:5] # select the top 5 best views
        print('Editing with best views for:', edit_prompt)
        sub_out_dir = os.path.join(out_dir, edit_prompt.replace(" ", "_"))
        os.makedirs(sub_out_dir, exist_ok=True)
        for i, view_path in enumerate(best_views):
            start_time = time.time()
            input_image = Image.open(view_path)
            edited_image = edit_image_flux2(pipe, input_image, edit_prompt + concluding_prompt)
            edited_image.save(os.path.join(sub_out_dir, f"flux2_best_view_{i}_edited.png"))
            end_time = time.time()
            total_time_flux2 += end_time - start_time
            print(f"Time taken for Flux2-Turbo: {end_time - start_time} seconds")
            # edited_image = edit_image_flux2_klein(pipe, Image.open(view_path), edit_prompt + concluding_prompt)
            # edited_image.save(os.path.join(sub_out_dir, f"f2k9_best_view_{i}_edited.png"))
            # end_time = time.time()
            # total_time_f2k9 += end_time - start_time
            # print(f"Time taken for Flux2Klein 9B: {end_time - start_time} seconds")
            # start_time = time.time()
            # edited_image_gpt = edit_image_gpt_image(client, view_path, edit_prompt + concluding_prompt)
            # edited_image_gpt.save(os.path.join(sub_out_dir, f"gpt_best_view_{i}_edited.png"))
            # end_time = time.time()
            # total_time_gpt += end_time - start_time
            # print(f"Time taken for GPT image 1.5: {end_time - start_time} seconds")
            num_imgs_processed += 1
    print(f"Total time taken for Flux2-Turbo: {total_time_flux2} seconds")
    print(f"Number of images processed: {num_imgs_processed}")
    print(f"Average time taken per image for Flux2-Turbo: {total_time_flux2 / num_imgs_processed} seconds")

def generating_image_pairs_from_object_and_attribute_new_validation(validation_file: str, out_dir: str):
    task_seq = []
    with open(validation_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip the header
        for row in reader:
            basic_description, attributes, asset_name = row            
            print('adding task: object:', basic_description, 'with attributes:', attributes, 'asset name:', asset_name)
            task_seq.append((basic_description, attributes, asset_name))
    os.makedirs(out_dir, exist_ok=True)
    prompt_pairs = []
    for basic_description, attributes, asset_name in task_seq:
        print('processing task: object:', basic_description, 'with attributes:', attributes, 'asset name:', asset_name)
        neg_prompt, pos_prompt = direction_generator_edit(basic_description, attributes)
        print('negative prompt:', neg_prompt, 'positive prompt:', pos_prompt)
        prompt_pairs.append((neg_prompt, pos_prompt))
    with open(os.path.join(out_dir, "new_val_prompt_pairs.tsv"), "w", encoding='utf-8') as f:
        for neg_prompt, pos_prompt in prompt_pairs:
            f.write(f"{neg_prompt}\t{pos_prompt}\n")
    print('Done generating image pairs for all tasks')

def generating_affected_view_prompts_from_object_and_attribute_new_validation(validation_file: str, out_dir: str):
    task_seq = []
    with open(validation_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip the header
        for row in reader:
            basic_description, attributes, asset_name = row            
            print('adding task: object:', basic_description, 'with attributes:', attributes, 'asset name:', asset_name)
            task_seq.append((basic_description, attributes, asset_name))
    os.makedirs(out_dir, exist_ok=True)
    affected_view_prompts = []
    for basic_description, attributes, asset_name in task_seq:
        print('processing task: object:', basic_description, 'with attributes:', attributes, 'asset name:', asset_name)
        affected_view_prompt = affected_view_prompt_generator(basic_description, attributes)
        print('affected view prompt:', affected_view_prompt)
        affected_view_prompts.append(affected_view_prompt)
    with open(os.path.join(out_dir, "new_val_affected_view_prompts.tsv"), "w", encoding='utf-8') as f:
        for affected_view_prompt in affected_view_prompts:
            f.write(f"{affected_view_prompt}\n")
    print('Done generating affected view prompts for all tasks')

def testing_image_editing_4_3d_editing():
    pipeline = get_image_to_3d_single_attribute_slider_pipeline()
    out_dir = "validation/test_3d_edit"
    views_dir = "validation/multi_views_new_dataset_40/obv_woman_22e5da448fa34ec0a0ea82f8d4659866_views"
    image_pairs_dir = "/home/rwang/TRELLIS/validation/new_val_edited_best_views_combined_40/obv_woman_22e5da448fa34ec0a0ea82f8d4659866_edited_views_combined_0"
    view_images = [Image.open(os.path.join(views_dir, f)) for f in os.listdir(views_dir) if f.endswith(".png")]
    positive_images = [Image.open(os.path.join(image_pairs_dir, f)) for f in os.listdir(image_pairs_dir) if "_pos_" in f]
    negative_images = [Image.open(os.path.join(image_pairs_dir, f)) for f in os.listdir(image_pairs_dir) if "_neg_" in f]
    cond = pipeline.get_reconstruct_edit_cond(view_images, positive_images, negative_images)
    edit_3d_assets_w_images(pipeline, cond, out_dir=os.path.join(out_dir, os.path.basename(image_pairs_dir)), cfg_strength=3)
    print('Done processing all tasks')

if __name__ == "__main__":
    # pass
    # test_select_best_views_and_edit()
    # generating_image_pairs_from_object_and_attribute_new_validation("validation/new_validation.csv", "validation/")
    # test_gemini_api()
    # generating_affected_view_prompts_from_object_and_attribute_new_validation("validation/new_validation.csv", "validation/")
    testing_image_editing_4_3d_editing()

    
    # sdxl_pipeline = get_sdxl_pipeline()
    # positive_prompts = ["a sofa, very industrial style"]
    # negative_prompts = ["a sofa, not industrial style at all"]
    # pos_img = generate_image(sdxl_pipeline, positive_prompts[0])
    # neg_img = generate_image(sdxl_pipeline, negative_prompts[0])
    # # save images to
    # out_dir = "outputs/test_img_slat"
    # os.makedirs(out_dir, exist_ok=True)
    # pos_img.save(os.path.join(out_dir, "pos_img.png"))
    # neg_img.save(os.path.join(out_dir, "neg_img.png"))


    # obj_name_pos_attribute_pairs = [
    #     ("a bed", "very futuristic style"),
    # ]
    # out_dir = "outputs/test_img_slat/img_direction_pairs_v4_empty_neg"
    # sdxl_pipeline = get_sdxl_pipeline()
    # batch_get_image_pairs_from_object_and_pos_attributes(sdxl_pipeline, obj_name_pos_attribute_pairs, 10, out_dir, empty_neg=True)

    # obj_name_pos_attribute_pairs = [
    #     # ("a sofa", "very ancient Japanese style"),
    #     # ("a bed", "very cozy like a cotton candy"),
    #     # ("a chair", "very industrial style, with many metal elements"),
    #     # ("a table", "very modern and futuristic style"),
    #     # ("a man", "with very very big and round eyes"),
    #     # ("a man, full body", "very strong and muscular"),
    #     # ("a steak", "very well-done and charred"),
    #     # ('a salmon bowl', 'very crispy and charred'),
    #     # ('a chicken drumstick', 'very crispy and charred'),
    #     ("a bed", "very futuristic style"),
    # ]
    # out_dir = "outputs/test_img_slat/img_direction_pairs_v5"
    # # sdxl_pipeline = get_sdxl_pipeline()
    # # batch_get_image_pairs_from_object_and_pos_attributes(sdxl_pipeline, obj_name_pos_attribute_pairs, 10, out_dir)
    # # qwen_pipeline = get_qwen_pipeline()
    # qwen_lora_pipeline = get_qwen_lora_pipeline()
    # batch_get_image_pairs_from_object_and_pos_attributes(qwen_lora_pipeline, generate_image_qwen_lora, obj_name_pos_attribute_pairs, 10, out_dir)


    # 1/22/2026 testing image slat editing
    # pipeline = get_image_to_3d_single_attribute_slider_pipeline()
    # # images_dir = "/home/rwang/TRELLIS/outputs/test_seq_edit_slats_uniform_vox_25/temp_processing/renders/Bed_70540c4f-9d4d-44f4-bc4d-fc835d5dca2c_2"
    # images_dir = "/home/rwang/TRELLIS/outputs/test_seq_edit_slats_uniform_vox_25/obv_steak_board"
    # images = [Image.open(os.path.join(images_dir, f)) for f in os.listdir(images_dir) if f.endswith(".png")]
    # direction_images_dir = "/home/rwang/TRELLIS/outputs/test_img_slat/img_direction_pairs_v5/generate_image_qwen_lora_a_steak_very_well-done_and_charred"
    # positive_image_paths = [os.path.join(direction_images_dir, f) for f in os.listdir(direction_images_dir) if f.endswith("_pos.png")]
    # negative_image_paths = [os.path.join(direction_images_dir, f) for f in os.listdir(direction_images_dir) if f.endswith("_neg.png")]
    
    # positive_image_paths.sort(key=lambda x: int(os.path.basename(x).split("_")[0]))
    # negative_image_paths.sort(key=lambda x: int(os.path.basename(x).split("_")[0]))
    # # test single image
    # test_single_image = False
    # if test_single_image:
    #     positive_image_paths = positive_image_paths[:1]
    #     negative_image_paths = negative_image_paths[:1]

    # positive_images = [Image.open(path) for path in positive_image_paths]
    # negative_images = [Image.open(path) for path in negative_image_paths]

    # reverse_images = True
    # if reverse_images:
    #     tmp = positive_images
    #     positive_images = negative_images
    #     negative_images = tmp
    # cond = pipeline.get_reconstruct_edit_cond(images, positive_images, negative_images)
    # edit_3d_assets_w_images(pipeline, cond, out_dir="outputs/test_img_slat/img_slat_editing_cfg_3_v5_steak_board", cfg_strength=3)

    # 1/24/2026 testing text slat editing
    # pipeline = get_text_to_3d_single_attribute_slider_pipeline()  # Uses default GPU (cuda:0)
    # # get conditions 
    # positive_prompts = []
    # negative_prompts = []
    # neutral_prompt = "A wooden cabinet in green color"
    # out_dir = "/home/rwang/TRELLIS/outputs/test_img_slat/img_direction_pairs_v5/A_wooden_cabinet_in_green_color_very_modern"
    # with open(os.path.join(out_dir, "img_gen_prompt_map.tsv"), "r") as f:
    #     for line in f:
    #         file, prompt = line.strip().split("\t")
    #         if '_pos.png' in file:
    #             positive_prompts.append(prompt)
    #         elif '_neg.png' in file:
    #             negative_prompts.append(prompt)
    # cond = pipeline.get_cond_gen_edit(neutral_prompt, positive_prompts, negative_prompts)
    # # edit_3d_assets_w_text(pipeline, cond, out_dir="outputs/test_img_slat/text_slat_editing_cfg_3_modern_v5", cfg_strength=3)
    # edit_3d_assets_w_text(pipeline, cond, out_dir="outputs/test_img_slat/text_slat_editing_cfg_4_modern_v5", cfg_strength=4)
