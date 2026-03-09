import os
import torch
import shutil
from typing import *
from PIL import Image

import time
import imageio
import numpy as np
import open3d as o3d
from torchvision import transforms
import torch.nn.functional as F
import utils3d
import json
from subprocess import DEVNULL, call
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel
import lpips
import matplotlib.pyplot as plt
import pandas as pd


def get_multi_view_from_3d_assets_for_evaluation(asset_path: str, out_dir: str, verbose: bool = False):
    """
    Gets multi-view images from a 3D asset.
    according to Trellis: We render 8 images per asset with yaw angles at every 45 deg, a pitch angle of 30 deg, and a radius of 2.
    
    Returns:
        Tuple[bool, str]: (success, error_message)
    """ 
    import subprocess
    
    BLENDER_PATH = 'blender' # Assumes blender is in PATH
    RENDER_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset_toolkits', 'blender_script', 'render.py')

    # Check if input file exists
    if not os.path.exists(asset_path):
        return False, f"Input file not found: {asset_path}"
    
    # Check if render script exists
    if not os.path.exists(RENDER_SCRIPT):
        return False, f"Render script not found: {RENDER_SCRIPT}"

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Generate camera poses
    yaws = np.arange(0, 360, 45)
    pitchs = [30] * len(yaws)
    radius = 2.0
    # deg to rad
    yaws = np.deg2rad(yaws)
    pitchs = np.deg2rad(pitchs)
    views = [{'yaw': float(y), 'pitch': float(p), 'radius': radius, 'fov': 40 / 180 * np.pi} 
                for y, p in zip(yaws, pitchs)]

    # Use CYCLES with GPU for headless rendering
    cmd = [
        BLENDER_PATH, '-b', '-P', RENDER_SCRIPT,
        '--',
        '--views', json.dumps(views),
        '--object', os.path.abspath(asset_path),
        '--resolution', str(512),
        '--output_folder', os.path.abspath(out_dir),
        '--engine', 'CYCLES',
        '--samples', '16',  # Low samples for speed (default is 128)
    ]
    
    try:
        if verbose:
            # Show output for debugging
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        else:
            # Capture but don't show
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if result.returncode != 0:
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            return False, f"Blender returned code {result.returncode}: {error_msg}"
        
        # VERIFY that images were actually created
        png_files = [f for f in os.listdir(out_dir) if f.endswith('.png')]
        if len(png_files) < 8:
            error_msg = f"Only {len(png_files)} images created (expected 8)"
            if verbose and result.stderr:
                error_msg += f"\nStderr: {result.stderr[:500]}"
            return False, error_msg
        
        return True, f"Successfully rendered {len(png_files)} views"
        
    except subprocess.TimeoutExpired:
        return False, f"Timeout after 120s"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def count_rendered_images(out_dir: str) -> int:
    """Count PNG files in output directory."""
    if not os.path.exists(out_dir):
        return 0
    return len([f for f in os.listdir(out_dir) if f.endswith('.png')])


def render_multi_views(val_obj_dir: str, num_workers: int = 4, verbose: bool = False):
    """
    Render multi-view images for all GLB files in the validation directory.
    
    Args:
        val_obj_dir: Root directory containing slider/object subdirectories
        num_workers: Number of parallel Blender processes (default 8)
        verbose: Show detailed error messages
    """
    # Collect all GLB files to render
    render_jobs = []
    already_done = 0
    
    slider_dirs = [os.path.join(val_obj_dir, d) for d in os.listdir(val_obj_dir) if os.path.isdir(os.path.join(val_obj_dir, d))]
    slider_dirs.sort()
    
    for slider_dir in slider_dirs:
        basic_obj_dirs = [os.path.join(slider_dir, d) for d in os.listdir(slider_dir) if os.path.isdir(os.path.join(slider_dir, d))]
        basic_obj_dirs.sort()
        for basic_obj_dir in basic_obj_dirs:
            glb_files = [os.path.join(basic_obj_dir, d) for d in os.listdir(basic_obj_dir) if d.endswith('.glb')]
            glb_files.sort()
            for glb_file in glb_files:
                old_out_dir = glb_file[:-4].replace('.', '*') + '_views'
                if os.path.exists(old_out_dir):
                    # remove it if it exists
                    shutil.rmtree(old_out_dir)
                out_dir = glb_file[:-4].replace('.', 'p') + '_views'
                # Skip if already rendered with 8 PNG files
                num_images = count_rendered_images(out_dir)
                if num_images >= 8:
                    already_done += 1
                    continue
                # If partially rendered, clean up and re-render
                if os.path.exists(out_dir) and num_images > 0:
                    shutil.rmtree(out_dir)
                render_jobs.append((glb_file, out_dir))
    
    print(f"Already rendered: {already_done}")
    print(f"To render: {len(render_jobs)} GLB files with {num_workers} workers...")
    
    if len(render_jobs) == 0:
        print("Nothing to render!")
        return
    
    # First, test with a single file to make sure Blender works
    print("Testing Blender with first file...")
    test_glb, test_out = render_jobs[0]
    success, msg = get_multi_view_from_3d_assets_for_evaluation(test_glb, test_out, verbose=True)
    if not success:
        print(f"ERROR: Test render failed: {msg}")
        print("Please fix Blender setup before running batch render.")
        return
    print(f"Test render OK: {msg}")
    render_jobs = render_jobs[1:]  # Remove the test file from queue
    
    if len(render_jobs) == 0:
        print("Only had 1 file and it's done!")
        return
    
    # Parallel rendering
    def render_single(args):
        glb_file, out_dir = args
        success, msg = get_multi_view_from_3d_assets_for_evaluation(glb_file, out_dir, verbose=verbose)
        return success, msg, glb_file
    
    success_count = 1  # Already did test file
    fail_count = 0
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(render_single, job): job for job in render_jobs}
        for future in tqdm(as_completed(futures), total=len(render_jobs), desc="Rendering"):
            success, msg, glb_file = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_files.append((glb_file, msg))
                if verbose:
                    tqdm.write(f"FAILED: {glb_file}: {msg}")
    
    print(f"\nRendering complete: {success_count} succeeded, {fail_count} failed")
    
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for glb_file, msg in failed_files[:10]:  # Show first 10
            print(f"  {glb_file}: {msg}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

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

def calc_LPIPS_similarity(loss_fn, images_1: List[Image.Image], images_2: List[Image.Image], device="cuda"):
    """
    Calculate mean LPIPS distance between two sets of corresponding images.
    Lower LPIPS = more similar.
    """
    assert len(images_1) == len(images_2), "Image lists must have same length"
    
    batch_1 = preprocess_images_for_lpips(images_1, device=device)  # [N, C, H, W]
    batch_2 = preprocess_images_for_lpips(images_2, device=device)  # [N, C, H, W]
    
    with torch.no_grad():
        # LPIPS returns [N, 1, 1, 1] distances for each pair
        distances = loss_fn(batch_1, batch_2)
    
    return distances.mean().item()

def calc_CLIP_similarity_slider_style(slider_parent_dir: str, neutral_prompt_file: str, output_file: str):
    import csv
    
    model, processor = get_CLIP_model_and_processor()
    slider_dirs = [os.path.join(slider_parent_dir, d) for d in os.listdir(slider_parent_dir) if os.path.isdir(os.path.join(slider_parent_dir, d))]
    slider_dirs.sort()

    # Open CSV writer with proper quoting to handle special characters in prompts
    csvfile = open(output_file, 'w', newline='')
    writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['slider_id', 'obj_id', 'slider_scale', 'neutral_prompt', 'pos_prompt', 'neg_prompt', 
                     'neutral_clip_score', 'pos_clip_score', 'neg_clip_score'])
    
    # read neutral prompts from file
    neutral_prompts = {}
    with open(neutral_prompt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                obj_id, prompt = parts[0], parts[1]
                # Sanitize prompt: remove newlines and extra whitespace
                prompt = ' '.join(prompt.split())
                neutral_prompts[int(obj_id)] = prompt

    for slider_dir in slider_dirs:
        print(f"Processing {slider_dir}...")
        obj_dirs = [os.path.join(slider_dir, d) for d in os.listdir(slider_dir) if os.path.isdir(os.path.join(slider_dir, d))]
        obj_dirs.sort()
        # get direction slider from the slider_dir name:
        direction_name = os.path.basename(slider_dir)
        pos_direction_name, neg_direction_name = direction_name.split('-')
        pos_direction = pos_direction_name.replace('_', ' ').replace('\n', '')
        neg_direction = neg_direction_name.replace('_', ' ').replace('\n', '')

        print("positive direction: ", pos_direction)
        print("negative direction: ", neg_direction)

        for obj_dir in tqdm(obj_dirs, desc="Processing objects"):
            view_folders = [os.path.join(obj_dir, d) for d in os.listdir(obj_dir) if os.path.isdir(os.path.join(obj_dir, d))]
            view_folders.sort()
            # get obj_id from the obj_dir name:
            obj_id = int(os.path.basename(obj_dir))
            neutral_prompt = neutral_prompts[obj_id]
            pos_prompt = f"{neutral_prompt}, {pos_direction}"
            neg_prompt = f"{neutral_prompt}" + (f", {neg_direction}" if neg_direction != "" else "")
            for view_folder in view_folders:
                slider_id, obj_id, slider_scale, cfg, _ = os.path.basename(view_folder).split('_')
                slider_scale = float(slider_scale[2:].replace('p', '.'))
                img_files = [os.path.join(view_folder, d) for d in os.listdir(view_folder) if d.endswith('.png')]
                img_files.sort()
                images = [Image.open(img_file) for img_file in img_files]
                neutral_clip_score = calc_CLIP_similarity_text_vs_images(model, processor, neutral_prompt, images)
                pos_clip_score = calc_CLIP_similarity_text_vs_images(model, processor, pos_prompt, images)
                neg_clip_score = calc_CLIP_similarity_text_vs_images(model, processor, neg_prompt, images)
                writer.writerow([slider_id, obj_id, slider_scale, neutral_prompt, pos_prompt, neg_prompt,
                                 neutral_clip_score, pos_clip_score, neg_clip_score])
    
    csvfile.close()


def calc_LPIPS_similarity_slider_style(slider_parent_dir: str, output_file: str):
    import csv
    
    loss_fn = get_lpips_model()
    slider_dirs = [os.path.join(slider_parent_dir, d) for d in os.listdir(slider_parent_dir) if os.path.isdir(os.path.join(slider_parent_dir, d))]
    slider_dirs.sort()

    # Open CSV writer with proper quoting to handle special characters in prompts
    csvfile = open(output_file, 'w', newline='')
    writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['slider_id', 'obj_id', 'slider_scale', 'lpips_score'])

    for slider_dir in slider_dirs:
        print(f"Processing {slider_dir}...")
        obj_dirs = [os.path.join(slider_dir, d) for d in os.listdir(slider_dir) if os.path.isdir(os.path.join(slider_dir, d))]
        obj_dirs.sort()

        for obj_dir in tqdm(obj_dirs, desc="Processing objects"):
            view_folders = [os.path.join(obj_dir, d) for d in os.listdir(obj_dir) if os.path.isdir(os.path.join(obj_dir, d))]
            view_folders.sort()

            for view_folder in view_folders:
                slider_id, obj_id, slider_scale, cfg, _ = os.path.basename(view_folder).split('_')
                slider_scale = float(slider_scale[2:].replace('p', '.'))
                if slider_scale == 0.00:
                    print('Found base view folder: ', view_folder)
                    base_view_folder = view_folder
                    break
            base_image_files = [os.path.join(base_view_folder, d) for d in os.listdir(base_view_folder) if d.endswith('.png')]
            base_image_files.sort()
            base_images = [Image.open(img_file) for img_file in base_image_files]
            # get teh base image where the slider scale is 0.00
            for view_folder in view_folders:
                if view_folder == base_view_folder:
                    writer.writerow([slider_id, obj_id, 0.00, 0.00])
                    continue
                slider_id, obj_id, slider_scale, cfg, _ = os.path.basename(view_folder).split('_')
                slider_scale = float(slider_scale[2:].replace('p', '.'))
                img_files = [os.path.join(view_folder, d) for d in os.listdir(view_folder) if d.endswith('.png')]
                img_files.sort()
                images = [Image.open(img_file) for img_file in img_files]
                lpips_score = calc_LPIPS_similarity(loss_fn, base_images, images)
                writer.writerow([slider_id, obj_id, slider_scale, lpips_score])
    
    csvfile.close()
        
def plot_CLIP_similarity_slider_style(input_file: str, output_dir: str, group_by: str = "all", obj_file_path: str = None):
    """
    Plot CLIP scores vs slider_scale.
    
    Args:
        input_file: TSV file with columns: slider_id, obj_id, slider_scale, neutral_clip_score, pos_clip_score, neg_clip_score
        output_file: Output image file path (e.g., .png, .pdf)
        group_by: "all" (average everything), "slider_id" (one subplot per slider), or "obj_id" (one subplot per object)
    """
    # Use proper CSV reader to handle quoted fields
    df = pd.read_csv(input_file, sep='\t', quoting=1)  # quoting=1 is QUOTE_ALL compatible
    score_cols = ['neutral_clip_score', 'pos_clip_score', 'neg_clip_score']
    labels = ['neutral', 'pos', 'neg']
    colors = ['gray', 'green', 'red']

    slider_map = {
        0: "very modern-very traditional",
        1: "very Japanese-not at all Japanese",
        2: "very Japanese-",
        3: "very minimalist-very ornate",
        4: "very industrial-not industrial at all",
        5: "very industrial-",
        6: "very American country-not at all American country",
        7: "very American country-",
    }

    obj_prompt = {}
    with open(obj_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                obj_id, prompt = parts[0], parts[1]
                obj_prompt[int(obj_id)] = prompt
    
    os.makedirs(output_dir, exist_ok=True)
    
    if group_by == "all":
        # Average across all slider_ids and obj_ids
        grouped = df.groupby('slider_scale')[score_cols].mean().reset_index()
        grouped = grouped.sort_values('slider_scale')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for col, label, color in zip(score_cols, labels, colors):
            ax.plot(grouped['slider_scale'], grouped[col], label=label, color=color, marker='o')
        ax.set_xlabel('Slider Scale')
        ax.set_ylabel('CLIP Score')
        ax.set_title('CLIP Similarity vs Slider Scale (Averaged by all sliders and objs)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'clip_similarity_slider_style_all.png'), dpi=150, bbox_inches='tight')
        plt.close()

    elif group_by == "slider_id":
        # One subplot per slider_id
        slider_ids = df['slider_id'].unique()
        n_sliders = len(slider_ids)
        fig, axes = plt.subplots(1, n_sliders, figsize=(6 * n_sliders, 5), squeeze=False)
        
        for i, slider_id in enumerate(sorted(slider_ids)):
            ax = axes[0, i]
            subset = df[df['slider_id'] == slider_id]
            grouped = subset.groupby('slider_scale')[score_cols].mean().reset_index()
            grouped = grouped.sort_values('slider_scale')
            
            for col, label, color in zip(score_cols, labels, colors):
                ax.plot(grouped['slider_scale'], grouped[col], label=label, color=color, marker='o')
            ax.set_xlabel('Slider Scale')
            ax.set_ylabel('CLIP Score')
            ax.set_title(f'Slider: {slider_map[slider_id]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'clip_similarity_slider_style_slider_id.png'), dpi=150, bbox_inches='tight')
        plt.close()

    elif group_by == "obj_id":
        # One subplot per obj_id
        # generate plots grouped by slider_id for each obj_id
        obj_ids = df['obj_id'].unique()
        for obj_id in tqdm(obj_ids, desc="Processing objects"):
            subset = df[df['obj_id'] == obj_id]
            slider_ids = subset['slider_id'].unique()
            for slider_id in slider_ids:
                subset_slider = subset[subset['slider_id'] == slider_id]
                grouped = subset_slider.groupby('slider_scale')[score_cols].mean().reset_index()
                grouped = grouped.sort_values('slider_scale')
                fig, ax = plt.subplots(figsize=(8, 6))
                for col, label, color in zip(score_cols, labels, colors):
                    ax.plot(grouped['slider_scale'], grouped[col], label=label, color=color, marker='o')
                ax.set_xlabel('Slider Scale')
                ax.set_ylabel('CLIP Score')
                ax.set_title(f'Object: {obj_prompt[obj_id]} - Slider: {slider_map[slider_id]}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, f'clip_similarity_slider_style_obj_id_{obj_id}_slider_id_{slider_id}.png'), dpi=150, bbox_inches='tight')
                plt.close()
    
    else:
        raise ValueError(f"group_by must be 'all', 'slider_id', or 'obj_id', got '{group_by}'")


def plot_LPIPS_similarity_slider_style(input_file: str, output_dir: str, obj_file_path: str, group_by: str = "all", ):
    """
    Plot LPIPS scores vs slider_scale.
    
    Args:
        input_file: TSV file with columns: slider_id, obj_id, slider_scale, lpips_score
        output_file: Output image file path (e.g., .png, .pdf)
        group_by: "all" (average everything), "slider_id" (one subplot per slider), or "obj_id" (one subplot per object)
    """
    # Use proper CSV reader to handle quoted fields
    df = pd.read_csv(input_file, sep='\t', quoting=1)  # quoting=1 is QUOTE_ALL compatible
    score_cols = ['lpips_score']
    labels = ['lpips']
    colors = ['blue']

    slider_map = {
        0: "very modern-very traditional",
        1: "very Japanese-not at all Japanese",
        2: "very Japanese-",
        3: "very minimalist-very ornate",
        4: "very industrial-not industrial at all",
        5: "very industrial-",
        6: "very American country-not at all American country",
        7: "very American country-",
    }
    obj_prompt = {}
    with open(obj_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                obj_id, prompt = parts[0], parts[1]
                obj_prompt[int(obj_id)] = prompt
    
    os.makedirs(output_dir, exist_ok=True)
    
    if group_by == "all":
        # Average across all slider_ids and obj_ids
        grouped = df.groupby('slider_scale')[score_cols].mean().reset_index()
        grouped = grouped.sort_values('slider_scale')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for col, label, color in zip(score_cols, labels, colors):
            ax.plot(grouped['slider_scale'], grouped[col], label=label, color=color, marker='o')
        ax.set_xlabel('Slider Scale')
        ax.set_ylabel('LPIPS Score')
        ax.set_title('LPIPS Similarity vs Slider Scale (Averaged by all sliders and objs)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'lpips_similarity_slider_style_all.png'), dpi=150, bbox_inches='tight')
        plt.close()

    elif group_by == "slider_id":
        # One subplot per slider_id
        slider_ids = df['slider_id'].unique()
        n_sliders = len(slider_ids)
        fig, axes = plt.subplots(1, n_sliders, figsize=(6 * n_sliders, 5), squeeze=False)
        
        for i, slider_id in enumerate(sorted(slider_ids)):
            ax = axes[0, i]
            subset = df[df['slider_id'] == slider_id]
            grouped = subset.groupby('slider_scale')[score_cols].mean().reset_index()
            grouped = grouped.sort_values('slider_scale')
            
            for col, label, color in zip(score_cols, labels, colors):
                ax.plot(grouped['slider_scale'], grouped[col], label=label, color=color, marker='o')
            ax.set_xlabel('Slider Scale')
            ax.set_ylabel('LPIPS Score')
            ax.set_title(f'Slider: {slider_map[slider_id]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'lpips_similarity_slider_style_by_slider_id.png'), dpi=150, bbox_inches='tight')
        plt.close()

    elif group_by == "obj_id":
        # One subplot per obj_id
        # generate plots grouped by slider_id for each obj_id
        obj_ids = df['obj_id'].unique()
        for obj_id in tqdm(obj_ids, desc="Processing objects"):
            subset = df[df['obj_id'] == obj_id]
            slider_ids = subset['slider_id'].unique()
            for slider_id in slider_ids:
                subset_slider = subset[subset['slider_id'] == slider_id]
                grouped = subset_slider.groupby('slider_scale')[score_cols].mean().reset_index()
                grouped = grouped.sort_values('slider_scale')
                fig, ax = plt.subplots(figsize=(8, 6))
                for col, label, color in zip(score_cols, labels, colors):
                    ax.plot(grouped['slider_scale'], grouped[col], label=label, color=color, marker='o')
                ax.set_xlabel('Slider Scale')
                ax.set_ylabel('LPIPS Score')
                ax.set_title(f'Object: {obj_prompt[obj_id]} - Slider: {slider_map[slider_id]}')
                ax.legend() 
                ax.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, f'lpips_similarity_slider_style_obj_id_{obj_id}_slider_id_{slider_id}.png'), dpi=150, bbox_inches='tight')
                plt.close()
    
    else:
        raise ValueError(f"group_by must be 'all', 'slider_id', or 'obj_id', got '{group_by}'")

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--val_dir', type=str, default="/data/ru_data/results/trellis_output/validation/style_sliders_basic")
    # parser.add_argument('--num_workers', type=int, default=4, help="Number of parallel Blender processes (keep low for GPU rendering)")
    # parser.add_argument('--verbose', action='store_true', help="Show detailed error messages")
    # args = parser.parse_args()
    
    # render_multi_views(args.val_dir, num_workers=args.num_workers, verbose=args.verbose)


    # calc_CLIP_similarity_slider_style(
    #     '/data/ru_data/results/trellis_output/validation/style_sliders_basic',
    #     "/data/ru_data/results/trellis_output/validation/style_sliders_basic/obj_idx_to_neutral_prompt.tsv",
    #     "/data/ru_data/results/trellis_output/validation/style_sliders_basic/clip_similarity_slider_style.tsv"
    # )

    # calc_LPIPS_similarity_slider_style(
    #     '/data/ru_data/results/trellis_output/validation/style_sliders_basic',
    #     "/data/ru_data/results/trellis_output/validation/style_sliders_basic/lpips_similarity_slider_style.tsv"
    # )


    # plot_CLIP_similarity_slider_style(
    #     "/data/ru_data/results/trellis_output/validation/style_sliders_basic/clip_similarity_slider_style.tsv",
    #     "/data/ru_data/results/trellis_output/validation/style_sliders_basic/clip_similarity_slider_style_plots",
    #     group_by="slider_id",
    #     obj_file_path="/data/ru_data/results/trellis_output/validation/style_sliders_basic/obj_idx_to_neutral_prompt.tsv"
    # )

    plot_LPIPS_similarity_slider_style(
        "/data/ru_data/results/trellis_output/validation/style_sliders_basic/lpips_similarity_slider_style.tsv",
        "/data/ru_data/results/trellis_output/validation/style_sliders_basic/lpips_similarity_slider_style_plots",
        "/data/ru_data/results/trellis_output/validation/style_sliders_basic/obj_idx_to_neutral_prompt.tsv",
        group_by="obj_id"
    )