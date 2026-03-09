import os
import sys
from huggingface_hub import ImageSegmentationInput
import numpy as np
from typing import List, Tuple, Dict
# os.environ['SPCONV_ALGO'] = 'native'
# os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
# os.environ.setdefault('TORCH_HOME', os.path.expanduser('~/.cache/torch'))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import slider_mapping
import csv
import json
from dotenv import load_dotenv
load_dotenv()
import traceback


def gen_new_sliders(
    view_images_dir: str,
    edited_images_dir: str,
    editing_prompt_pair: Tuple[str, str],
    info_path: str,
    out_dir: str,
    reference_view_images_paths: List[str],
    pool: slider_mapping.GPUWorkerPool,
):
    # step 1. find boundary
    # if os.path.exists(os.path.join(out_dir, "boundaries.txt")):
    #     with open(os.path.join(out_dir, "boundaries.txt"), "r") as f:
    #         lines = f.readlines()
    #         pos_boundary = float(lines[0].split(": ")[1])
    #         neg_boundary = float(lines[1].split(": ")[1])
    #     print(f"Found boundaries in {os.path.join(out_dir, 'boundaries.txt')}")
    # else:
    #     # read unaffected_questions and unaffected_dependencies from file
    #     json_info = json.load(open(info_path))
    #     unaffected_questions = json_info["filtered_questions"]
    #     # clean \u2019 from unaffected_questions
    #     unaffected_questions = [q.replace("\u2019", "'") for q in unaffected_questions]
    #     unaffected_dependencies = json_info["dependencies"]
    #     pos_boundary, neg_boundary = slider_mapping.find_boundary_binary_search(
    #         view_images_dir, edited_images_dir, unaffected_questions, unaffected_dependencies,
    #         gpu_ids=[0,1],
    #         explore_growth=1, # uniform explore at the beginning
    #         max_range=15,
    #         threshold=0.5,
    #         out_dir=out_dir,
    #         reference_view_images_paths=reference_view_images_paths,
    #         editing_prompt_pair=editing_prompt_pair,
    #     )
    # print(f"Positive boundary: {pos_boundary}, Negative boundary: {neg_boundary}")

    # step 2. get LPIPS curve
    gpu_ids = pool.gpu_ids
    n_samples = 9
    if os.path.exists(os.path.join(out_dir, f"lpips_curve_gradient_{n_samples}.txt")):
        print(f"LPIPS curve already exists for {out_dir} with {n_samples} samples. Skipping...")
    else:
        pool.set_conds(view_images_dir, edited_images_dir)
        slider_mapping.get_lpips_curve_gradient_based(
            view_images_dir,
            edited_images_dir,
            editing_prompt_pair,
            [5, -5],
            out_dir,
            gpu_ids=gpu_ids,
            n_samples=n_samples,
            pool=pool,
        )

    # step 3. generate new slider values
    pool.set_conds(view_images_dir, edited_images_dir)
    slider_mapping.get_slider_values_from_lpips_percentages(
        lpips_curve_path=os.path.join(out_dir, f"lpips_curve_gradient_{n_samples}.txt"),
        percentages=np.linspace(0, 100, 9),
        view_images_dir=view_images_dir,
        edited_images_dir=edited_images_dir,
        editing_prompt_pair=editing_prompt_pair,
        out_dir=out_dir,
        gpu_ids=gpu_ids,
        pool=pool,
    )

def batch_gen_new_sliders(
    asset_info_path: str,
    prompt_pair_path: str,
    view_base_dir: str,
    edited_base_dir: str,
    info_base_dir: str,
    out_dir: str,
):
    assets_info = []
    with open(asset_info_path, "r") as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            basic_description, attributes, asset_name = row
            assets_info.append((basic_description, attributes, asset_name[:-4]))
    editing_prompt_pairs = []
    with open(prompt_pair_path, "r") as f: # tsv
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            neg_prompt, pos_prompt = row
            editing_prompt_pairs.append((neg_prompt.strip(), pos_prompt.strip()))
    gpu_ids = [1, 2, 3]
    with slider_mapping.GPUWorkerPool(gpu_ids, workers_per_gpu=3) as pool:
        for i, (basic_description, attributes, asset_name) in enumerate(assets_info):
            print(f"Generating new slider values for {asset_name}...")
            view_images_dir = os.path.join(view_base_dir, asset_name + "_views")
            edited_images_dir = os.path.join(edited_base_dir, asset_name + f"_edited_views_{i}")
            info_dir = os.path.join(info_base_dir, asset_name)
            info_file_path = os.path.join(info_dir, f"info_{i}.json")
            out_dir_asset = os.path.join(out_dir, asset_name + f"_{i}")
            reference_view_images_dir = os.path.join(info_dir, 'views')
            reference_view_images_paths = [os.path.join(reference_view_images_dir, f) for f in sorted(os.listdir(reference_view_images_dir)) if f.endswith(".png")]
            try:
                gen_new_sliders(view_images_dir, edited_images_dir, editing_prompt_pairs[i], info_file_path, out_dir_asset, reference_view_images_paths, pool)
            except Exception as e:
                print(f"Error generating new slider values for {asset_name}: {e}")
                with open(os.path.join(out_dir_asset, "error.txt"), "a") as f:
                    f.write(f"{asset_name}: {e}\n{traceback.format_exc()}\n\n")
                continue
    print(f"Finished generating new slider values for {len(assets_info)} assets")


def make_figures(image_seqs: List[List[str]], values):
    """
    Concatenate image sequences into a single figure.
    Each sub-list becomes a vertical column; columns are stacked horizontally.
    The value is drawn on the top of every image in that column.
    """
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    columns = []
    for imgs, val in zip(image_seqs, values):
        text = f"{float(val):.2f}"
        pil_imgs = []
        for p in imgs:
            im = Image.open(p).copy()
            draw = ImageDraw.Draw(im)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = (im.width - tw) // 2
            draw.rectangle((tx - 4, 2, tx + tw + 4, th + 6), fill=(0, 0, 0, 180))
            draw.text((tx, 2), text, fill=(255, 255, 255), font=font)
            pil_imgs.append(im)
        w = max(im.width for im in pil_imgs)
        h = sum(im.height for im in pil_imgs)
        col = Image.new('RGB', (w, h))
        y = 0
        for im in pil_imgs:
            col.paste(im, (0, y))
            y += im.height
        columns.append(col)

    max_h = max(c.height for c in columns)
    total_w = sum(c.width for c in columns)
    result = Image.new('RGB', (total_w, max_h), (255, 255, 255))

    x = 0
    for col in columns:
        result.paste(col, (x, 0))
        x += col.width

    return result

def make_figures_horizontal(image_paths: List[str], values):
    """
    Draw images horizontally in a single row, with value labels on top of each.
    """
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    pil_imgs = []
    for p, val in zip(image_paths, values):
        im = Image.open(p).copy()
        draw = ImageDraw.Draw(im)
        text = f"{float(val):.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (im.width - tw) // 2
        draw.rectangle((tx - 4, 2, tx + tw + 4, th + 6), fill=(0, 0, 0, 180))
        draw.text((tx, 2), text, fill=(255, 255, 255), font=font)
        pil_imgs.append(im)

    max_h = max(im.height for im in pil_imgs)
    total_w = sum(im.width for im in pil_imgs)
    result = Image.new('RGB', (total_w, max_h), (255, 255, 255))

    x = 0
    for im in pil_imgs:
        result.paste(im, (x, 0))
        x += im.width

    return result


def get_slider_values_from_lpips_curve_gradient(lpips_curve_gradient_path: str, n_samples: List[float]) -> List[float]:
    candidates = []
    with open(lpips_curve_gradient_path, "r") as f:
        # segment the whole file into two parts using '\n\n'
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            if line == "":
                break # here is the boundary of the two parts
            x, y, validity = line.split(",") 
            x = float(x)
            y = float(y)
            validity = int(validity) # 1 if valid, 0 if invalid
            candidates.append((x, y, validity))
    if not candidates:
        return []

    y_values = [c[1] for c in candidates]
    y_min, y_max = min(y_values), max(y_values)
    bucket_size = (y_max - y_min) / n_samples
    target_ys = np.linspace(y_min, y_max, n_samples)

    xs = np.array([c[0] for c in candidates])
    ys = np.array([c[1] for c in candidates])
    vs = np.array([c[2] for c in candidates])

    valid_mask = vs == 1
    invalid_mask = vs == 0

    used = np.zeros(len(candidates), dtype=bool)

    selected_xs = []
    for ty in target_ys:
        dists = np.abs(ys - ty)
        dists_available = np.where(~used, dists, np.inf)

        avail_valid = valid_mask & ~used
        avail_invalid = invalid_mask & ~used

        best_idx = None

        if avail_valid.any():
            valid_dists = np.where(avail_valid, dists, np.inf)
            best_valid_idx = int(np.argmin(valid_dists))
            if valid_dists[best_valid_idx] <= bucket_size:
                best_idx = best_valid_idx

        if best_idx is None and avail_invalid.any():
            invalid_dists = np.where(avail_invalid, dists, np.inf)
            best_invalid_idx = int(np.argmin(invalid_dists))
            if avail_valid.any():
                best_idx = best_invalid_idx if invalid_dists[best_invalid_idx] < valid_dists[best_valid_idx] else best_valid_idx
            else:
                best_idx = best_invalid_idx

        if best_idx is None:
            best_idx = int(np.argmin(dists_available))

        used[best_idx] = True
        selected_xs.append(xs[best_idx])

    return selected_xs


def gen_big_figures(
    asset_info_path: str,
    out_base_dir: str,
):
    assets_info = []
    with open(asset_info_path, "r") as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            basic_description, attributes, asset_name = row
            assets_info.append((basic_description, attributes, asset_name[:-4]))
    
    for i, (basic_description, attributes, asset_name) in enumerate(assets_info):
        out_dir_asset = os.path.join(out_base_dir, asset_name + f"_{i}")
        new_slider_values_path = os.path.join(out_dir_asset, "lpips_curve_gradient_9.txt")
        if not os.path.exists(new_slider_values_path):
            print(f"New slider values not found for {asset_name}")
            continue
        new_slider_values = get_slider_values_from_lpips_curve_gradient(new_slider_values_path, 9)
        # don't sort svs!!!
        new_slider_images = []
        for new_slider_value in new_slider_values:
            # get images from the folder corresponding to the new slider value
            if new_slider_value == 0:
                new_slider_value_dir = os.path.join(out_dir_asset, "search_at_0.00000")
            else:
                new_slider_value_dir = os.path.join(out_dir_asset, f"search_at_{new_slider_value:.5f}", "adjusted_views")
            if not os.path.exists(new_slider_value_dir):
                print(f"!!! New slider value directory not found for {asset_name} at {new_slider_value:.5f}")
                continue
            images = [f for f in os.listdir(new_slider_value_dir) if f.endswith(".png") and f.startswith("key_frame_")]
            images = sorted(images)
            images = [os.path.join(new_slider_value_dir, image) for image in images]
            new_slider_images.append(images) # [n_slider_values, n_views]
        old_slider_values = np.linspace(-5, 5, 9)
        old_slider_images = []
        for old_slider_value in old_slider_values:
            if old_slider_value == 0:
                old_slider_value_dir = os.path.join(out_dir_asset, "search_at_0.00000")
            else:
                old_slider_value_dir = os.path.join(out_dir_asset, f"search_at_{old_slider_value:.5f}", "adjusted_views")
            if not os.path.exists(old_slider_value_dir):
                print(f"!!! Old slider value directory not found for {asset_name} at {old_slider_value:.5f}")
                continue
            images = [f for f in os.listdir(old_slider_value_dir) if f.endswith(".png") and f.startswith("key_frame_")]
            images = sorted(images)
            images = [os.path.join(old_slider_value_dir, image) for image in images]
            old_slider_images.append(images) # [n_slider_values, n_views]
        # make figures
        views = [0, 1, 2, 3]
        for view in views:
            new_slider_images_this_view = [images[view] for images in new_slider_images]
            old_slider_images_this_view = [images[view] for images in old_slider_images]
            print(f"Making figures for {asset_name} view {view}...")
            new_figure = make_figures_horizontal(new_slider_images_this_view, new_slider_values)
            old_figure = make_figures_horizontal(old_slider_images_this_view, old_slider_values)
            print(f"Saving figures for {asset_name} view {view}...")
            new_figure.save(os.path.join(out_dir_asset, f"figure_view_{view}_new_sliders.png"))
            old_figure.save(os.path.join(out_dir_asset, f"figure_view_{view}_old_sliders.png"))
        
    print(f"Finished making figures for {len(assets_info)} assets")

if __name__ == "__main__":
    asset_info_path = "/home/rwang/TRELLIS/validation/new_validation.csv"
    view_base_dir = "/home/rwang/TRELLIS/validation/multi_views_new_dataset_40"
    edited_base_dir = "/home/rwang/TRELLIS/validation/new_val_edited_views_RM_edppv2_mv40_verify_gpt52"
    prompt_pair_path = "/home/rwang/TRELLIS/validation/new_val_prompt_pairs_v2.tsv"
    info_base_dir = "/data/ru_data/results/trellis_output/validation/test_slider_preparation/"
    out_dir = "/data/ru_data/results/trellis_output/validation/test_lpips_curve/"
    batch_gen_new_sliders(asset_info_path, prompt_pair_path, view_base_dir, edited_base_dir, info_base_dir, out_dir)
    # gen_big_figures(asset_info_path, out_dir)