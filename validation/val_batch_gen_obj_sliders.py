import os
import numpy as np
import imageio
import sys  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trellis.pipelines import TrellisImageTo3DAttributeSliderPipeline
from trellis.utils import render_utils
import csv
from PIL import Image
import torch
import torch.multiprocessing as mp
from PIL import ImageDraw
os.environ['SPCONV_ALGO'] = 'native' 

def merge_videos_and_save(videos, out_path):
    slider_values = videos.keys()
    # save videos for negatives and positives separately
    neg_values = [slider_value for slider_value in slider_values if slider_value <= 0]
    neg_values.sort(reverse=True) # 0, -1, -2, -3, -4, -5
    pos_values = [slider_value for slider_value in slider_values if slider_value >= 0]
    pos_values.sort() # 0, 1, 2, 3, 4, 5
    # merge videos for negatives and positives
    videos_row_one = [videos[slider_value] for slider_value in neg_values]
    videos_row_two = [videos[slider_value] for slider_value in pos_values]
    if len(videos_row_one) == len(videos_row_two):
        merged_row_one = np.concatenate(videos_row_one, axis=2)
        merged_row_two = np.concatenate(videos_row_two, axis=2)
        merged_videos = np.concatenate([merged_row_one, merged_row_two], axis=1)
    else: # if the number of videos for negatives and positives are not the same, use the videos for positives
        merged_videos = np.concatenate(videos_row_two, axis=2)
    imageio.mimsave(out_path, merged_videos, fps=25)

def get_image_to_3d_single_attribute_slider_pipeline(device="cuda"):
    pipeline = TrellisImageTo3DAttributeSliderPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.to(device)
    return pipeline

def edit_3d_assets_w_images(
        pipeline: TrellisImageTo3DAttributeSliderPipeline, 
        cond: dict,
        seed: int = 42,
        out_dir: str = None,
        cfg_strength: float = 3,
        tag = None,
    ):
    os.makedirs(out_dir, exist_ok=True)
    # slider_values = np.linspace(-10, 10, 11)
    # extreme = 5
    # slider_values = np.linspace(-extreme, extreme, extreme * 2 + 1)
    slider_values = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
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
        print('video_gs shape:', len(video_gs))
        # write slider value on top center of each frame
        for i, frame in enumerate(video_gs):
            frame = Image.fromarray(frame)
            # print('frame shape:', frame.size)
            draw = ImageDraw.Draw(frame)
            draw.text((frame.width // 2, 10), f"{slider_value}", fill=(0, 0, 0))
            frame = np.array(frame)
            # print('frame shape after drawing:', frame.shape)
            video_gs[i] = frame
        videos[slider_value] = video_gs
        # video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        # video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
        # imageio.mimsave(os.path.join(out_dir, f"img_slat_editing_video_{slider_value}.mp4"), video, fps=25)
    print('Merging videos and saving...')
    merge_videos_and_save(videos, os.path.join(out_dir, f"merged_{tag}.mp4"))


def _single_task_worker(gpu_id: int, task: tuple, views_dir: str, image_pairs_dir: str, out_dir: str):
    """
    Run a single task in its own process on a specific GPU.
    If spconv crashes (SIGFPE / illegal memory), only this one task is lost.
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    i, basic_description, attributes, asset_name = task

    print(f"[GPU {gpu_id}] Loading pipeline for task {i}...")
    pipeline = get_image_to_3d_single_attribute_slider_pipeline(device=device)

    print(f"[GPU {gpu_id}] Processing task {i}: {basic_description} | {attributes} | {asset_name}")
    asset_views_dir = os.path.join(views_dir, asset_name[:-4] + "_views")
    edited_views_dir = os.path.join(image_pairs_dir, asset_name[:-4] + f"_edited_views_{i}")
    view_images = [Image.open(os.path.join(asset_views_dir, f)) for f in sorted(os.listdir(asset_views_dir)) if f.endswith(".png")]

    positive_image_paths = [os.path.join(edited_views_dir, f) for f in os.listdir(edited_views_dir) if f.endswith("_pos.png")]
    negative_image_paths = [os.path.join(edited_views_dir, f) for f in os.listdir(edited_views_dir) if f.endswith("_neg.png")]

    positive_image_paths.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))
    negative_image_paths.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))
    positive_images = [Image.open(path) for path in positive_image_paths]
    negative_images = [Image.open(path) for path in negative_image_paths]

    print(f"[GPU {gpu_id}] Editing 3D assets for: {basic_description} | {attributes} | {asset_name}")
    cond = pipeline.get_reconstruct_edit_cond(view_images, positive_images, negative_images)
    edit_3d_assets_w_images(pipeline, cond, out_dir=out_dir, cfg_strength=3, tag=asset_name[:-4] + f"_edited_{i}")
    print(f"[GPU {gpu_id}] Task {i} done.")

def batch_gen_edit_new_validation_dataset(
    asset_info_file: str,
    views_dir: str,
    image_pairs_dir: str,
    out_dir: str,
    gpu_ids: list[int] = None,
    num_workers: int = None,
):
    """
    Args:
        gpu_ids:     Which physical GPUs to use (default: all).
        num_workers: Total number of concurrent processes across all GPUs.
                     Workers are assigned to GPUs round-robin so the load is
                     balanced automatically (e.g. 8 workers on 4 GPUs = 2 per GPU).
                     Default: one worker per GPU.
    """
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))
    if num_workers is None:
        num_workers = len(gpu_ids)
    num_workers = max(1, num_workers)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Using {len(gpu_ids)} GPU(s) with {num_workers} concurrent worker(s)")

    # Read all tasks
    task_seq = []

    # just for testing:
    filtered_asset_names = []
    with open("new_validation_fine.csv", "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)  # skip the header
        for i, row in enumerate(reader):
            basic_description, attributes, asset_name = row
            filtered_asset_names.append(asset_name)
    print('filtered_asset_names:', filtered_asset_names)

    with open(asset_info_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)  # skip the header
        for i, row in enumerate(reader):
            basic_description, attributes, asset_name = row
            if 'merged_' + asset_name[:-4] + f"_edited_{i}.mp4" in os.listdir(out_dir):
                print(f"Skipping {asset_name} because it already exists in {out_dir}")
                continue
            if '63e80ac8ff7147ea84f898e7d0fa1d5a' in asset_name:
                print(f"Skipping {asset_name} to avoid OpenAI safety warning")
                continue
            if asset_name not in filtered_asset_names:
                print(f"Skipping {asset_name} because it is not in the filtered_asset_names")
                continue
            print('adding task:', basic_description, '|', attributes, '|', asset_name)
            task_seq.append((i, basic_description, attributes, asset_name))

    
    mp.set_start_method("spawn", force=True)

    failed_tasks = []

    # Build a list of worker slots, each mapped to a GPU (round-robin).
    # slot_id -> gpu_id
    slot_to_gpu = {slot: gpu_ids[slot % len(gpu_ids)] for slot in range(num_workers)}

    pending = list(task_seq)
    # active: slot_id -> (process, task)
    active: dict[int, tuple[mp.Process, tuple]] = {}

    while pending or active:
        # Fill available worker slots
        free_slots = [s for s in range(num_workers) if s not in active]
        while pending and free_slots:
            slot = free_slots.pop(0)
            gpu_id = slot_to_gpu[slot]
            task = pending.pop(0)
            p = mp.Process(
                target=_single_task_worker,
                args=(gpu_id, task, views_dir, image_pairs_dir, out_dir),
            )
            p.start()
            active[slot] = (p, task)

        # Wait for any process to finish (poll every 2s)
        import time
        time.sleep(2)
        finished_slots = []
        for slot, (p, task) in active.items():
            if not p.is_alive():
                p.join()
                i = task[0]
                gpu_id = slot_to_gpu[slot]
                if p.exitcode != 0:
                    print(f"[Slot {slot} / GPU {gpu_id}] WARNING: Task {i} crashed (exit code {p.exitcode}), skipping.")
                    failed_tasks.append(task)
                finished_slots.append(slot)
        for slot in finished_slots:
            del active[slot]

    if failed_tasks:
        print(f"\n{len(failed_tasks)} task(s) failed:")
        for (i, desc, attr, name) in failed_tasks:
            print(f"  Task {i}: {desc} | {attr} | {name}")
    else:
        print("All tasks completed successfully.")
    print('Done processing all tasks')


if __name__ == "__main__":
    batch_gen_edit_new_validation_dataset(
        asset_info_file="new_validation.csv",
        views_dir="multi_views_new_dataset_25",
        image_pairs_dir="new_val_edited_best_views_RM_edprp_mv40",
        out_dir="/data/ru_data/results/trellis_output/validation/edit_video_output_021126",
        gpu_ids=[0,1],
        num_workers=10,  # e.g. 4 processes all on GPU 1, or spread across multiple GPUs
    )
