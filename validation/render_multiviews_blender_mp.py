import os
from typing import *

import numpy as np
import open3d as o3d
import json
import subprocess
import shutil
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import time
import csv
import sys
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset_toolkits'))
from utils import sphere_hammersley_sequence

BLENDER_PATH = 'blender'
RENDER_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset_toolkits', 'blender_script', 'render.py')


def _render_views_worker(args: tuple) -> dict:
    """
    Worker function: renders a subset of views for an asset.
    Each worker is pinned to a specific GPU via CUDA_VISIBLE_DEVICES so
    Blender processes don't all compete for every GPU.
    """
    asset_path, views_subset, resolution, temp_out_dir, worker_id, save_mesh, gpu_id = args

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = [
        BLENDER_PATH, '-b', '-P', RENDER_SCRIPT,
        '--',
        '--views', json.dumps(views_subset),
        '--object', os.path.abspath(asset_path),
        '--resolution', str(resolution),
        '--output_folder', temp_out_dir,
        '--engine', 'CYCLES',
    ]
    if save_mesh:
        cmd.append('--save_mesh')

    try:
        ret = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
        ).returncode
        if ret != 0:
            return {'worker_id': worker_id, 'status': 'failed', 'views': len(views_subset)}
        return {'worker_id': worker_id, 'status': 'success', 'views': len(views_subset)}
    except Exception as e:
        return {'worker_id': worker_id, 'status': 'error', 'message': str(e)}


def _detect_gpu_count() -> int:
    """Return the number of NVIDIA GPUs visible to the system."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL,
        )
        return max(1, len(out.decode().strip().splitlines()))
    except Exception:
        return 1


def get_multi_view_from_3d_assets_mp(
    asset_path: str, 
    num_views: int, 
    out_dir: str, 
    max_workers: int = None,
    random_seed: int = None,
    eval_mode: bool = False
) -> bool:
    """
    Multiprocess version: Renders multiple views of a single 3D asset in parallel.
    
    Uses multiprocessing.Pool to launch Blender subprocesses.
    Each worker is pinned to a specific GPU (round-robin) to avoid GPU contention.
    
    Args:
        asset_path: Path to the 3D asset file (GLB, OBJ, etc.)
        num_views: Total number of views to render
        out_dir: Output directory for rendered images
        max_workers: Number of parallel Blender processes (default: min(num_views, cpu_count-2))
        random_seed: Random seed for camera pose generation (default: None for random)
    
    Returns:
        True if successful, False otherwise
    """
    RESOLUTION = 512
    
    if max_workers is None:
        max_workers = max(1, min(num_views, cpu_count() - 2))
    
    # Ensure we don't have more workers than views
    max_workers = min(max_workers, num_views)
    
    os.makedirs(out_dir, exist_ok=True)
    
    num_gpus = _detect_gpu_count()
    
    all_views = []
    if eval_mode:
        # Generate camera poses
        yaws = np.arange(0, 360, 360 / num_views)
        pitchs = [30] * len(yaws)
        radius = 2.0
        # deg to rad
        yaws = np.deg2rad(yaws)
        pitchs = np.deg2rad(pitchs)
        all_views = [{'yaw': float(y), 'pitch': float(p), 'radius': radius, 'fov': 40 / 180 * np.pi} 
                    for y, p in zip(yaws, pitchs)]
    else:
        # Generate all camera poses upfront with optional seeding
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
            offset = (rng.random(), rng.random())
        else:
            offset = (np.random.rand(), np.random.rand())
        for i in range(num_views):
            y, p = sphere_hammersley_sequence(i, num_views, offset)
            all_views.append({'yaw': y, 'pitch': p, 'radius': 2.0, 'fov': 40 / 180 * np.pi})
    
    # Split views among workers
    views_per_worker = []
    chunk_size = len(all_views) // max_workers
    remainder = len(all_views) % max_workers
    
    start_idx = 0
    for i in range(max_workers):
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        views_per_worker.append(all_views[start_idx:end_idx])
        start_idx = end_idx
    
    # Create temp directories and worker argument tuples
    temp_dirs = []
    worker_args = []
    for i, views_chunk in enumerate(views_per_worker):
        temp_dir = os.path.join(out_dir, f'_temp_worker_{i}')
        os.makedirs(temp_dir, exist_ok=True)
        temp_dirs.append(temp_dir)
        save_mesh = False
        gpu_id = i % num_gpus  # round-robin GPU assignment
        worker_args.append((asset_path, views_chunk, RESOLUTION, temp_dir, i, save_mesh, gpu_id))
    
    print(f"Rendering {num_views} views with {max_workers} workers across {num_gpus} GPU(s)...")
    
    # Launch parallel rendering with multiprocessing.Pool
    with Pool(processes=max_workers) as pool:
        results = pool.map(_render_views_worker, worker_args)
    
    for r in results:
        status = r['status']
        if status == 'success':
            print(f"  Worker {r['worker_id']}: rendered {r['views']} views")
        else:
            print(f"  Worker {r['worker_id']}: {status} - {r.get('message', '')}")
    
    # Check if all workers succeeded
    if not all(r['status'] == 'success' for r in results):
        print(f"Failed to render some views for {asset_path}")
        return False
    
    # Merge results: combine transforms.json and move images
    print("Merging results...")
    combined_frames = []
    view_counter = 0
    
    for i, temp_dir in enumerate(temp_dirs):
        transforms_path = os.path.join(temp_dir, 'transforms.json')
        if os.path.exists(transforms_path):
            with open(transforms_path, 'r') as f:
                data = json.load(f)
            
            # Rename and move images to final directory with sequential numbering
            for frame in data['frames']:
                old_path = os.path.join(temp_dir, frame['file_path'])
                new_filename = f'{view_counter:03d}.png'
                new_path = os.path.join(out_dir, new_filename)
                
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                
                # Update frame info
                frame['file_path'] = new_filename
                combined_frames.append(frame)
                view_counter += 1
            
            # Copy mesh from first worker
            if i == 0:
                mesh_src = os.path.join(temp_dir, 'mesh.ply')
                mesh_dst = os.path.join(out_dir, 'mesh.ply')
                if os.path.exists(mesh_src):
                    os.rename(mesh_src, mesh_dst)
    
    # Write combined transforms.json
    # Use camera parameters from first worker's data (they should all be same except frames)
    with open(os.path.join(temp_dirs[0], 'transforms.json'), 'r') as f:
        base_data = json.load(f)
    
    base_data['frames'] = combined_frames
    with open(os.path.join(out_dir, 'transforms.json'), 'w') as f:
        json.dump(base_data, f, indent=2)
    
    # Cleanup temp directories
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"Successfully rendered {num_views} views from {asset_path} using {max_workers} workers.")
    return True

def test_num_views_vs_num_workers():
    # generate a testing script to test the number of views vs the number of workers
    num_views_list = [25, 50, 75, 100]
    num_workers_ratios = [0, 1/10, 1/8, 1/6, 1/5, 1/4, 1/3, 1/2, 1]
    total_process = cpu_count()

    testing_asset_path = 'new_dataset/obv_bmy_bd75b4485e3346a2a0899668cfcd7411.glb'
    testing_out_dir = 'test_mp_results'
    time_results = []
    for num_views in num_views_list:
        for num_worker_ratio in num_workers_ratios:
            num_workers = int(num_views * num_worker_ratio)
            if num_workers > total_process:
                num_workers = total_process
            if num_workers == 0:
                num_workers = 1
            out_dir = os.path.join(testing_out_dir, f'num_views_{num_views}_num_workers_{num_workers}_ratio_{num_worker_ratio}')
            os.makedirs(out_dir, exist_ok=True)
            print(f'Start rendering... num_views: {num_views}, num_workers: {num_workers}, num_worker_ratio: {num_worker_ratio}')
            time_start = time.time()
            get_multi_view_from_3d_assets_mp(testing_asset_path, num_views, out_dir, num_workers)
            time_end = time.time()
            print(f'num_views: {num_views}, num_workers: {num_workers}, time: {time_end - time_start}')
            time_results.append([num_views, num_workers, num_worker_ratio, time_end - time_start])
            # write to file as we go
            with open(os.path.join(testing_out_dir, 'time_results.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([num_views, num_workers, num_worker_ratio, time_end - time_start])
    print('Done testing')

def plot_time_results():
    testing_out_dir = 'test_mp_results'
    # plot the time results
    time_results = []
    # read results into a pandas dataframe, add headers
    df = pd.read_csv(os.path.join(testing_out_dir, 'time_results.csv'), names=['num_views', 'num_workers', 'num_worker_ratio', 'time'])
    # x: number of workers, y: time, legend: different number of views
    for num_views in df['num_views'].unique():
        # filter the time results by num_views
        time_results_num_views = df[df['num_views'] == num_views]
        plt.plot(time_results_num_views['num_worker_ratio'], time_results_num_views['time'], label=f'{num_views} views')
    plt.legend()
    plt.xlabel('Number of Workers Ratio')
    plt.ylabel('Time (s)')
    plt.title('Time vs Number of Workers')
    plt.savefig(os.path.join(testing_out_dir, f'time_vs_num_workers_views.png'))

def render_multiviews_from_3d_assets(assets_dir: str, asset_paths_dir: str, num_views: int, out_dir: str):

    # read assets from 3d-interior-val/new_dataset, guided by new_validation.csv
    # assets_dir = '../3d-interior-val/new_dataset'
    # asset_paths_dir = '../3d-interior-val/new_validation.csv'
    # out_dir = 'validation/multi_views_new_dataset'
    with open(asset_paths_dir, 'r') as f:
        reader = csv.reader(f)
        # skip the first row
        next(reader)
        for row in reader:
            basic_description,attributes,asset_name = row
            asset_path = os.path.join(assets_dir, asset_name)
            views_dir = os.path.join(out_dir, asset_name[:-4] + '_views')
            if not os.path.exists(views_dir) or len([f for f in os.listdir(views_dir) if f.endswith('.png')]) != num_views:
                os.makedirs(views_dir, exist_ok=True)
                get_multi_view_from_3d_assets_mp(asset_path, num_views, views_dir, max_workers=int(num_views * 0.2))
                print(f"Saved views to {views_dir}")
            else:
                print(f"Views already exist for {asset_name}")

if __name__ == '__main__':
    # test_num_views_vs_num_workers()
    # plot_time_results()
    num_views = [30, 40, 50]
    for num_view in num_views:
        render_multiviews_from_3d_assets(
            assets_dir='new_dataset', 
            asset_paths_dir='new_validation.csv', 
            num_views=num_view, 
            out_dir=f'multi_views_new_dataset_{num_view}')