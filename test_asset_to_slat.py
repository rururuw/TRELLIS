import os
import torch
import shutil
from typing import *
from PIL import Image
import csv
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
# os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import time
import imageio
import numpy as np
import open3d as o3d
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.pipelines.base import Pipeline
from trellis.utils import render_utils, postprocessing_utils
import trellis.models as models
import trellis.modules.sparse as sp
from torchvision import transforms
import torch.nn.functional as F
import utils3d
import json
from subprocess import DEVNULL, call
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'dataset_toolkits'))
from utils import sphere_hammersley_sequence

def get_pipeline():
    pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-large")
    pipeline.cuda()
    return pipeline

def load_slat(slat_path: str):
    slat = np.load(slat_path)
    coords, feats = slat["coords"], slat["feats"]
    coords = torch.from_numpy(coords).int()
    # Add back batch index (0) to coords! 
    if coords.shape[1] == 3:
        batch_idx = torch.zeros(coords.shape[0], 1, dtype=torch.int)
        coords = torch.cat([batch_idx, coords], dim=1)
    
    coords = coords.cuda()
    feats = torch.from_numpy(feats).float().cuda()
    return sp.SparseTensor(coords=coords, feats=feats)

def get_multi_view_from_3d_assets(asset_path: str, num_views: int, out_dir: str):
    """
    Gets multi-view images from a 3D asset.
    """
    BLENDER_PATH = 'blender' # Assumes blender is in PATH
    RENDER_SCRIPT = os.path.join(os.path.dirname(__file__), 'dataset_toolkits', 'blender_script', 'render.py')
    NUM_VIEWS = num_views
    RESOLUTION = 512

    # Generate camera poses
    yaws, pitchs = [], []
    offset = (np.random.rand(), np.random.rand())
    for i in range(NUM_VIEWS):
        y, p = sphere_hammersley_sequence(i, NUM_VIEWS, offset)
        yaws.append(y)
        pitchs.append(p)
    
    views = [{'yaw': y, 'pitch': p, 'radius': 2.0, 'fov': 40 / 180 * np.pi} 
                for y, p in zip(yaws, pitchs)]

    cmd = [
        BLENDER_PATH, '-b', '-P', RENDER_SCRIPT,
        '--',
        '--views', json.dumps(views),
        '--object', os.path.abspath(asset_path),
        '--resolution', str(RESOLUTION),
        '--output_folder', out_dir,
        '--engine', 'CYCLES',
        '--save_mesh' # Important: saves mesh.ply for voxelization
    ]
    print('Rendering command: ', cmd)
    ret = call(cmd, stdout=DEVNULL, stderr=DEVNULL)
    # ret = call(cmd)
    if ret != 0:
        print(f"Failed to render {asset_path}. Make sure Blender is installed and in PATH.")
        return False
    print(f"Successfully rendered {num_views} views from {asset_path}.")
    return True


def _render_views_worker(args: tuple) -> dict:
    """
    Worker function: renders a subset of views for an asset.
    Args: (asset_path, views_subset, resolution, temp_out_dir, worker_id, save_mesh)
    """
    asset_path, views_subset, resolution, temp_out_dir, worker_id, save_mesh = args
    
    BLENDER_PATH = 'blender'
    RENDER_SCRIPT = os.path.join(os.path.dirname(__file__), 'dataset_toolkits', 'blender_script', 'render.py')
    
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
        ret = call(cmd, stdout=DEVNULL, stderr=DEVNULL)
        if ret != 0:
            return {'worker_id': worker_id, 'status': 'failed', 'views': len(views_subset)}
        return {'worker_id': worker_id, 'status': 'success', 'views': len(views_subset), 'temp_dir': temp_out_dir}
    except Exception as e:
        return {'worker_id': worker_id, 'status': 'error', 'message': str(e)}


def get_multi_view_from_3d_assets_mp(
    asset_path: str, 
    num_views: int, 
    out_dir: str, 
    max_workers: int = None,
    random_seed: int = None
) -> bool:
    """
    Multiprocess version: Renders multiple views of a single 3D asset in parallel.
    
    Splits the views among multiple Blender processes to speed up rendering.
    
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
    
    # Generate all camera poses upfront with optional seeding
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        offset = (rng.random(), rng.random())
    else:
        offset = (np.random.rand(), np.random.rand())
    
    all_views = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        all_views.append({'yaw': y, 'pitch': p, 'radius': 2.0, 'fov': 40 / 180 * np.pi})
    
    # Split views among workers
    views_per_worker = []
    chunk_size = num_views // max_workers
    remainder = num_views % max_workers
    
    start_idx = 0
    for i in range(max_workers):
        # Distribute remainder among first workers
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        views_per_worker.append(all_views[start_idx:end_idx])
        start_idx = end_idx
    
    # Create temp directories for each worker
    temp_dirs = []
    worker_args = []
    for i, views_chunk in enumerate(views_per_worker):
        temp_dir = os.path.join(out_dir, f'_temp_worker_{i}')
        os.makedirs(temp_dir, exist_ok=True)
        temp_dirs.append(temp_dir)
        # Only first worker saves the mesh to avoid redundant work
        save_mesh = (i == 0)
        worker_args.append((asset_path, views_chunk, RESOLUTION, temp_dir, i, save_mesh))
    
    print(f"Rendering {num_views} views with {max_workers} parallel Blender processes...")
    
    # Launch parallel rendering
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(_render_views_worker, worker_args):
            status = result['status']
            if status == 'success':
                print(f"  Worker {result['worker_id']}: rendered {result['views']} views")
            else:
                print(f"  Worker {result['worker_id']}: {status} - {result.get('message', '')}")
            results.append(result)
    
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
    import shutil
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"Successfully rendered {num_views} views from {asset_path} using {max_workers} workers.")
    return True


def convert_3d_assets_to_slats(assets_dir: str, num_views: int, out_dir: str):
    """
    Converts 3D assets (GLB) in assets_dir to SLAT latents in out_dir.
    """
    

    # Configuration
    BLENDER_PATH = 'blender' # Assumes blender is in PATH
    RENDER_SCRIPT = os.path.join(os.path.dirname(__file__), 'dataset_toolkits', 'blender_script', 'render.py')
    NUM_VIEWS = num_views
    RESOLUTION = 512
    MAX_WORKERS = 4 # Number of parallel Blender instances
    
    # 1. Setup Models
    print("Loading models...")
    # DINOv2 for feature extraction
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    dinov2_model.eval().cuda()
    dinov2_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Sparse VAE Encoder
    encoder = models.from_pretrained('microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16')
    encoder.eval().cuda()

    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, 'temp_processing')
    os.makedirs(temp_dir, exist_ok=True)

    if os.path.isfile(assets_dir):
        assets = [os.path.basename(assets_dir)]
        assets_dir = os.path.dirname(assets_dir)
    else:
        assets = [f for f in os.listdir(assets_dir) if f.endswith('.glb')]
    
    for asset_name in assets:
        asset_path = os.path.join(assets_dir, asset_name)
        sha256 = os.path.splitext(asset_name)[0]
        print(f"Processing assets {asset_name}...")

        # Directories for this asset
        render_dir = os.path.join(temp_dir, 'renders', sha256)
        voxel_dir = os.path.join(temp_dir, 'voxels')
        os.makedirs(voxel_dir, exist_ok=True)

        # 2. Render Views
        if not os.path.exists(os.path.join(render_dir, 'transforms.json')):
            print("Rendering views...")
            # Generate camera poses
            yaws, pitchs = [], []
            offset = (np.random.rand(), np.random.rand())
            for i in range(NUM_VIEWS):
                y, p = sphere_hammersley_sequence(i, NUM_VIEWS, offset)
                yaws.append(y)
                pitchs.append(p)
            
            views = [{'yaw': y, 'pitch': p, 'radius': 2.0, 'fov': 40 / 180 * np.pi} 
                     for y, p in zip(yaws, pitchs)]

            cmd = [
                BLENDER_PATH, '-b', '-P', RENDER_SCRIPT,
                '--',
                '--views', json.dumps(views),
                '--object', os.path.abspath(asset_path),
                '--resolution', str(RESOLUTION),
                '--output_folder', render_dir,
                '--engine', 'CYCLES',
                '--save_mesh' # Important: saves mesh.ply for voxelization
            ]
            print('Rendering command: ', cmd)
            ret = call(cmd, stdout=DEVNULL, stderr=DEVNULL)
            # ret = call(cmd)
            if ret != 0:
                print(f"Failed to render {asset_name}. Make sure Blender is installed and in PATH.")
                continue

        # 3. Voxelize
        ply_path = os.path.join(render_dir, 'mesh.ply')
        voxel_path = os.path.join(voxel_dir, f'{sha256}.ply')
        print(f"Voxelizing {asset_name}...")
        if not os.path.exists(voxel_path):
            print("Voxelizing...")
            try:
                mesh = o3d.io.read_triangle_mesh(ply_path)
                vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                num_points = int(1e6) # 1 million points
                sampled_points = mesh.sample_points_uniformly(number_of_points=num_points)
                # voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
                #     mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
                voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
                    sampled_points, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
                vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
                # Normalize voxels to [-0.5, 0.5] based on grid index (0-63)
                # Note: voxelize.py does: (vertices + 0.5) / 64 - 0.5
                # This maps index 0 to -0.5 + 0.5/64, index 63 to 0.5 - 0.5/64 roughly
                # But extract_feature expects 0-63 indices. Let's save normalized coords as voxelize.py does.
                normalized_voxels = (vertices + 0.5) / 64 - 0.5
                utils3d.io.write_ply(voxel_path, normalized_voxels)
            except Exception as e:
                print(f"Voxelization failed: {e}")
                continue

        # 4. Extract Features
        print("Extracting Features...")
        # Load renders metadata
        with open(os.path.join(render_dir, 'transforms.json'), 'r') as f:
            meta = json.load(f)
        
        # Prepare data loader
        frames = meta['frames']
        n_views = len(frames)
        
        # Load voxel positions
        positions = utils3d.io.read_ply(voxel_path)[0]
        positions = torch.from_numpy(positions).float().cuda()
        # Convert back to indices 0-63 for projection
        indices = ((positions + 0.5) * 64).long()
        
        # Process in batches
        BATCH_SIZE = 16
        patchtokens_lst = []
        uv_lst = []
        
        with torch.no_grad():
            for i in range(0, n_views, BATCH_SIZE):
                batch_frames = frames[i:i+BATCH_SIZE]
                batch_images = []
                batch_extrinsics = []
                batch_intrinsics = []
                
                for frame in batch_frames:
                    # Load image
                    img_path = os.path.join(render_dir, frame['file_path'])
                    img = Image.open(img_path).resize((518, 518), Image.Resampling.LANCZOS)
                    img = np.array(img).astype(np.float32) / 255
                    img = img[:, :, :3] * img[:, :, 3:] # premultiply alpha
                    img = torch.from_numpy(img).permute(2, 0, 1).float()
                    batch_images.append(dinov2_transform(img))
                    
                    # Camera parameters
                    c2w = torch.tensor(frame['transform_matrix'])
                    c2w[:3, 1:3] *= -1 # Blender to OpenCV conversion likely
                    extrinsics = torch.inverse(c2w)
                    fov = frame['camera_angle_x']
                    intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
                    
                    batch_extrinsics.append(extrinsics)
                    batch_intrinsics.append(intrinsics)
                
                batch_images = torch.stack(batch_images).cuda()
                batch_extrinsics = torch.stack(batch_extrinsics).cuda()
                batch_intrinsics = torch.stack(batch_intrinsics).cuda()
                
                # Forward DINOv2
                features = dinov2_model(batch_images, is_training=True)
                
                # Project voxels to image
                # positions: [N, 3], extrinsics: [B, 4, 4], intrinsics: [B, 3, 3]
                # project_cv expects positions in world space
                uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
                
                # Extract features
                # features['x_prenorm']: [B, L, C] (L=1370 for 518x518 with patch 14 + registers)
                # DINOv2 reg tokens: 4
                n_patch = 518 // 14
                patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(len(batch_frames), 1024, n_patch, n_patch)
                
                patchtokens_lst.append(patchtokens)
                uv_lst.append(uv)

            patchtokens = torch.cat(patchtokens_lst, dim=0) # [V, C, H, W]
            uv = torch.cat(uv_lst, dim=0) # [V, N, 2]
            
            # Sample features for each voxel from each view
            # grid_sample expects [B, C, H, W] input and [B, N, 1, 2] grid
            # We want to sample for each view, then aggregate.
            # patchtokens: [V, C, H, W]
            # uv: [V, N, 2] -> [V, N, 1, 2]
            sampled_tokens = F.grid_sample(
                patchtokens,
                uv.unsqueeze(2),
                mode='bilinear',
                align_corners=False
            ).squeeze(3).permute(0, 2, 1) # [V, N, C]
            
            # Aggregate: Mean pooling over views
            # Note: extract_feature.py does simple mean. 
            # Ideally should handle occlusion, but replicating the script logic exactly:
            aggregated_features = torch.mean(sampled_tokens, dim=0) # [N, C]

        # 5. Encode Latents
        print("Encoding SLAT...")
        with torch.no_grad():
            sparse_feats = sp.SparseTensor(
                feats = aggregated_features.float(),
                coords = torch.cat([
                    torch.zeros(indices.shape[0], 1).int().cuda(),
                    indices.int()
                ], dim=1)
            )
            
            latent = encoder(sparse_feats, sample_posterior=False)
            
            # Save results
            save_path = os.path.join(out_dir, f'{sha256}.npz')
            np.savez_compressed(
                save_path,
                feats=latent.feats.cpu().numpy().astype(np.float32),
                coords=latent.coords[:, 1:].cpu().numpy().astype(np.uint8) # drop batch index
            )
            print(f"Saved SLAT to {save_path}")

    # Cleanup
    # 1/4/2026 no clean up for debugging
    # shutil.rmtree(temp_dir)
    print("Done.")

def convert_3d_assets_to_ss_latents(assets_dir: str, out_dir: str):
    """
    Converts 3D assets (GLB) in assets_dir to sparse structure latents in out_dir.
    """
    # Configuration
    # BLENDER_PATH = 'blender' # Assumes blender is in PATH
    # RENDER_SCRIPT = os.path.join(os.path.dirname(__file__), 'dataset_toolkits', 'blender_script', 'render.py')
    RESOLUTION = 64
    
    # 1. Setup Models
    print("Loading SS Encoder...")
    encoder = models.from_pretrained('microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16')
    encoder.eval().cuda()

    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, 'temp_processing_ss')
    os.makedirs(temp_dir, exist_ok=True)

    if os.path.isfile(assets_dir):
        assets = [os.path.basename(assets_dir)]
        assets_dir = os.path.dirname(assets_dir)
    else:
        assets = [f for f in os.listdir(assets_dir) if f.endswith('.glb')]
    
    for asset_name in assets:
        asset_path = os.path.join(assets_dir, asset_name)
        sha256 = os.path.splitext(asset_name)[0]
        print(f"Processing asset {asset_name}...")

        # Directories for this asset
        voxel_dir = os.path.join(temp_dir, 'voxels')
        os.makedirs(voxel_dir, exist_ok=True)

        # 2. Voxelize
        voxel_path = os.path.join(voxel_dir, f'{sha256}.ply')
        
        print(f"Voxelizing {asset_name}...")
        try:
            # Try loading GLB directly with Open3D
            # enable_post_processing=True usually helps merge meshes in the file
            mesh = o3d.io.read_triangle_mesh(asset_path, enable_post_processing=True)
            # here the glb is y-up, so we need to convert it to z-up by rotating 90 degrees counterclockwise around x axis
            R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
            mesh.rotate(R, center=(0, 0, 0))
            
            # Check if mesh is valid
            if len(mesh.vertices) == 0:
                print(f"Warning: Open3D loaded 0 vertices from {asset_path}. Trying without post-processing.")
                mesh = o3d.io.read_triangle_mesh(asset_path, enable_post_processing=False)

            # Normalize to unit cube centered at origin [-0.5, 0.5]
            # This mimics what the Blender script does implicitly by rendering in a normalized space,
            # or what typical preprocessing does.
            vertices = np.asarray(mesh.vertices)
            vmin = vertices.min(axis=0)
            vmax = vertices.max(axis=0)
            center = (vmin + vmax) / 2
            scale = (vmax - vmin).max()
            
            # Normalize
            vertices = (vertices - center) / scale
            # Clip to strictly inside [-0.5, 0.5] to avoid boundary issues during voxelization
            vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
            mesh.vertices = o3d.utility.Vector3dVector(vertices)

            num_points = int(1e6) 
            sampled_points = mesh.sample_points_uniformly(number_of_points=num_points)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
                sampled_points, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
            vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
            print(f"Vertices shape: {vertices.shape}")
            print(f"Vertices[0]: {vertices[0]}")
            # Normalize voxels to [-0.5, 0.5]
            normalized_voxels = (vertices + 0.5) / 64 - 0.5
            utils3d.io.write_ply(voxel_path, normalized_voxels)
        except Exception as e:
            print(f"Voxelization failed: {e}")
            continue
    
        # 3. Encode SS Latent
        print("Encoding SS Latent...")
        # try:
        position = normalized_voxels
        # print(f"Position: {position}")
        print(f"Position shape: {position.shape}")
        # Convert back to grid coords [0, 64)
        coords = ((torch.tensor(position) + 0.5) * RESOLUTION).int().contiguous()
        print(f"Coords shape: {coords.shape}")
        # Construct dense tensor
        # ss = torch.zeros(1, RESOLUTION, RESOLUTION, RESOLUTION, dtype=torch.long)
        # We need [Batch, Channels, D, H, W]. Batch=1, Channels=1
        ss = torch.zeros(1, 1, RESOLUTION, RESOLUTION, RESOLUTION, dtype=torch.long)
        print(f"SS shape: {ss.shape}")
        # Clip coords just in case
        # coords = torch.clamp(coords, 0, RESOLUTION - 1)
        # ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        ss[:, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        ss = ss.float().cuda()

        
        with torch.no_grad():
            latent = encoder(ss, sample_posterior=False)
            print(f"Latent shape: {latent.shape}")
        # Save results
        save_path = os.path.join(out_dir, f'{sha256}.npz')
        np.savez_compressed(
            save_path,
            mean=latent[0].cpu().numpy()
        )
        print(f"Saved SS latent to {save_path}")
            
        # except Exception as e:
        #     print(f"Encoding failed: {e}")
        #     continue

    print("Done.")

def slat_to_3D(slat_path: str, out_dir: str):
    """
    Converts a SLAT to a 3D mesh.
    """
    slat = load_slat(slat_path)
    print(f"Loaded SLAT from {slat_path}")
    print(f"SLAT feats shape: {slat.feats.shape}")
    print(f"SLAT coords shape: {slat.coords.shape}")
    pipeline = get_pipeline()
    outputs = pipeline.run_w_slat(slat)
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export(os.path.join(out_dir, f"{os.path.splitext(os.path.basename(slat_path))[0]}_reconstructed.glb"))

def ss_latent_to_3D(ss_latent_path: str, out_dir: str):
    """
    Converts a SS Latent to a 3D mesh.
    """
    ss_latent = np.load(ss_latent_path)['mean']
    # add one dim to the ss_latent
    ss_latent = ss_latent[None, :]
    print(f"Loaded SS Latent from {ss_latent_path}")
    print(f"SS Latent shape: {ss_latent.shape}")
    pipeline = get_pipeline()
    coords = pipeline.decode_ss_latent(torch.from_numpy(ss_latent).cuda().float())
    print(f"Coords shape: {coords.shape}")
    # drop batch index
    coords = coords[:, 1:]
    # from coords to voxels in point cloud. Each coords[i] is a voxel index (x,y,z) in the point cloud.
    # Generate a ply file from the voxels
    voxels = np.zeros((coords.shape[0], 3))
    for i in range(coords.shape[0]):
        voxels[i] = coords[i].cpu().numpy()
    print(f"Voxels shape: {voxels.shape}")

    # normalize voxels to [-0.5, 0.5]
    voxels = (voxels + 0.5) / 64 - 0.5
    
    out_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(ss_latent_path))[0]}_reconstructed.ply")
    utils3d.io.write_ply(out_path, voxels)
    print(f"Saved reconstructed mesh to {out_path}")

if __name__ == "__main__":
    num_view = 25

    # read assets from 3d-interior-val/new_dataset, guided by new_validation.csv
    assets_dir = '../3d-interior-val/new_dataset'
    asset_paths_dir = '../3d-interior-val/new_validation.csv'
    out_dir = 'validation/multi_views_new_dataset'
    with open(asset_paths_dir, 'r') as f:
        reader = csv.reader(f)
        # skip the first row
        next(reader)
        for row in reader:
            basic_description,attributes,asset_name = row
            asset_path = os.path.join(assets_dir, asset_name)
            views_dir = os.path.join(out_dir, asset_name[:-4] + '_views')
            if not os.path.exists(views_dir):
                os.makedirs(views_dir, exist_ok=True)
                get_multi_view_from_3d_assets_mp(asset_path, num_view, views_dir, max_workers=10)
                print(f"Saved views to {views_dir}")
            else:
                print(f"Views already exist for {asset_name}")
            
    # convert_3d_assets_to_slats(
    #     assets_dir='/home/rwang/3d-interior-val/3d-front-samples/Sofa_dce45bd9-5dd7-44fa-bd87-a272fec00cb5_1.glb', 
    #     num_views=num_view, 
    #     out_dir=f'outputs/test_seq_edit_slats_uniform_vox_{num_view}',
    # )
    # convert_3d_assets_to_slats(assets_dir='validation/test_seq_edit/A_bed.glb', num_views=num_view, out_dir=f'outputs/test_seq_edit_slats_uniform_vox_{num_view}')
    
    # slat_to_3D(slat_path='outputs/test_seq_edit_slats_uniform_vox_50/Bed_51d8f810-d983-4ab4-a460-0f3c4c8efa30_1.npz', out_dir='outputs/test_seq_edit_slats_uniform_vox_50')
    # slat_to_3D(slat_path='outputs/test_seq_edit_slats_uniform_vox_50/A_bed.npz', out_dir='outputs/test_seq_edit_slats_uniform_vox_50')
    
    # convert_3d_assets_to_ss_latents(
    #     assets_dir='/home/rwang/TRELLIS/outputs/test_slat_editing/editing_ss_slat_from_gen/a-pink-bed-made-of-rough-clothes-and-wood/ss_slat_gen_slider_glb_0.glb', 
    #     out_dir=f'outputs/test_seq_edit_ss_latents_uniform_vox'
    # )
    # time.sleep(2)
    # ss_latent_to_3D(ss_latent_path='outputs/test_seq_edit_ss_latents_uniform_vox/ss_slat_gen_slider_glb_0.npz', out_dir='outputs/test_seq_edit_ss_latents_uniform_vox')