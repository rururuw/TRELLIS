import os
import torch
import shutil
from typing import *
from PIL import Image
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

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
from concurrent.futures import ThreadPoolExecutor

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

def convert_3d_assets_to_slats(assets_dir: str, out_dir: str):
    """
    Converts 3D assets (GLB) in assets_dir to SLAT latents in out_dir.
    """
    import sys
    # Add dataset_toolkits to path to import utils if needed
    sys.path.append(os.path.join(os.path.dirname(__file__), 'dataset_toolkits'))
    from utils import sphere_hammersley_sequence

    # Configuration
    BLENDER_PATH = 'blender' # Assumes blender is in PATH
    RENDER_SCRIPT = os.path.join(os.path.dirname(__file__), 'dataset_toolkits', 'blender_script', 'render.py')
    NUM_VIEWS = 300
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
                voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
                    mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
                voxels = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
                # Normalize voxels to [-0.5, 0.5] based on grid index (0-63)
                # Note: voxelize.py does: (vertices + 0.5) / 64 - 0.5
                # This maps index 0 to -0.5 + 0.5/64, index 63 to 0.5 - 0.5/64 roughly
                # But extract_feature expects 0-63 indices. Let's save normalized coords as voxelize.py does.
                normalized_voxels = (voxels + 0.5) / 64 - 0.5
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

if __name__ == "__main__":
    # convert_3d_assets_to_slats(assets_dir='validation/test_seq_edit', out_dir='test_seq_edit_slats')
    slat_to_3D(slat_path='test_seq_edit_slats/A_bed.npz', out_dir='test_seq_edit_slats')