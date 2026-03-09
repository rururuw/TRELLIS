import os
import torch
import shutil
from typing import *
from PIL import Image
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import numpy as np
import imageio
import open3d as o3d
from trellis.pipelines import TrellisTextTo3DPipeline, TrellisAttributeSliderPipeline
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

def get_attribute_slider_pipeline():
    pipeline = TrellisAttributeSliderPipeline.from_pretrained("microsoft/TRELLIS-text-large")
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

def load_ss_latent(ss_latent_path: str):
    ss_latent = np.load(ss_latent_path)['mean']
    # add one dim to the ss_latent
    ss_latent = ss_latent[None, :]
    ss_latent = torch.from_numpy(ss_latent).float().cuda()
    return ss_latent

def slat_editing_from_slat(slat_path: str, out_dir_prefix: str):
    """
    Edits a SLAT.
    """
    slat = load_slat(slat_path)
    print(f"Loaded SLAT from {slat_path}")
    print(f"SLAT feats shape: {slat.feats.shape}")
    print(f"SLAT coords shape: {slat.coords.shape}")
    pipeline = get_attribute_slider_pipeline()
    out_dir = os.path.join(out_dir_prefix, slat_path.split('/')[-1].split('.')[0] + '_editing')
    # make sure the out_dir exists
    os.makedirs(out_dir, exist_ok=True)
    init_prompt = 'a bed'
    seed = 3
    pos_prompt = 'very japanese style'
    neg_prompt = 'very western style'
    slider_scales = [300, 275, 250, 225, 200, 175, 150, 125, 100, 50, 10, 0, -10, -50, -100, -150, -200, -250, -300]
    # slider_scales = [-100, -150, -175, -200, -225, -250, -300, -350]
    for slider_scale in slider_scales:

        print(f">>> Running with slider scale: {slider_scale}")
        outputs, slat_out, v_steps = pipeline.run_variant_from_slat(
            slat,
            pos_prompt,
            seed=seed,
            neg_prompt=neg_prompt,
            neutral_prompt=init_prompt,
            slat_sampler_params={
                "steps": 12,
                # "cfg_strength": cfg_strength,
            },
            slider_scale=slider_scale,
        )
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=0.95,          # Ratio of triangles to remove in the simplification process
            texture_size=1024,      # Size of the texture used for the GLB
        )
        glb.export(os.path.join(out_dir, f"edit_from_slat_slider_{slider_scale}.glb"))

        # Render the outputs
        video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
        imageio.mimsave(os.path.join(out_dir, f"slat_editing_video_{slider_scale}.mp4"), video, fps=25)

def read_base_mesh(path: str):
    base_mesh = o3d.io.read_triangle_mesh(path)
    return base_mesh

def edit_object(object_path: str, out_dir: str):
    """
    Edits an object.
    """
    object = read_base_mesh(object_path)
    print(f"Loaded object from {object_path}")
    pipeline = get_attribute_slider_pipeline()
    init_prompt = 'a bed'
    seed = 3
    pos_prompt = 'very maximalistic, traditional looking, with very complex patterns'
    neg_prompt = 'very minimalistic, modern looking, with very simple patterns'
    slider_scales = [10]
    for slider_scale in slider_scales:
        print(f">>> Running with slider scale: {slider_scale}")
        outputs = pipeline.run_variant(
            object,
            pos_prompt,
            seed=seed,
            neg_prompt=neg_prompt,
            neutral_prompt=init_prompt,
            slat_sampler_params={
                "steps": 12,
                # "cfg_strength": cfg_strength,
                },
            slider_scale=slider_scale,
        )
        video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
        imageio.mimsave(os.path.join(out_dir, f"object_editing_video_{slider_scale}.mp4"), video, fps=25)


def sequential_editing(slat_path: str, out_dir: str):
    slat = load_slat(slat_path)
    print(f"Loaded SLAT from {slat_path}")
    print(f"SLAT feats shape: {slat.feats.shape}")
    print(f"SLAT coords shape: {slat.coords.shape}")
    pipeline = get_attribute_slider_pipeline()
    init_prompt1 = 'a bed'
    seed = 3
    pos_prompt1 = 'very maximalistic, traditional looking, with very complex patterns'
    neg_prompt1 = 'very minimalistic, modern looking, with very simple patterns'
    outputs, slat_out, v_steps = pipeline.run_variant_from_slat(
        slat,
        pos_prompt1,
        seed=seed,
        neg_prompt=neg_prompt1,
        neutral_prompt=init_prompt1,
        slat_sampler_params={
            "steps": 12,
        # "cfg_strength": cfg_strength,
        },
        slider_scale=150,
    )

    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export(os.path.join(out_dir, f"seq_slat_editing_1_150.glb"))

    slat_out1 = slat_out
    v_steps1 = v_steps
    init_prompt2 = 'a bed'
    pos_prompt2 = 'very simplistic, futuristic looking, with very simple patterns'
    neg_prompt2 = 'very maximalist, traditional looking, with very complex patterns'
    
    outputs, slat_out, v_steps = pipeline.run_variant_from_slat(
        slat_out1,
        pos_prompt2,
        seed=seed,
        neg_prompt=neg_prompt2,
        neutral_prompt=init_prompt2,
        slat_sampler_params={
            "steps": 12,
        # "cfg_strength": cfg_strength,
        },
        slider_scale=200,
        v_steps_inv=v_steps1,
    )
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export(os.path.join(out_dir, f"seq_slat_editing_2_150.glb"))

    # Render the outputs
    video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
    video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
    video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
    imageio.mimsave(os.path.join(out_dir, f"seq_slat_editing_video_200.mp4"), video, fps=25)

def ss_slat_editing_from_gen(slat_path: str, out_dir_prefix: str):
    slat = load_slat(slat_path)
    print(f"Loaded SLAT from {slat_path}")
    print(f"SLAT feats shape: {slat.feats.shape}")
    print(f"SLAT coords shape: {slat.coords.shape}")
    pipeline = get_attribute_slider_pipeline()
    print('sparse structure sampler: ', pipeline.sparse_structure_sampler)
    init_prompt = 'a pink bed made of rough clothes and wood'
    out_dir = os.path.join(out_dir_prefix, init_prompt.replace(' ', '-'))
    os.makedirs(out_dir, exist_ok=True)
    seed = 3
    pos_prompt = 'very simplistic, futuristic looking, with very simple patterns'
    neg_prompt = 'very maximalist, traditional looking, with very complex patterns'
    slider_scales = [0]
    for slider_scale in slider_scales:
        print(f">>> Running with slider scale: {slider_scale}")
        outputs = pipeline.run_variant_prompt_to_ss_slat(
            pos_prompt,
            seed=seed,
            neg_prompt=neg_prompt,
            neutral_prompt=init_prompt,
            slider_scale=slider_scale,
        )
        video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
        imageio.mimsave(os.path.join(out_dir, f"ss_slat_gen_slider_video_{slider_scale}.mp4"), video, fps=25)

        # save glb
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=0.95,          # Ratio of triangles to remove in the simplification process
            texture_size=1024,      # Size of the texture used for the GLB
        )
        glb.export(os.path.join(out_dir, f"ss_slat_gen_slider_glb_{slider_scale}.glb"))

def ss_slat_editing_from_ss(ss_latent_path: str, out_dir_prefix: str):
    ss_latent = load_ss_latent(ss_latent_path)
    print(f"Loaded SS Latent from {ss_latent_path}")
    print(f"SS Latent shape: {ss_latent.shape}")
    pipeline = get_attribute_slider_pipeline()
    init_prompt = 'a pink bed made of rough clothes and wood'
    out_dir = os.path.join(out_dir_prefix, init_prompt.replace(' ', '-'))
    os.makedirs(out_dir, exist_ok=True)
    seed = 3
    pos_prompt = 'very simplistic, futuristic looking, with very simple patterns'
    neg_prompt = 'very maximalist, traditional looking, with very complex patterns'
    slider_scales = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    for slider_scale in slider_scales:
        print(f">>> Running with slider scale: {slider_scale}")
        outputs, z_s, v_steps_ss = pipeline.run_variant_from_ss(
            ss_latent,
            pos_prompt,
            neg_prompt=neg_prompt,
            neutral_prompt=init_prompt,
            seed=seed,
            slider_scale=slider_scale,
        )
        video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
        imageio.mimsave(os.path.join(out_dir, f"ss_slat_editing_from_ss_latent_video_{slider_scale}.mp4"), video, fps=25)

def ss_slat_editing_from_ss_slat(ss_latent_path: str, slat_path: str, out_dir_prefix: str):
    slat = load_slat(slat_path)
    print(f"Loaded SLAT from {slat_path}")
    print(f"SLAT feats shape: {slat.feats.shape}")
    print(f"SLAT coords shape: {slat.coords.shape}")
    ss_latent = load_ss_latent(ss_latent_path)
    print(f"Loaded SS Latent from {ss_latent_path}")
    print(f"SS Latent shape: {ss_latent.shape}")
    pipeline = get_attribute_slider_pipeline()
    init_prompt = 'a bed' # should not be used
    out_dir = os.path.join(out_dir_prefix, init_prompt.replace(' ', '-'))
    os.makedirs(out_dir, exist_ok=True)
    seed = 3
    pos_prompt = 'very simplistic, futuristic looking, with very simple patterns'
    neg_prompt = 'very maximalist, traditional looking, with very complex patterns'
    slider_scales = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    for slider_scale in slider_scales:
        print(f">>> Running with slider scale: {slider_scale}")
        outputs, z_s, v_steps_ss, new_slat, v_steps_slat = pipeline.run_variant_from_ss_slat(
            ss_latent,
            slat,
            pos_prompt,
            seed=seed,
            neg_prompt=neg_prompt,
            neutral_prompt=init_prompt,
            slider_scale=slider_scale,
        )
        video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
        imageio.mimsave(os.path.join(out_dir, f"ss_slat_editing_from_ss_latent_video_{slider_scale}.mp4"), video, fps=25)



if __name__ == "__main__":
    # slat_path = "outputs/test_seq_edit_slats_uniform_vox_50/A_bed.npz"
    # slat_path = "outputs/test_seq_edit_slats_uniform_vox/Bed_51d8f810-d983-4ab4-a460-0f3c4c8efa30_1.npz"
    slat_path = '/home/rwang/TRELLIS/outputs/test_seq_edit_slats_uniform_vox_50/ss_slat_gen_slider_glb_0.npz'
    object_path = "validation/basic_objects/A_bed.ply"
    out_dir_prefix = "outputs/test_slat_editing/editing_ss_slat_from_ss"
    # out_dir_prefix = "outputs/test_slat_editing/editing_ss_slat_from_ss_slat"

    os.makedirs(out_dir_prefix, exist_ok=True)
    # slat_editing_from_slat(slat_path, out_dir_prefix)
    ss_latent_path = "outputs/test_seq_edit_ss_latents_uniform_vox/ss_slat_gen_slider_glb_0.npz"
    ss_slat_editing_from_ss(ss_latent_path, out_dir_prefix)
    # edit_object(object_path, out_dir)
    # sequential_editing(slat_path, out_dir)
    # ss_slat_editing(slat_path, out_dir)

    # ss_slat_editing_from_gen(slat_path, out_dir_prefix)
    # ss_latent_path = "outputs/test_seq_edit_ss_latents_uniform_vox/Bed_51d8f810-d983-4ab4-a460-0f3c4c8efa30_1.npz"
    # ss_slat_editing_from_ss_slat(ss_latent_path, slat_path, out_dir_prefix)