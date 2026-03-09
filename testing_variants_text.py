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
from trellis.pipelines import TrellisTextTo3DPipeline, TrellisAttributeSliderPipeline, TrellisImageTo3DAttributeSliderPipeline
from trellis.pipelines.base import Pipeline
from trellis.utils import render_utils, postprocessing_utils

def get_pipeline():
    pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-large")
    pipeline.cuda()
    return pipeline

def get_attribute_slider_pipeline():
    pipeline = TrellisAttributeSliderPipeline.from_pretrained("microsoft/TRELLIS-text-large")
    pipeline.cuda()
    return pipeline

def get_image_to_3d_attribute_slider_pipeline():
    pipeline = TrellisImageTo3DAttributeSliderPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    return pipeline

def generate_mesh(pipeline: Pipeline, prompt: Union[str, Image.Image], seed: int): 
    # "A living room scene with a table, a chair, and a sofa, 4k quality, no walls or ceiling."
    # Load a pipeline from a model folder or a Hugging Face model hub.
    print("Loading pipeline...")
    # print sparse structure sampler parameters
    print("Sparse structure sampler used:", pipeline.sparse_structure_sampler)
    # print which slat sampler is used
    print("Slat sampler used:", pipeline.slat_sampler)

    # Run the pipeline
    print("Running pipeline.run()...")
    outputs = pipeline.run(
        # "A chair looking like a avocado.",
        prompt,
        seed=seed,

    )
    return outputs

def get_base_mesh(outputs):
    mesh_result = outputs['mesh'][0]
    vertices_np = mesh_result.vertices.cpu().numpy()
    faces_np = mesh_result.faces.cpu().numpy()
    base_mesh = o3d.geometry.TriangleMesh()
    base_mesh.vertices = o3d.utility.Vector3dVector(vertices_np.astype(np.float64))
    base_mesh.triangles = o3d.utility.Vector3iVector(faces_np.astype(np.int32))

    # Save to file and reload as a workaround for Open3D bugs
    print("Saving mesh to temp file as workaround...")
    o3d.io.write_triangle_mesh("temp_mesh.ply", base_mesh)
    print("Reloading mesh...")
    base_mesh = o3d.io.read_triangle_mesh("temp_mesh.ply")
    print("Mesh reloaded successfully")

    return base_mesh

def read_base_mesh(path: str):
    base_mesh = o3d.io.read_triangle_mesh(path)
    return base_mesh

def generate_variants(pipeline: TrellisTextTo3DPipeline, base_mesh: o3d.geometry.TriangleMesh, prompt: str, neg_prompt: str, seed: int, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    print("Running pipeline.run_variant()...")
    # cfg_strengths = np.linspace(-20, 20, 9)
    cfg_strengths = [-5, 0, 5]
    for cfg_strength in cfg_strengths:
        print(f"Running pipeline.run_variant() with cfg_strength={cfg_strength}...")
        outputs = pipeline.run_variant(
            base_mesh,
            prompt,
            seed=seed,
            neg_prompt=neg_prompt,
            slat_sampler_params={
                "steps": 24,
                "cfg_strength": cfg_strength,
            },
        )

        # Render the outputs
        video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
        imageio.mimsave(os.path.join(outdir, f"sample_variant_cfg_{cfg_strength}.mp4"), video, fps=25)

def generate_variants_single_attribute(pipeline: TrellisAttributeSliderPipeline, base_mesh: o3d.geometry.TriangleMesh, prompt: str, neg_prompt: str, neutral_prompt: str, seed: int, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    print("Running pipeline.run_variant()...")
    # slider_scales = np.linspace(-10, 10, 9)
    # slider_scales = np.linspace(-20, 20, 9)
    slider_scales = [-5, 0, 5]
    for slider_scale in slider_scales:
        print(f"Running pipeline.run_variant() with slider_scale={slider_scale}...")
        outputs = pipeline.run_variant(
            base_mesh,
            prompt,
            seed=seed,
            neg_prompt=neg_prompt,
            neutral_prompt=neutral_prompt,
            slat_sampler_params={
                "steps": 12,
                # "cfg_strength": cfg_strength,
            },
            slider_scale=slider_scale,
        )

        # Render the outputs
        video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
        imageio.mimsave(os.path.join(outdir, f"sample_variant_slider_{slider_scale}.mp4"), video, fps=25)

        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=0.95,          # Ratio of triangles to remove in the simplification process
            texture_size=1024,      # Size of the texture used for the GLB
        )
        glb.export(os.path.join(outdir, f"sample_variant_slider_{slider_scale}.glb"))
        # outputs['gaussian'][0].save_ply(os.path.join(outdir, f"sample_variant_slider_{slider_scale}.ply"))

def generate_variants_single_attribute_from_prompt( # for testing steps from. Result: not working
    pipeline: TrellisAttributeSliderPipeline, 
    prompt: str, 
    neg_prompt: str, 
    neutral_prompt: str, 
    seed: int, 
    outdir: str,
    slider_scales: List[float] = None,
    save_to_glb: bool = False,
):
    os.makedirs(outdir, exist_ok=True)
    print("Running pipeline.run_variant()...")
    if slider_scales is None:
        slider_scales = np.linspace(-5, 5, 11)
    for slider_scale in slider_scales:
        print(f"Running pipeline.run_variant() with slider_scale={slider_scale}...")
        outputs = pipeline.run_variant_prompt_to_ss_slat(
            prompt=prompt,
            seed=seed,
            neg_prompt=neg_prompt,
            neutral_prompt=neutral_prompt,
            # Optional parameters
            sparse_structure_sampler_params={
                "steps": 12,
                "cfg_strength": 4,
            },
            slat_sampler_params={
                "steps": 12,
                "cfg_strength": 4,
            },
            slider_scale=slider_scale
        )

        # Render the outputs
        video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
        imageio.mimsave(os.path.join(outdir, f"slider_{slider_scale}.mp4"), video, fps=25)

        if save_to_glb:
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                # Optional parameters
                simplify=0.95,          # Ratio of triangles to remove in the simplification process
                texture_size=1024,      # Size of the texture used for the GLB
            )
            glb.export(os.path.join(outdir, f"slider_{slider_scale}.glb"))

def generate_variants_single_attribute_image(
    pipeline: TrellisImageTo3DAttributeSliderPipeline, 
    base_mesh: o3d.geometry.TriangleMesh, 
    positive_image: Image.Image, 
    negative_image: Image.Image, 
    neutral_image: Image.Image, 
    seed: int, 
    span: int, 
    spacing: int,
    outdir: str
) -> None:
    os.makedirs(outdir, exist_ok=True)
    print("Running pipeline.run_variant()...")
    steps = span // spacing * 2 + 1
    slider_scales = np.linspace(-span, span, steps)
    for slider_scale in slider_scales:
        print(f"Running pipeline.run_variant() with slider_scale={slider_scale}...")
        outputs = pipeline.run_variant(
            base_mesh,
            positive_image=positive_image,
            negative_image=negative_image,
            neutral_image=neutral_image,
            seed=seed,
            slat_sampler_params={
                "steps": 12,
                # "cfg_strength": cfg_strength,
            },
            slider_scale=slider_scale,
        )

        # Render the outputs
        video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
        video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
        imageio.mimsave(os.path.join(outdir, f"sample_variant_slider_{slider_scale}.mp4"), video, fps=25)

        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=0.95,          # Ratio of triangles to remove in the simplification process
            texture_size=1024,      # Size of the texture used for the GLB
        )
        glb.export(os.path.join(outdir, f"sample_variant_slider_{slider_scale}.glb"))


def run_test(from_scratch: bool = True, tag: str = '', mesh_file_name: str = 'temp_mesh.ply'):
    init_prompt = 'A egg-shaped chair'
    pos_prompt = init_prompt + ', very maximalistic, traditional looking, with very complex patterns'
    neg_prompt = init_prompt + ', very minimalistic, modern looking, with very simple patterns'
    pipeline = get_pipeline()

    if from_scratch:
        generated_mesh = generate_mesh(pipeline, init_prompt, 42)
        # workaround for Open3D bugs
        base_mesh = get_base_mesh(generated_mesh)
    else:
        base_mesh = read_base_mesh(mesh_file_name)
    

    outdir = 'variants_original/{}'.format((init_prompt + '+' + pos_prompt).replace(" ", "_"))
    generate_variants(pipeline, base_mesh, pos_prompt, neg_prompt, 3, outdir)
    # generate_variants(pipeline, base_mesh, "A very modern blue sofa.", None, 42, "variants")

def run_test_single_attribute(from_scratch: bool = True, tag: str = '', mesh_file_name: str = 'temp_mesh.ply'):
    # init_prompt = 'A blue sofa' # very basic neutral prompt
    pos_prompt = 'very maximalistic, traditional looking, with very complex patterns'
    neg_prompt = 'very minimalistic, modern looking, with very simple patterns'
    # init_prompt = 'A egg-shaped chair'
    init_prompt = 'A bed made of composite board in white color'
    # init_prompt = 'A living room with a sofa and an accent table'
    # other_prompts = ['red', 'blue', 'green', 'yellow', 'round', 'rectangular', 'leather', 'fabric', 'metal', 'plastic']

    if from_scratch:
        basic_pipeline = get_pipeline()
        generated_mesh = generate_mesh(basic_pipeline, init_prompt, 42)
        # workaround for Open3D bugs
        base_mesh = get_base_mesh(generated_mesh)
    else:
        base_mesh = read_base_mesh(mesh_file_name)

    pipeline = get_attribute_slider_pipeline()
    outdir = 'variants_single_attribute/{}{}'.format((tag + '_') if tag else '', (init_prompt + '+' + pos_prompt).replace(" ", "_"))
    generate_variants_single_attribute(pipeline, base_mesh, pos_prompt, neg_prompt, init_prompt, 3, outdir)

def run_test_single_attribute_image(from_scratch: bool = True, tag: str = '', span: int = 20, spacing: int = 5, mesh_file_name: str = 'temp_mesh.ply'):
    pos_image = Image.open('assets/example_sofa/traditional_sofa_2_sam.png')
    neg_image = Image.open('assets/example_sofa/modern_sofa_2_sam.png')
    neutral_image = Image.open('assets/example_sofa/neutral_sofa_sam.png')
    pipeline = get_image_to_3d_attribute_slider_pipeline()
    if from_scratch:
        generated_mesh = generate_mesh(pipeline, neutral_image, 42)
        # workaround for Open3D bugs
        base_mesh = get_base_mesh(generated_mesh)
    else:
        base_mesh = read_base_mesh(mesh_file_name)

    # get image names
    neutral_image_name = neutral_image.filename.split('/')[-1].split('.')[0]
    pos_image_name = pos_image.filename.split('/')[-1].split('.')[0]
    neg_image_name = neg_image.filename.split('/')[-1].split('.')[0]
    outdir = 'variants_single_attribute_image/{}{}'.format((tag + '_') if tag else '', (neutral_image_name + '+' + pos_image_name + '+' + neg_image_name).replace(" ", "_"))
    generate_variants_single_attribute_image(pipeline, base_mesh, pos_image, neg_image, neutral_image, 3, span, spacing, outdir)

def run_test_single_attribute_from_prompt(tag: str = ''): # for testinf steps from. Result: not working
    neutral_prompt = 'A male doctor with green hair and blue eyes'
    pos_prompt = 'his eyes are very big and round'
    neg_prompt = 'his eyes are very small'
    
    pipeline = get_attribute_slider_pipeline()
    outdir = 'outputs/diversity_test/{}{}'.format((tag + '_') if tag else '', (pos_prompt + '+' + neg_prompt + '+' + neutral_prompt).replace(" ", "_"))
    generate_variants_single_attribute_from_prompt(pipeline, pos_prompt, neg_prompt, neutral_prompt, 42, outdir, slider_scales=[-3, 0, 3], save_to_glb=True)

if __name__ == "__main__":
    # run_test(False)
    # run_test_single_attribute(False, 'quality_test', 'test_base_mesh.ply')
    # span = 5
    # spacing = 1
    # run_test_single_attribute_image(False, 'test_uncond_neutral_pn_slider_{}_{}'.format(span, spacing), span, spacing, 'assets/example_sofa/a_blue_sofa.ply')
    run_test_single_attribute_from_prompt('test_torch_dt')