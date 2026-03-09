import os
import torch
import shutil
from typing import *
from PIL import Image

import imageio
import numpy as np
import trimesh
import open3d as o3d
import sys
# go to the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

def generate_basic_object(pipeline: Pipeline, prompt: str, seed: int, outdir: str):
    print("generating basic object: ", prompt)
    outputs = pipeline.run(
        # "A chair looking like a avocado.",
        prompt,
        seed=seed,
        # num_samples=3,
    )

    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    obj_base_path = os.path.join(outdir, f"{prompt.replace(' ', '_').replace('.', '')}")
    glb.export(obj_base_path + '.glb')
    print("basic object generated: ", obj_base_path + '.glb')

    # Save Gaussians as PLY files
    # outputs['gaussian'][0].save_ply(os.path.join(outdir, f"{prompt.replace(' ', '_')}.ply"))
    write_base_mesh(outputs, obj_base_path + '.ply')

def read_base_mesh(path: str):
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=False)
    return mesh

def write_base_mesh(outputs, outpath):
    mesh_result = outputs['mesh'][0]
    vertices_np = mesh_result.vertices.cpu().numpy()
    faces_np = mesh_result.faces.cpu().numpy()
    base_mesh = o3d.geometry.TriangleMesh()
    base_mesh.vertices = o3d.utility.Vector3dVector(vertices_np.astype(np.float64))
    base_mesh.triangles = o3d.utility.Vector3iVector(faces_np.astype(np.int32))

    # Save to file and reload as a workaround for Open3D bugs
    print("Saving base mesh...")
    o3d.io.write_triangle_mesh(outpath, base_mesh)

    return base_mesh

def get_mesh_density(glb_path):
    # load the glb file
    try:
        mesh = trimesh.load(glb_path, force='mesh')
    except Exception as e:
        print(f"Error loading {glb_path}: {e}")
        return 0
    
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            return 0
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        
    area = mesh.area
    if area < 1e-6:
        return 0
    return len(mesh.faces) / area

def generate_variants_single_attribute(
        pipeline: TrellisAttributeSliderPipeline, 
        base_mesh: o3d.geometry.TriangleMesh, 
        prompt: str, 
        neg_prompt: str, 
        neutral_prompt: str, 
        seed: int, 
        outdir: str, 
        mesh_density: int
    ):
    os.makedirs(outdir, exist_ok=True)
    print("Running pipeline.run_variant()...")
    slider_scales = np.linspace(-10, 10, 21)
    # slider_scales = np.linspace(-20, 20, 9)
    for slider_scale in slider_scales:
        print(f"Running pipeline.run_variant() with slider_scale={slider_scale} for {neutral_prompt}...")
        outputs = pipeline.run_variant(
            base_mesh,
            prompt,
            seed=seed,
            neg_prompt=neg_prompt,
            neutral_prompt=neutral_prompt,
            slat_sampler_params={
                "steps": 12,
                "cfg_strength": 3,
            },
            slider_scale=slider_scale,
        )

        # Render the outputs
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=0.95,          # Ratio of triangles to remove in the simplification process
            texture_size=1024,      # Size of the texture used for the GLB
        )
        glb_base_path = os.path.join(outdir, f"{neutral_prompt.replace(' ', '_').replace('.', '')}_d={int(mesh_density)}_s={slider_scale}")
        glb.export(glb_base_path + '.glb')
        # # Render the outputs
        # video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
        # video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        # video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
        # imageio.mimsave(os.path.join(outdir, f"{neutral_prompt.replace(' ', '_')}_density_{int(mesh_density)}_slider_{slider_scale}.mp4"), video, fps=25)
        
        # Save Gaussians as PLY files
        # outputs['gaussian'][0].save_ply(glb_base_path + '.ply')
        # write_base_mesh(outputs, glb_base_path + '.ply')

def generate_variants_single_attribute_sliders_from_prompts(
        pipeline: TrellisAttributeSliderPipeline, 
        prompt: str, 
        neg_prompt: str, 
        neutral_prompt: str,
        seed_: int, 
        outdir: str,
        dir_idx: int,
        obj_idx: int,
        cfg_strength: float = 3,
    ):
    os.makedirs(outdir, exist_ok=True)
    print("Running pipeline.run_variant()...")
    slider_scales = np.linspace(-5, 5, 21)
    # slider_scales = np.linspace(-20, 20, 9)
    for slider_scale in slider_scales:
        print(f"Running pipeline.run_variant() with slider_scale={slider_scale} for {neutral_prompt}...")
        outputs = pipeline.run_variant_prompt_to_ss_slat(
            prompt=prompt,
            neg_prompt=neg_prompt,
            neutral_prompt=neutral_prompt,
            slider_scale=slider_scale,
            seed=seed_,
            sparse_structure_sampler_params={
                "steps": 12,
                "cfg_strength": cfg_strength,
            },
            slat_sampler_params={
                "steps": 12,
                "cfg_strength": cfg_strength,
            }
        )

        # Render the outputs
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=0.95,          # Ratio of triangles to remove in the simplification process
            texture_size=1024,      # Size of the texture used for the GLB
        )
        # glb_base_path = os.path.join(outdir, f"{neutral_prompt.replace(' ', '_').replace('.', '')}_d={int(mesh_density)}_s={slider_scale}")
        # make the idx and obj_idx padded to 2 digits
        dir_idx = str(dir_idx).zfill(2)
        obj_idx = str(obj_idx).zfill(2)
        glb.export(os.path.join(outdir, f"{dir_idx}_{obj_idx}_s={slider_scale:.2f}_cfg={cfg_strength:.2f}.glb"))


def batch_generate_basic_objects(pipeline: Pipeline, prompts_path, seed: int, outdir: str):
    # read the prompts 
    with open(prompts_path, 'r') as f:
        prompts = f.readlines()
    for prompt in prompts:
        generate_basic_object(pipeline, prompt.strip(), seed, outdir)

def batch_generate_multi_attr_objects(pipeline: Pipeline, prompts_path, seed: int, outdir: str):
    # read the prompts 
    with open(prompts_path, 'r') as f:
        prompts = f.readlines()
    for prompt in prompts:
        if prompt.strip() == 'N/A':
            continue
        generate_basic_object(pipeline, prompt.strip(), seed, outdir)

def batch_generate_variants_single_attribute_style(pipeline: Pipeline, neutral_prompt_path: str, dir_prompts_path: str, seed_: int, outdir: str):
    # read the prompts from neutral_cat_colors_materials.txt
    with open(neutral_prompt_path, 'r') as f:
        neutral_prompts = f.readlines()
    # for style directions, simple
    with open(os.path.join(dir_prompts_path, 'all.txt'), 'r') as f:
        dir_prompts = f.readlines()
    # write the file_obj_map to a tsv file
    with open(os.path.join(outdir, 'obj_idx_to_neutral_prompt.tsv'), 'w') as f:
        for obj_idx, neutral_prompt in enumerate(neutral_prompts):
            f.write(f"{obj_idx}\t{neutral_prompt.strip()}\n")

    for dir_idx, dir_prompt in enumerate(dir_prompts):
        pos, neg = dir_prompt.split('\t') # don't strip. there are empty prompts
        dir_folder_name = pos.replace(' ', '_') + '-' + neg.replace(' ', '_')
        os.makedirs(os.path.join(outdir, dir_folder_name), exist_ok=True)
        print("Generating variants for direction: ", pos, ' -> ', neg if neg.strip() != '' else '[empty]')
        for obj_idx, neutral_prompt in enumerate(neutral_prompts):
            obj_folder_path = os.path.join(outdir, dir_folder_name, str(obj_idx).zfill(2))
            os.makedirs(obj_folder_path, exist_ok=True)
            generate_variants_single_attribute_sliders_from_prompts(pipeline, pos, neg, neutral_prompt, seed_, obj_folder_path, dir_idx, obj_idx, cfg_strength=4)
    

def batch_generate_variants_single_attribute_material(pipeline: Pipeline, mesh_dir: str, neutral_prompt_path: str, dir_prompts_path: str, seed: int, outdir: str):
    # read the prompts from neutral_cat_colors_styles.txt or neutral_cat_only.txt
    with open(neutral_prompt_path, 'r') as f:
        neutral_prompts = f.readlines()
    # for material directions, diff for each furniture type
    dir_prompts = {}
    furniture_files = os.listdir(dir_prompts_path)
    for furniture_file in furniture_files:
        with open(os.path.join(dir_prompts_path, furniture_file), 'r') as f:
            furniture_type = furniture_file.split('.')[0].lower()
            dir_prompts[furniture_type] = f.readlines()
    
    for furniture_type, dir_prompts in dir_prompts.items():
        for dir_prompt in dir_prompts:
            pos, neg = dir_prompt.split('\t') # don't strip. there are empty prompts
            dir_folder_name = furniture_type + '-' + pos.replace(' ', '_') + '-' + neg.replace(' ', '_')
            os.makedirs(os.path.join(outdir, dir_folder_name), exist_ok=True)
            print("Generating variants for direction: ", furniture_type, pos, ' -> ', neg if neg.strip() != '' else '[empty]')
            selected_neutral_prompts = [p for p in neutral_prompts if (furniture_type != 'lighting' and furniture_type in p) or (furniture_type == 'lighting' and ('light' in p or 'lamp' in p))]
            print("Selected neutral prompts: ", selected_neutral_prompts)
            if len(selected_neutral_prompts) == 0:
                print(f">>>>>>> Skipping {furniture_type}: No neutral prompts found. Problematic!!!!")
                continue
            for neutral_prompt in selected_neutral_prompts:
                # there exists files that start with neutral_prompt, skip
                existing_files = os.listdir(os.path.join(outdir, dir_folder_name))
                if any([neutral_prompt.strip().replace(' ', '_').replace('.', '') in f for f in existing_files]):
                    print(f">>>>>>> Skipping {neutral_prompt}: Found existing files")
                    continue
                # read the base mesh
                mesh_path = os.path.join(mesh_dir, neutral_prompt.strip().replace(' ', '_').replace('.', '') + '.ply')
                base_mesh = read_base_mesh(mesh_path)
                if base_mesh is None:
                    print(f">>>>>>> Skipping {neutral_prompt}: Failed to load base mesh")
                    continue
                generate_variants_single_attribute(pipeline, base_mesh, pos, neg, neutral_prompt, seed, os.path.join(outdir, dir_folder_name), get_mesh_density(mesh_path))

def testing_mesh_quality(glb_path):
    base_mesh = read_base_mesh(glb_path)
    # save the base mesh
    o3d.io.write_triangle_mesh('test_base_mesh.ply', base_mesh)

def revert_rotation(glb_path):
    # to resolve the rotation issue between ply and glb output from trellis
    # for glb (y-up), apply rotation to convert to z-up, and get the mesh ready to use for trellis 
    mesh = read_base_mesh(glb_path)
    R = np.array([
        [1,  0, 0],
        [0,  0, -1],
        [0, 1, 0],
    ])
    mesh.rotate(R, center=(0,0,0))
    o3d.io.write_triangle_mesh('test_base_mesh.ply', mesh)

if __name__ == '__main__':
    DO_INFER = True
    if DO_INFER:
        # os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
        os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                                    # 'auto' is faster but will do benchmarking at the beginning.
                                                    # Recommended to set to 'native' if run only once.
    
    seed = 42

    # basic_obj_outdir = 'validation/basic_objects'
    # os.makedirs(basic_obj_outdir, exist_ok=True)
    # pipeline = get_pipeline()
    # batch_generate_basic_objects(pipeline, os.path.join('../3d-interior-val', "neutral_cat_only.txt"), seed, basic_obj_outdir)

    # pre 12/19, and 1/22/2026, generating style sliders for basic objects
    pipeline_single_attr = get_attribute_slider_pipeline()
    slider_outdir = '/data/ru_data/results/trellis_output/validation/style_sliders_basic'
    os.makedirs(slider_outdir, exist_ok=True)
    batch_generate_variants_single_attribute_style(
        pipeline_single_attr, 
        # 'validation/basic_objects', 
        '../3d-interior-val/3d_gen_prompts/neutral_cat_colors_materials_simple.txt', 
        '../3d-interior-val/3d_gen_prompts/style_direction_prompts', 
        seed, 
        slider_outdir
    )

    # 12/29, generating material sliders for basic objects
    # pipeline_single_attr = get_attribute_slider_pipeline()
    # material_outdir = '/data/ru_data/results/trellis_output/validation/material_sliders_basic'
    # os.makedirs(material_outdir, exist_ok=True)
    # batch_generate_variants_single_attribute_material(
    #     pipeline_single_attr, 
    #     'validation/basic_objects', 
    #     '../3d-interior-val/neutral_cat_only.txt', 
    #     '../3d-interior-val/material_direction_prompts', 
    #     seed, 
    #     material_outdir
    # )

    # multi_attr_outdir = 'validation/multi_attr_objects'
    # multi_attr_outdir_styles = os.path.join(multi_attr_outdir, 'styles')
    # multi_attr_outdir_materials = os.path.join(multi_attr_outdir, 'materials')
    
    # os.makedirs(multi_attr_outdir_styles, exist_ok=True)
    # os.makedirs(multi_attr_outdir_materials, exist_ok=True)
    # batch_generate_multi_attr_objects(pipeline, os.path.join('../3d-interior-val', "neutral_cat_colors_styles.txt"), seed, multi_attr_outdir_styles)
    # batch_generate_multi_attr_objects(pipeline, os.path.join('../3d-interior-val', "neutral_cat_colors_materials.txt"), seed, multi_attr_outdir_materials)
    
    # pipeline_single_attr = get_attribute_slider_pipeline()
    # slider_outdir = '/data/ru_data/results/trellis_output/validation/style_sliders'
    # os.makedirs(slider_outdir, exist_ok=True)
    # batch_generate_variants_single_attribute_style(pipeline_single_attr, 'validation/multi_attr_objects/materials', '../3d-interior-val/neutral_cat_colors_materials.txt', '../3d-interior-val/style_direction_prompts', seed, slider_outdir)

    # testing_mesh_quality('validation/multi_attr_objects/A_bed_made_of_composite_board_in_white_color..glb')
    # revert_rotation('validation/multi_attr_objects/A_bed_made_of_composite_board_in_white_color..glb')