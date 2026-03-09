from typing import *
import torch
import torch.nn as nn
from contextlib import contextmanager
import numpy as np
from transformers import CLIPTextModel, AutoTokenizer
import open3d as o3d
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp

from .samplers.flow_euler import FlowEulerSamplerAttributeSlider

class TrellisAttributeSliderPipeline(Pipeline):
    """
    Pipeline for inferring Trellis text-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        text_cond_model (str): The name of the text conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        text_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self._init_text_cond_model(text_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisAttributeSliderPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisAttributeSliderPipeline, TrellisAttributeSliderPipeline).from_pretrained(path)
        new_pipeline = TrellisAttributeSliderPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler = FlowEulerSamplerAttributeSlider(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        print('sparse structure sampler name: ', args['sparse_structure_sampler']['name'])
        print('sparse structure sampler parameters: ', new_pipeline.sparse_structure_sampler_params)

        new_pipeline.slat_sampler = FlowEulerSamplerAttributeSlider(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']
        print('slat sampler parameters: ', new_pipeline.slat_sampler_params)

        new_pipeline.slat_normalization = args['slat_normalization']
        print('slat normalization: ', new_pipeline.slat_normalization)

        new_pipeline._init_text_cond_model(args['text_cond_model'])

        return new_pipeline
    
    def _init_text_cond_model(self, name: str):
        """
        Initialize the text conditioning model.
        """
        # load model
        model = CLIPTextModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and all(isinstance(t, str) for t in text), "text must be a list of strings"
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].cuda()
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state
        
        return embeddings
    

    def get_cond(self, prompt: str, neg_prompt: str = None) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            prompt (str): The text prompt.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_text([prompt]) # make sure encode_text is called with a list of a single string
        # neg_cond = self.text_cond_model['null_cond']
        neg_cond = self.text_cond_model['null_cond'] if neg_prompt is None else self.encode_text([neg_prompt]) # make sure encode_text is called with a list of a single string
        return {
            'cond': cond,
            'neg_cond': neg_cond,
            'neutral_cond': None,
            'other_conds_mix_pos': None,
            'other_conds_mix_neg': None,
        }
    
    def mix_cond(self, neutral_prompt: str, positive_prompt: str, neg_prompt: str = None, neutral_null: bool = False) -> dict:
        """
        Mix the conditioning information for the model.

        Args:
            neutral_prompt (str): The neutral prompt. a single string
            positive_prompt (str): The positive prompt. a single string
            neg_prompt (str): The negative prompt. a single string
            other_prompts (List[str]): The other attributes that are irrelevant to the positive prompt. a list of strings, each string is a single prompt

        Returns:
            dict: The conditioning information
        """
        # mixing pos cond with neutral cond
        pos_cond = self.encode_text([neutral_prompt + ", " + positive_prompt])
        neg_cond = self.encode_text([neutral_prompt + ", " + neg_prompt])
        neutral_cond = self.encode_text([neutral_prompt]) if not neutral_null else self.text_cond_model['null_cond']
        empty_cond = self.text_cond_model['null_cond']
        return {
            'cond': pos_cond,
            'neg_cond': neg_cond,
            'neutral_cond': neutral_cond,
            'empty_cond': empty_cond,
            # 'other_conds_mix_pos': None,
            # 'other_conds_mix_neg': None,
        }
        # other_conds_mix_pos = [con  for con in other_prompts] if other_prompts is not None else []
        # other_conds_mix_pos = [self.encode_text([con]) for con in other_conds_mix_pos] # make sure encode_text is called with a list of a single string
        # neu_cond = self.encode_text([neutral_prompt]) # neutral_prompt
        # other_conds_mix_neg = [con + ", " + neg_prompt for con in other_prompts] if other_prompts is not None else []
        # other_conds_mix_neg = [self.encode_text([con]) for con in other_conds_mix_neg] # make sure encode_text is called with a list of a single string

        # return {
        #     'cond': None,
        #     'neg_cond': None,
        #     # added for attribute slider
        #     'neutral_cond': neu_cond,
        #     'other_conds_mix_pos': other_conds_mix_pos,
        #     'other_conds_mix_neg': other_conds_mix_neg,
        # }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        slider_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            slider_scale=slider_scale,
        ).samples
        print(f"z_s shape: {z_s.shape}")
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords
    
    def sample_sparse_structure_from_ss_latent(
        self,
        cond: dict,
        ss_latent: torch.Tensor,
        num_samples: int = 1,
        sampler_params: dict = {},
        slider_scale: float = 1.0,
        v_steps_inv: Optional[List[Any]] = None, # t: 1 -> 0
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Any]]:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        print('>>> Sampling sparse structure from SS Latent...')
        print(f"ss_latent shape: {ss_latent.shape}")
        print(f"ss_latent: {type(ss_latent)}")
        output = self.sparse_structure_sampler.sample_from_latent(
            flow_model,
            ss_latent,
            **cond,
            **sampler_params,
            verbose=True,
            slider_scale=slider_scale,
            v_steps_inv=v_steps_inv,
        )
        z_s = output.samples
        v_steps = output.v_steps
        print(f"z_s shape: {z_s.shape}")
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords, z_s, v_steps

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            # print('here in decode_slat(), decoding mesh, using model: ', self.models['slat_decoder_mesh'])
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            # print('here in decode_slat(), decoding gaussian, using model: ', self.models['slat_decoder_gs'])
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            # print('here in decode_slat(), decoding radiance field, using model: ', self.models['slat_decoder_rf'])
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat_from_slat(
        self,
        cond: dict,
        slat: sp.SparseTensor,
        sampler_params: dict = {},
        slider_scale: float = 1.0,
        noise_strength: float = 0.7,
        v_steps_inv: Optional[List[Any]] = None, # t: 1 -> 0
    ) -> Tuple[sp.SparseTensor, List[Any]]:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']

        # Normalize slat
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        print(f">>> mean: {mean}, std: {std}")
        normed_feats = (slat.feats - mean) / std # let's try normalize slat first

        init_slat = sp.SparseTensor(
            feats=normed_feats,
            coords=slat.coords,
        )
        
        # init_slat = sp.SparseTensor(
        #     feats=normed_feats,
        #     coords=slat.coords,
        # )
        
        # Add noise for SDEdit (Slat-to-Slat)
        # noise = sp.SparseTensor(
        #     feats=torch.randn_like(init_slat.feats),
        #     coords=init_slat.coords,
        # )
        # start_feats = (1 - noise_strength) * init_slat.feats + noise_strength * noise.feats
        # start_slat = sp.SparseTensor(feats=start_feats, coords=init_slat.coords)
        
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        output = self.slat_sampler.sample_from_latent(
            flow_model,
            init_slat,
            **cond,
            **sampler_params,
            verbose=True,
            slider_scale=slider_scale,
            noise_strength=noise_strength,
            v_steps_inv=v_steps_inv,
        )
        slat = output.samples
        v_steps = output.v_steps
        # Denormalize
        slat = slat * std + mean
        
        return slat, v_steps
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
        slider_scale: float = 1.0,
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            slider_scale=slider_scale,
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    @torch.no_grad()
    def run(
        self,
        prompt: str,    # positive prompt
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        # neg_prompt: str = None,
        # neutral_prompt: str = None,
        # other_prompts: List[str] = None,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond(prompt) # only a single prompt is given, neutral cond
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        # print('here in run(), coords shape:', coords.shape)
        # print('first serveral coords:', coords[:5])
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
    
    def voxelize(self, mesh: o3d.geometry.TriangleMesh) -> torch.Tensor:
        """
        Voxelize a mesh.

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to voxelize.
            sha256 (str): The SHA256 hash of the mesh.
            output_dir (str): The output directory.
        """
        vertices = np.asarray(mesh.vertices)
        # print('here in voxelize(), first vertices shape:', vertices.shape)
        # print('first serveral vertices:', vertices[:5])
        vertices = np.asarray(mesh.vertices)
        if len(vertices) == 0:
            print('here in voxelize(), vertices is empty')
            return torch.zeros((0, 3), dtype=torch.int, device='cuda')
        # print('here in voxelize(), first vertices shape:', vertices.shape)
        # print('first serveral vertices:', vertices[:5])
        aabb = np.stack([vertices.min(0), vertices.max(0)])
        center = (aabb[0] + aabb[1]) / 2
        scale = (aabb[1] - aabb[0]).max()
        vertices = (vertices - center) / scale
        vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        # print('here in voxelize(), after clip(), vertices shape:', vertices.shape)
        # print('first serveral vertices:', vertices[:5])
        # voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
        # vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        # return torch.tensor(vertices).int().cuda()

        # Fixed: use create_from_triangle_mesh instead of create_from_triangle_mesh_within_bounds
        # to avoid segmentation fault with Open3D 0.17.0 on non-watertight meshes
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=1/64)
        # print('here in voxelize(), after create_from_triangle_mesh(), voxel_grid dimension:', voxel_grid.dimension())
        voxel_indices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        # print('here in voxelize(), after get_voxels(), voxel_indices shape:', voxel_indices.shape)
        # print('first serveral voxels indices:', voxel_indices[:5])
        # Filter voxels to be within the expected bounds [0, 63] in each dimension
        if voxel_indices.shape[0] == 0:
            print('here in voxelize(), voxel_indices is empty')
            return torch.zeros((0, 3), dtype=torch.int, device='cuda')
            
        valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < 64), axis=1)
        voxel_indices = voxel_indices[valid_mask]
        # print('here in voxelize(), after valid_mask(), voxel_indices shape:', voxel_indices.shape)
        return torch.tensor(voxel_indices).int().cuda()

    @torch.no_grad()
    def run_variant_from_slat(
        self,
        slat: sp.SparseTensor,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        neg_prompt: str = None,
        neutral_prompt: str = None,
        v_steps_inv: Optional[List[Any]] = None, # t: 1 -> 0
        slider_scale: float = 1.0,
        noise_strength: float = 0.3,
        other_prompts: List[str] = None,
    ) -> dict:
        """
        Run the pipeline for making variants of an asset.
        
        Args:
            mesh (o3d.geometry.TriangleMesh): The base mesh.
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            neg_prompt (str): The negative prompt.
            neutral_prompt (str): The neutral prompt.
            other_prompts (List[str]): The other attributes that are irrelevant to the positive prompt. a list of strings, each string is a single prompt
        """
        cond = self.mix_cond(
            neutral_prompt=neutral_prompt, 
            positive_prompt=prompt, 
            neg_prompt=neg_prompt,
        )
        
        torch.manual_seed(seed)
        slat, v_steps = self.sample_slat_from_slat(cond, slat, slat_sampler_params, slider_scale, noise_strength=noise_strength, v_steps_inv=v_steps_inv)
        return self.decode_slat(slat, formats), slat, v_steps

    @torch.no_grad()
    def run_variant_from_ss_slat(
        self,
        ss_latent_in: torch.Tensor,
        slat_in: sp.SparseTensor,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        neg_prompt: str = None,
        neutral_prompt: str = None,
        v_steps_ss_inv: Optional[List[Any]] = None, # t: 1 -> 0
        v_steps_slat_inv: Optional[List[Any]] = None, # t: 1 -> 0
        slider_scale: float = 1.0,
        noise_strength: float = 0.3,
        other_prompts: List[str] = None,
    ) -> dict:
        """
        Run the pipeline for making variants of an asset.
        
        Args:
            ss_latent_in: torch.Tensor: The sparse structure latent.
            slat_in: sp.SparseTensor: The structured latent.
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            neg_prompt (str): The negative prompt.
            neutral_prompt (str): The neutral prompt.
            other_prompts (List[str]): The other attributes that are irrelevant to the positive prompt. a list of strings, each string is a single prompt
        """
        cond = self.mix_cond(
            neutral_prompt=neutral_prompt, 
            positive_prompt=prompt, 
            neg_prompt=neg_prompt,
        )
        torch.manual_seed(seed)

        coords, z_s, v_steps_ss = self.sample_sparse_structure_from_ss_latent(cond, ss_latent_in, num_samples, slider_scale=slider_scale, v_steps_inv=v_steps_ss_inv)
        print(f"z_s shape: {z_s.shape}")
        print(f"coords shape: {coords.shape}")
        print(f"v_steps (to ss) shape: {len(v_steps_ss)}, {v_steps_ss[0].shape}")

        # here merge coords and slat to get the new input slat
        print(f"slat_in feats shape: {slat_in.feats.shape}")
        # do something bold here to merge coords and slat when the new coords dim 0 is not the same as the slat_in feats dim 0
        # if slat_in.feats.shape[0] > coords.shape[0], then we need to truncate the slat_in feats to the same length as coords.shape[0]
        # if slat_in.feats.shape[0] < coords.shape[0], then we need to add zeros to the slat_in feats to the same length as coords.shape[0]
        new_feats = slat_in.feats
        if slat_in.feats.shape[0] > coords.shape[0]:
            print(f"truncating slat_in feats from {slat_in.feats.shape[0]} to {coords.shape[0]}")
            new_feats = new_feats[:coords.shape[0]].cuda()
        elif slat_in.feats.shape[0] < coords.shape[0]:
            print(f"adding zeros to slat_in feats from {slat_in.feats.shape[0]} to {coords.shape[0]}")
            new_feats = torch.cat([new_feats, torch.zeros(coords.shape[0] - slat_in.feats.shape[0], slat_in.feats.shape[1]).cuda()], dim=0)
        new_slat_in = sp.SparseTensor(
            feats=new_feats,
            coords=coords,
        )
        new_slat_out, v_steps_slat = self.sample_slat_from_slat(cond, new_slat_in, slat_sampler_params, slider_scale, noise_strength=noise_strength, v_steps_inv=v_steps_slat_inv)
        return self.decode_slat(new_slat_out, formats), z_s, v_steps_ss, new_slat_out, v_steps_slat
    
    @torch.no_grad()
    def run_variant_from_ss(
        self,
        ss_latent_in: torch.Tensor,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        neg_prompt: str = None,
        neutral_prompt: str = None,
        v_steps_ss_inv: Optional[List[Any]] = None, # t: 1 -> 0
        v_steps_slat_inv: Optional[List[Any]] = None, # t: 1 -> 0
        slider_scale: float = 1.0,
        noise_strength: float = 0.3,
        other_prompts: List[str] = None,
    ) -> dict:
        """
        Run the pipeline for making variants of an asset.
        
        Args:
            ss_latent_in: torch.Tensor: The sparse structure latent.
            slat_in: sp.SparseTensor: The structured latent.
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            neg_prompt (str): The negative prompt.
            neutral_prompt (str): The neutral prompt.
            other_prompts (List[str]): The other attributes that are irrelevant to the positive prompt. a list of strings, each string is a single prompt
        """
        cond = self.mix_cond(
            neutral_prompt=neutral_prompt, 
            positive_prompt=prompt, 
            neg_prompt=neg_prompt,
        )
        torch.manual_seed(seed)

        coords, z_s, v_steps_ss = self.sample_sparse_structure_from_ss_latent(cond, ss_latent_in, num_samples, slider_scale=slider_scale, v_steps_inv=v_steps_ss_inv)
        print(f"z_s shape: {z_s.shape}")
        print(f"coords shape: {coords.shape}")
        print(f"v_steps (to ss) shape: {len(v_steps_ss)}, {v_steps_ss[0].shape}")

        slat_out = self.sample_slat(cond, coords, slat_sampler_params, slider_scale)
        return self.decode_slat(slat_out, formats), z_s, v_steps_ss


    @torch.no_grad()
    def run_variant(
        self,
        mesh: o3d.geometry.TriangleMesh,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        neg_prompt: str = None,
        neutral_prompt: str = None,
        slider_scale: float = 1.0,
        other_prompts: List[str] = None,
    ) -> dict:
        """
        Run the pipeline for making variants of an asset.

        Args:
            mesh (o3d.geometry.TriangleMesh): The base mesh.
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            neg_prompt (str): The negative prompt.
            neutral_prompt (str): The neutral prompt.
            other_prompts (List[str]): The other attributes that are irrelevant to the positive prompt. a list of strings, each string is a single prompt
        """
        cond = self.mix_cond(neutral_prompt=neutral_prompt, positive_prompt=prompt, neg_prompt=neg_prompt)
        coords = self.voxelize(mesh)
        # print('here in run_variant() after voxelize(), coords shape:', coords.shape)
        # print('first serveral coords:', coords[:5])
        coords = torch.cat([
            torch.arange(num_samples).repeat_interleave(coords.shape[0], 0)[:, None].int().cuda(),
            coords.repeat(num_samples, 1)
        ], 1)
        # print('here in run_variant() after cat(), coords shape:', coords.shape)
        # print('first serveral coords:', coords[:5])
        torch.manual_seed(seed)
        slat = self.sample_slat(cond, coords, slat_sampler_params, slider_scale)
        return self.decode_slat(slat, formats)
    
    @torch.no_grad()
    def run_variant_prompt_to_ss_slat(
        self,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
        sparse_structure_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        neg_prompt: str = None,
        neutral_prompt: str = None,
        slider_scale: float = 1.0,
        other_prompts: List[str] = None,
    ) -> dict:
        """
        Run the pipeline for making variants of an asset.

        Args:
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            neg_prompt (str): The negative prompt.
            neutral_prompt (str): The neutral prompt.
            other_prompts (List[str]): The other attributes that are irrelevant to the positive prompt. a list of strings, each string is a single prompt
        """

        # Sample sparse structure first
        cond = self.mix_cond(neutral_prompt=neutral_prompt, positive_prompt=prompt, neg_prompt=neg_prompt)
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params, slider_scale=slider_scale)
        slat = self.sample_slat(cond, coords, slat_sampler_params, slider_scale)
        return self.decode_slat(slat, formats)
    
    @contextmanager
    def inject_sampler_multi_text(
        self,
        sampler_name: str,
    ):
        """
        Inject a sampler with multiple texts as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        from .samplers import FlowEulerSamplerAttributeSlider
        def _new_inference_model(self, model, x_t, t, cond, **kwargs):
            preds = []
            for i in range(len(cond)):
                preds.append(FlowEulerSamplerAttributeSlider._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
            pred = sum(preds) / len(preds)
            return pred
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')


    def get_cond_gen_edit(self, neutral_prompt: str, positive_prompt: List[str], neg_prompt: List[str], neutral_null: bool = False) -> dict:
        """
        Mix the conditioning information for the model.

        Args:
            neutral_prompt (str): The neutral prompt. a single string
            positive_prompt (List[str]): The positive prompt. a list of strings, each string is a single prompt
            neg_prompt (List[str]): The negative prompt. a list of strings, each string is a single prompt

        Returns:
            dict: The conditioning information
        """
        # get conditions

        pos_cond = self.encode_text(positive_prompt) #.mean(dim=0, keepdim=True) 
        print(f"pos_cond shape: {pos_cond.shape}")
        neg_cond = self.encode_text(neg_prompt) #.mean(dim=0, keepdim=True)
        print(f"neg_cond shape: {neg_cond.shape}")
        neutral_cond = self.encode_text([neutral_prompt]) if not neutral_null else self.text_cond_model['null_cond']
        print(f"neutral_cond shape: {neutral_cond.shape}")
        empty_cond = self.text_cond_model['null_cond']
        print(f"empty_cond shape: {empty_cond.shape}")
        return {
            'cond': pos_cond,
            'neg_cond': neg_cond,
            'neutral_cond': neutral_cond,
            'empty_cond': empty_cond,
        }

    @torch.no_grad()
    def run_variant_condition_to_ss_slat(
        self,
        cond: dict,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
        sparse_structure_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        slider_scale: float = 1.0,
    ) -> dict:
        """
        Run the pipeline for making variants of an asset.

        Args:
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            neg_prompt (str): The negative prompt.
            neutral_prompt (str): The neutral prompt.
            other_prompts (List[str]): The other attributes that are irrelevant to the positive prompt. a list of strings, each string is a single prompt
        """

        # Sample sparse structure first
        torch.manual_seed(seed)
        with self.inject_sampler_multi_text('sparse_structure_sampler'):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params, slider_scale=slider_scale)
        with self.inject_sampler_multi_text('slat_sampler'):
            slat = self.sample_slat(cond, coords, slat_sampler_params, slider_scale)
        return self.decode_slat(slat, formats)
    
    @torch.no_grad()
    def run_w_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Run the pipeline with a given SLAT.
        """
        # std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        # mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        # slat = slat * std + mean
        return self.decode_slat(slat, formats)  