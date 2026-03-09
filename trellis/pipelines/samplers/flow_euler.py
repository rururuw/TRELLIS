from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
from ...modules import sparse as sp


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        # print('t at _inference_model:', t)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        # print('t_pairs:', t_pairs)
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerSamplerAttributeSlider(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        # print('t at _inference_model:', t)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t, cond, **kwargs)


    @torch.no_grad()
    def sample_inverse(
        self,
        model,
        x_0,
        cond: Optional[Any] = None,
        empty_cond=None,
        v_steps_inv: Optional[List[Any]] = None,
        steps: int = 50,
        rescale_t: float = 1.0
    ):
        """
        Invert x_0 to noise using Euler method.
        
        Args:
            model: The model to sample from.
            x_0: The initial clean sample.
            cond: conditional information (usually the source prompt).
            empty_cond: the empty conditional information.
            v_steps_inv: The list of flow steps (t: 1 -> 0), need to reverse here.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the noisy sample (x_1).
            - 'pred_x_t': a list of prediction of x_t.
        """
        sample = x_0
        # For inversion, we go from 0 to 1
        t_seq = np.linspace(0, 1, steps + 1)
        # Apply rescaling if needed
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        if v_steps_inv is not None:
            v_steps_inv = v_steps_inv[::-1]
            assert len(v_steps_inv) == len(t_pairs)
        
        ret = edict({'steps': t_pairs, "samples": [], "v_steps": []})
        for i, (t, t_next) in enumerate(tqdm(t_pairs, desc="Inverting", disable=False)):
            # Predict flow v at current state x_t and time t.
            # We use the source conditioning (cond) to guide the inversion.
            if v_steps_inv is not None:
                v = v_steps_inv[i]
            else:
                v = self._inference_model(model, sample, t, cond)
            
            # Update sample: x_{t+1} = x_t + dt * v
            dt = t_next - t
            sample = sample + dt * v
            
            ret.samples.append(sample)
            ret.v_steps.append(v)
        return ret

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        neutral_cond=None, 
        neg_cond=None,
        empty_cond=None,
        cfg_strength: float = 3.0,
        slider_scale: float = 1.0,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            neutral_cond: The neutral conditional information.
            neg_cond: The negative conditional information.
            empty_cond: The empty conditional information.
            cfg_strength: The strength of classifier-free guidance.
            slider_scale: The scale of the slider.
            step_from: The step to start from. A ratio [0, 1] of the total steps.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        guidance_denoise = cfg_strength
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        # print('t_pairs:', t_pairs)
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            v_neutral = self._inference_model(model, sample, t, neutral_cond)
            v_uncond = self._inference_model(model, sample, t, empty_cond)
            v_guided = v_uncond + (v_neutral - v_uncond) * guidance_denoise

            if slider_scale != 0:
                v_pos = self._inference_model(model, sample, t, cond)
                v_neg = self._inference_model(model, sample, t, neg_cond)
                v_diff = v_pos - v_neg
                v_target = v_guided + v_diff * slider_scale
            else:
                v_target = v_guided

            pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=sample, t=t, v=v_target)
            
            sample = sample - (t - t_prev) * v_target
            
            ret.pred_x_t.append(sample)
            ret.pred_x_0.append(pred_x_0)

        ret.samples = sample
        return ret

    @torch.no_grad()
    def sample_from_latent(
        self,
        model,
        latent,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        neutral_cond=None, 
        neg_cond=None,
        empty_cond=None,
        v_steps_inv: Optional[List[Any]] = None, # t: 1 -> 0
        cfg_strength: float = 3.0,
        slider_scale: float = 1.0,
        noise_strength: float = 0.7,
        **kwargs
    ):
        original_coords = latent.coords.clone() if isinstance(latent, sp.SparseTensor) else latent.clone()
        # first do a inversion to get the noise
        print(f">>> Inverting...")
        inversion_ret = self.sample_inverse(model, latent, neutral_cond, empty_cond, v_steps_inv, steps, rescale_t)
        sample_step = inversion_ret.samples[-1] # go from pure noise to the initial slat
        #check if the coords are the same if the sample is a sparse tensor
        if isinstance(sample_step, sp.SparseTensor):
            if (sample_step.coords != original_coords).any():
                print("!!!!!! WARNING: Coords changed during inversion!")
            sample_step = sp.SparseTensor(
                feats=sample_step.feats,
                coords=original_coords,
            )
        else:
            if (sample_step != original_coords).any():
                print("!!!!!! WARNING: Coords changed during inversion!")
            pass # do nothing for regular tensors

        # for forward sampling, we need to reverse the t_pairs and v_steps
        # v, t: 0 -> 1, but here we want to go from 1 to 0, so we reverse the list
        t_pairs = [(t, t_prev) for t_prev, t in inversion_ret.steps[::-1]]
        v_steps = inversion_ret.v_steps[::-1]

        print(f">>> Inversion done. t_pairs: {t_pairs}")
        
        # print('t_pairs:', t_pairs)
        guidance_denoise = 3.0
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": [], "v_steps": []})
        for i, (t, t_prev) in enumerate(tqdm(t_pairs, desc="Sampling", disable=not verbose)):
            
            v_pos = self._inference_model(model, sample_step, t, cond)
            v_neg = self._inference_model(model, sample_step, t, neg_cond)
            # v_uncond = self._inference_model(model, sample_step, t, empty_cond)

            # Use v_steps (inversion trace) directly as the reconstruction backbone.
            # v_steps[i] approximately equals v(sample, t, neutral_cond) from the inversion path.
            # We treat it as the "neutral" flow that preserves identity.
            # We add the edit direction (v_pos - v_neg) scaled by slider_scale.
            
            # v_guided = v_uncond + (v_steps[i] - v_uncond) * guidance_denoise # This might be too strong/rigid
            
            v_reconstruct = v_steps[i]
            v_edit = v_pos - v_neg
            
            v_target = v_reconstruct + v_edit * slider_scale

            if verbose:
                if isinstance(sample_step, sp.SparseTensor):
                    # only look at feats 
                    print(f"t={t:.3f} | |v_reconstruct|={torch.norm(v_reconstruct.feats):.4f} | |v_edit|={torch.norm(v_edit.feats):.4f} | ratio={torch.norm(v_edit.feats)/torch.norm(v_reconstruct.feats):.4f}")
                    print(f"sample_step.feats.shape={sample_step.feats.shape} | sample_step.feats_norm={torch.norm(sample_step.feats):.4f} | v_target.feats_norm={torch.norm(v_target.feats):.4f}")
                else:
                    print(f"t={t:.3f} | |v_reconstruct|={torch.norm(v_reconstruct):.4f} | |v_edit|={torch.norm(v_edit):.4f} | ratio={torch.norm(v_edit)/torch.norm(v_reconstruct):.4f}")
                    print(f"sample_step.shape={sample_step.shape} | sample_step_norm={torch.norm(sample_step):.4f} | v_target_norm={torch.norm(v_target):.4f}")
                 
            pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=sample_step, t=t, v=v_target)
            
            sample_step = sample_step - (t - t_prev) * v_target
            
            ret.v_steps.append(v_target)
            ret.pred_x_t.append(sample_step)
            ret.pred_x_0.append(pred_x_0)

        ret.samples = sample_step
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
