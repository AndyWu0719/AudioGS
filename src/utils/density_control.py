"""
Density Control for Audio Gaussian Splatting.
SIMPLIFIED VERSION - Speech-Aware Design.

Design Principles:
==================
1. SIMPLE: Remove unnecessary complex mechanisms
2. SPEECH-AWARE: Parameters match speech acoustics
3. CONSERVATIVE: Prefer under-pruning over over-pruning
4. GRADUAL: Use soft thresholds, avoid hard boundaries

Removed Mechanisms (caused metric degradation):
- Track B Physics Enforcement (destroyed valid harmonics)
- Energy-Aware Pruning (removed consonant transients)
- Late-Stage Freeze (prevented convergence)
- Aggressive HF Clone Block (reduced HF detail)

Issue 4 Fix: Import sigma bounds from shared config for consistency.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import math

# Import shared defaults and parser for initialization config
from src.utils.config import InitConfig


class DensityController:
    """
    Simplified density control for AudioGS.
    
    Split/Clone/Prune logic designed for speech reconstruction:
    - Split: Only gradient-driven, respects Constant-Q envelope
    - Clone: Allowed for most frequencies (up to 80% Nyquist)
    - Prune: Simple amplitude threshold, gentle HF penalty
    """
    
    def __init__(
        self,
        grad_threshold: float = 0.0002,
        prune_amplitude_threshold: float = 0.0005,
        max_num_atoms: int = 20000,
        init_config: Optional[dict] = None,
        clone_sigma_ratio_max: float = 1.25,
        clone_sigma_max: float = 0.03,
        prune_sigma_exponent: float = 0.5,
        prune_sigma_max_boost: float = 4.0,
    ):
        self.grad_threshold = grad_threshold
        self.prune_amplitude_threshold = prune_amplitude_threshold
        self.max_num_atoms = max_num_atoms
        init_cfg = InitConfig.from_dict(init_config)
        self.sigma_min = init_cfg.sigma_min
        self.sigma_max = init_cfg.sigma_max
        self.q_factor = init_cfg.constant_q_cycles
        self.constant_q_use_2pi = init_cfg.constant_q_use_2pi
        self.clone_sigma_ratio_max = clone_sigma_ratio_max
        self.clone_sigma_max = clone_sigma_max
        self.prune_sigma_exponent = prune_sigma_exponent
        self.prune_sigma_max_boost = prune_sigma_max_boost
        
    def get_densification_mask(
        self,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determine which atoms should be split vs cloned.
        
        SIMPLIFIED LOGIC:
        =================
        Split: High gradient AND sigma significantly exceeds Constant-Q expectation
        Clone: High gradient AND small sigma AND frequency < 80% Nyquist
        
        NO Track B physics enforcement - let the optimizer decide.
        """
        avg_grads = model.get_average_gradients()
        high_grad_mask = avg_grads > self.grad_threshold
        
        sigma = model.sigma
        omega = model.omega
        nyquist = model.nyquist_freq
        
        # Constant-Q sigma: sigma = Q / omega
        # This is the expected sigma for a "well-formed" Gabor atom
        denom = omega + 1e-8
        if self.constant_q_use_2pi:
            denom = 2.0 * math.pi * denom
        expected_sigma = (self.q_factor / denom).clamp(
            min=self.sigma_min,
            max=self.sigma_max,
        )
        
        # Split: sigma is 2x larger than expected (truly over-extended)
        # This is conservative - only split obviously problematic atoms
        split_threshold = expected_sigma * 2.0
        split_mask = high_grad_mask & (sigma > split_threshold)
        
        # Clone: sigma is within expectation, NOT extreme high frequency
        high_freq_mask = omega > 0.9 * nyquist
        clone_mask = high_grad_mask & (sigma <= expected_sigma * self.clone_sigma_ratio_max) & (~high_freq_mask)
        if self.clone_sigma_max is not None:
            clone_mask = clone_mask & (sigma <= self.clone_sigma_max)
        
        return split_mask, clone_mask
    
    def get_prune_mask(self, model: nn.Module) -> torch.Tensor:
        """
        Determine which atoms should be pruned (kept=True).
        
        AMPLITUDE-BASED PRUNING with smooth HF penalty:
        - Keep atoms with amplitude >= threshold
        - Apply a continuous HF penalty to avoid sharp banding
        """
        amplitude = model.amplitude
        sigma = model.sigma
        omega = model.omega
        nyquist = model.nyquist_freq
        
        base_threshold = self.prune_amplitude_threshold
        
        # Smooth HF penalty from 0.8 -> 0.99 Nyquist
        omega_ratio = omega / (nyquist + 1e-8)
        hf_boost = torch.clamp((omega_ratio - 0.8) / 0.19, min=0.0, max=1.0)
        sigma_ratio = (sigma / (self.sigma_min + 1e-8)).clamp(min=1.0, max=self.prune_sigma_max_boost)
        sigma_boost = sigma_ratio ** self.prune_sigma_exponent
        adaptive_threshold = base_threshold * (1.0 + 4.0 * hf_boost * hf_boost) * sigma_boost
        
        keep_mask = amplitude >= adaptive_threshold
        return keep_mask
    
    def densify_and_prune(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        do_split: bool = True,
        do_clone: bool = True,
        do_prune: bool = True,
        clone_config: Optional[Dict[str, float]] = None,
    ) -> Dict[str, int]:
        """
        Perform density control.
        
        NO late-stage freeze - let the system naturally stabilize.
        Gradient priority when budget is tight.
        """
        stats = {"split": 0, "cloned": 0, "pruned": 0}
        current_atoms = model.num_atoms
        
        # Get masks
        split_mask, clone_mask = self.get_densification_mask(model)
        avg_grads = model.get_average_gradients()
        
        split_indices = torch.where(split_mask)[0] if do_split else torch.tensor([], device=model.device, dtype=torch.long)
        clone_indices = torch.where(clone_mask)[0] if do_clone else torch.tensor([], device=model.device, dtype=torch.long)
        
        # Budget calculation
        n_split = len(split_indices)
        n_clone = len(clone_indices)
        
        remaining_budget = self.max_num_atoms - current_atoms
        projected_add = n_split * 2 + n_clone
        
        # Budget truncation with gradient priority
        if remaining_budget <= 0:
            split_indices = torch.tensor([], device=model.device, dtype=torch.long)
            clone_indices = torch.tensor([], device=model.device, dtype=torch.long)
        elif projected_add > remaining_budget:
            # Prioritize by gradient (higher gradient = more important)
            if n_split * 2 > remaining_budget:
                allowed_splits = remaining_budget // 2
                if allowed_splits > 0 and len(split_indices) > 0:
                    grads = avg_grads[split_indices]
                    sorted_idx = torch.argsort(grads, descending=True)
                    split_indices = split_indices[sorted_idx[:allowed_splits]]
                else:
                    split_indices = torch.tensor([], device=model.device, dtype=torch.long)
                clone_indices = torch.tensor([], device=model.device, dtype=torch.long)
            else:
                remaining = remaining_budget - n_split * 2
                if remaining > 0 and len(clone_indices) > 0:
                    grads = avg_grads[clone_indices]
                    sorted_idx = torch.argsort(grads, descending=True)
                    clone_indices = clone_indices[sorted_idx[:remaining]]
                else:
                    clone_indices = torch.tensor([], device=model.device, dtype=torch.long)
        
        # Execute operations
        old_params = self._get_param_dict(model)
        any_changes = False
        
        if len(split_indices) > 0:
            model.split_atoms_by_indices(split_indices)
            stats["split"] = len(split_indices)
            any_changes = True

        if len(clone_indices) > 0:
            num_cloned = model.clone_atoms_by_indices(clone_indices, clone_config=clone_config)
            stats["cloned"] = num_cloned
            if num_cloned > 0:
                any_changes = True
        
        keep_indices = None
        if do_prune:
            prune_mask = self.get_prune_mask(model)
            num_to_prune = (~prune_mask).sum().item()
            if num_to_prune > 0:
                keep_indices = model.remove_atoms(prune_mask)
                stats["pruned"] = num_to_prune
                any_changes = True

        if any_changes:
            self._update_optimizer_state(model, optimizer, old_params, keep_indices)
            model.reset_gradient_accumulators()
        
        return stats

    def add_atoms_from_residual(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        residual: torch.Tensor,
        num_atoms: int,
        init_config: Optional[dict] = None,
        n_fft: int = 2048,
        hop_length: int = 512,
        amp_scale: float = 0.3,
        peaks_per_frame: int = 4,
        min_peak_ratio: float = 0.25,
        time_jitter_ratio: float = 0.5,
        freq_jitter_bins: float = 0.5,
        selection_strategy: str = "stft_peak",
        sigma_multiplier: float = 5.0,
        mp_amp_max: Optional[float] = None,
        mp_normalize: bool = True,
        mp_score_min: float = 0.0,
    ) -> int:
        """Add residual-guided atoms and update optimizer state."""
        if num_atoms <= 0:
            return 0
        old_params = self._get_param_dict(model)
        num_added = model.add_atoms_from_residual(
            residual,
            num_atoms,
            init_config=init_config,
            n_fft=n_fft,
            hop_length=hop_length,
            amp_scale=amp_scale,
            peaks_per_frame=peaks_per_frame,
            min_peak_ratio=min_peak_ratio,
            time_jitter_ratio=time_jitter_ratio,
            freq_jitter_bins=freq_jitter_bins,
            selection_strategy=selection_strategy,
            sigma_multiplier=sigma_multiplier,
            mp_amp_max=mp_amp_max,
            mp_normalize=mp_normalize,
            mp_score_min=mp_score_min,
        )
        if num_added > 0:
            self._update_optimizer_state(model, optimizer, old_params, keep_indices=None)
            model.reset_gradient_accumulators()
        return num_added
    
    def _get_param_dict(self, model: nn.Module) -> Dict[str, nn.Parameter]:
        return {
            "amplitude_logit": model._amplitude_logit,
            "tau": model._tau,
            "omega_logit": model._omega_logit,
            "sigma_logit": model._sigma_logit,
            "phi_vector": model._phi_vector,
            "gamma": model._gamma,
        }
    
    def _update_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        old_params: Dict[str, nn.Parameter],
        keep_indices: Optional[torch.Tensor] = None,
    ):
        """Update optimizer state with correct index mapping."""
        new_params = self._get_param_dict(model)
        
        # Capture LRs
        old_param_id_to_lr = {}
        for group in optimizer.param_groups:
            for p in group['params']:
                old_param_id_to_lr[id(p)] = group['lr']
        
        name_to_lr = {}
        for name, old_tensor in old_params.items():
            name_to_lr[name] = old_param_id_to_lr.get(id(old_tensor), 0.001)
        
        # Backup states
        old_states = {}
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p in optimizer.state:
                    old_states[id(p)] = optimizer.state[p]
        
        old_param_states = {}
        for name, old_param in old_params.items():
            if id(old_param) in old_states:
                old_param_states[name] = old_states[id(old_param)]
        
        # Rebuild
        optimizer.param_groups = []
        optimizer.state = {}
        
        for name, param in new_params.items():
            lr = name_to_lr.get(name, 0.001)
            optimizer.add_param_group({"params": [param], "lr": lr})
            
            if name in old_param_states:
                old_state = old_param_states[name]
                if "exp_avg" in old_state:
                    old_exp_avg = old_state["exp_avg"]
                    old_exp_avg_sq = old_state["exp_avg_sq"]
                    is_2d = old_exp_avg.dim() == 2
                    
                    if keep_indices is not None:
                        old_len = old_exp_avg.shape[0]
                        valid_mask = keep_indices < old_len
                        valid_old_indices = keep_indices[valid_mask]
                        
                        new_exp_avg = torch.zeros_like(param)
                        new_exp_avg_sq = torch.zeros_like(param)
                        
                        n_valid = len(valid_old_indices)
                        if n_valid > 0:
                            if is_2d:
                                new_exp_avg[:n_valid, :] = old_exp_avg[valid_old_indices, :]
                                new_exp_avg_sq[:n_valid, :] = old_exp_avg_sq[valid_old_indices, :]
                            else:
                                new_exp_avg[:n_valid] = old_exp_avg[valid_old_indices]
                                new_exp_avg_sq[:n_valid] = old_exp_avg_sq[valid_old_indices]
                    else:
                        n_copy = min(old_exp_avg.shape[0], param.shape[0])
                        new_exp_avg = torch.zeros_like(param)
                        new_exp_avg_sq = torch.zeros_like(param)
                        if is_2d:
                            new_exp_avg[:n_copy, :] = old_exp_avg[:n_copy, :]
                            new_exp_avg_sq[:n_copy, :] = old_exp_avg_sq[:n_copy, :]
                        else:
                            new_exp_avg[:n_copy] = old_exp_avg[:n_copy]
                            new_exp_avg_sq[:n_copy] = old_exp_avg_sq[:n_copy]
                    
                    optimizer.state[param] = {
                        "step": old_state.get("step", torch.tensor(0)),
                        "exp_avg": new_exp_avg,
                        "exp_avg_sq": new_exp_avg_sq,
                    }


class AdaptiveDensityController(DensityController):
    """Adaptive controller with gradual threshold decay."""
    
    def __init__(
        self,
        grad_threshold: float = 0.0002,
        prune_amplitude_threshold: float = 0.0005,
        max_num_atoms: int = 20000,
        init_config: Optional[dict] = None,
        warmup_iters: int = 100,
        decay_factor: float = 0.995,  # Slower decay
        clone_sigma_ratio_max: float = 1.25,
        clone_sigma_max: float = 0.03,
        prune_sigma_exponent: float = 0.5,
        prune_sigma_max_boost: float = 4.0,
    ):
        super().__init__(
            grad_threshold=grad_threshold,
            prune_amplitude_threshold=prune_amplitude_threshold,
            max_num_atoms=max_num_atoms,
            init_config=init_config,
            clone_sigma_ratio_max=clone_sigma_ratio_max,
            clone_sigma_max=clone_sigma_max,
            prune_sigma_exponent=prune_sigma_exponent,
            prune_sigma_max_boost=prune_sigma_max_boost,
        )
        self.warmup_iters = warmup_iters
        self.decay_factor = decay_factor
        self.iteration = 0
        self.initial_grad_threshold = grad_threshold
        
    def update_thresholds(self, loss: float):
        self.iteration += 1
        if self.iteration > self.warmup_iters:
            decayed = self.initial_grad_threshold * (
                self.decay_factor ** (self.iteration - self.warmup_iters)
            )
            # Floor at 1e-5 for stability
            self.grad_threshold = max(decayed, 1e-5)


def rebuild_optimizer_from_model(
    model: nn.Module,
    optimizer_class: type = torch.optim.Adam,
    lr_config: Optional[Dict[str, float]] = None,
    **optimizer_kwargs,
) -> torch.optim.Optimizer:
    if lr_config is None:
        lr_config = {}
    param_groups = model.get_optimizer_param_groups(lr_config)
    return optimizer_class(param_groups, **optimizer_kwargs)
