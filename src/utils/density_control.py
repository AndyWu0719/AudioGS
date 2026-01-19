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
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import math

# =====================================================
# SPEECH-AWARE CONSTANTS
# =====================================================
# These are based on speech acoustics literature:
# - Typical phoneme duration: 50-200ms
# - Consonant transients: 5-30ms
# - Fundamental frequency range: 80-400Hz

SIGMA_MIN = 0.002         # 2ms - minimum for consonant transients
SIGMA_MAX = 0.050         # 50ms - maximum for vowel formants
Q_FACTOR = 4.0            # Gabor quality factor (standard in audio processing)


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
    ):
        self.grad_threshold = grad_threshold
        self.prune_amplitude_threshold = prune_amplitude_threshold
        self.max_num_atoms = max_num_atoms
        
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
        expected_sigma = (Q_FACTOR / (omega + 1e-8)).clamp(min=SIGMA_MIN, max=SIGMA_MAX)
        
        # Split: sigma is 2x larger than expected (truly over-extended)
        # This is conservative - only split obviously problematic atoms
        split_threshold = expected_sigma * 2.0
        split_mask = high_grad_mask & (sigma > split_threshold)
        
        # Clone: sigma is within expectation, NOT high frequency
        # Relaxed HF limit: 80% Nyquist (was 40%)
        high_freq_mask = omega > 0.8 * nyquist
        clone_mask = high_grad_mask & (sigma <= expected_sigma * 1.5) & (~high_freq_mask)
        
        return split_mask, clone_mask
    
    def get_prune_mask(self, model: nn.Module) -> torch.Tensor:
        """
        Determine which atoms should be pruned (kept=True).
        
        SIMPLE AMPLITUDE-BASED PRUNING:
        ==============================
        - Keep atoms with amplitude >= threshold
        - Gentle penalty for very high frequencies (>90% Nyquist): 1.5x threshold
        
        NO energy-aware pruning - it was destroying consonant transients.
        """
        amplitude = model.amplitude
        omega = model.omega
        nyquist = model.nyquist_freq
        
        base_threshold = self.prune_amplitude_threshold
        
        # Only penalize VERY high frequencies (>90% Nyquist)
        # These are likely to be noise, not speech content
        very_high_freq_mask = omega > 0.9 * nyquist
        threshold = torch.where(
            very_high_freq_mask,
            base_threshold * 1.5,  # Gentle 1.5x penalty (was 2x)
            base_threshold
        )
        
        keep_mask = amplitude >= threshold
        return keep_mask
    
    def densify_and_prune(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        do_split: bool = True,
        do_clone: bool = True,
        do_prune: bool = True,
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
            num_cloned = model.clone_atoms_by_indices(clone_indices)
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
        warmup_iters: int = 100,
        decay_factor: float = 0.995,  # Slower decay
    ):
        super().__init__(
            grad_threshold=grad_threshold,
            prune_amplitude_threshold=prune_amplitude_threshold,
            max_num_atoms=max_num_atoms,
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