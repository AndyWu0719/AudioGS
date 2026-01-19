"""
Density Control for Audio Gaussian Splatting.
Major Refactor: Fixed Optimizer Index Mapping, High-Freq Clone Prevention.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import math


class DensityController:
    """
    Manages density control operations for AudioGS model.
    
    REFACTOR: Critical fix for optimizer state mapping during pruning.
    """
    
    def __init__(
        self,
        grad_threshold: float = 0.0002,
        sigma_split_threshold: float = 0.01,
        prune_amplitude_threshold: float = 0.001,
        clone_scale: float = 1.5,
        max_num_atoms: int = 20000,
    ):
        self.grad_threshold = grad_threshold
        self.sigma_split_threshold = sigma_split_threshold
        self.prune_amplitude_threshold = prune_amplitude_threshold
        self.clone_scale = clone_scale
        self.max_num_atoms = max_num_atoms
        
    def get_densification_mask(
        self,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determine which atoms should be split vs cloned.
        
        REFACTOR: High-freq atoms (>40% Nyquist) are blocked from cloning.
        """
        # Get average accumulated gradients
        avg_grads = model.get_average_gradients()
        
        # High gradient atoms
        high_grad_mask = avg_grads > self.grad_threshold
        
        # Get sigma and omega values
        sigma = model.sigma
        omega = model.omega
        
        # Split: high gradient AND large sigma
        split_mask = high_grad_mask & (sigma > self.sigma_split_threshold)
        
        # REFACTOR: Clone: high gradient AND small sigma AND NOT high frequency
        # High-freq atoms (>40% Nyquist) should Split or Prune, NOT Clone
        # This prevents high-frequency noise artifacts from harmonic doubling
        high_freq_mask = omega > 0.4 * model.nyquist_freq
        clone_mask = high_grad_mask & (sigma <= self.sigma_split_threshold) & (~high_freq_mask)
        
        return split_mask, clone_mask
    
    def get_prune_mask(self, model: nn.Module) -> torch.Tensor:
        """
        Determine which atoms should be pruned (kept=True).
        
        Frequency-aware pruning - high-freq atoms have stricter threshold.
        """
        amplitude = model.amplitude
        omega = model.omega
        nyquist = model.nyquist_freq
        
        base_threshold = self.prune_amplitude_threshold
        
        # High-freq atoms (>70% Nyquist) have 2× stricter threshold
        high_freq_mask = omega > 0.7 * nyquist
        adaptive_threshold = torch.where(
            high_freq_mask,
            base_threshold * 2.0,
            base_threshold
        )
        
        keep_mask = amplitude >= adaptive_threshold
        return keep_mask
    
    def densify_and_prune(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        do_split: bool = False,
        do_clone: bool = True,
        do_prune: bool = True,
    ) -> Dict[str, int]:
        """
        Perform density control with proper optimizer state mapping.
        
        REFACTOR: Uses keep_indices from remove_atoms for correct momentum copying.
        """
        stats = {"split": 0, "cloned": 0, "pruned": 0}
        current_atoms = model.num_atoms
        
        # 1. Get densification masks
        split_mask, clone_mask = self.get_densification_mask(model)
        
        split_indices = torch.where(split_mask)[0] if do_split else torch.tensor([], device=model.device, dtype=torch.long)
        clone_indices = torch.where(clone_mask)[0] if do_clone else torch.tensor([], device=model.device, dtype=torch.long)
        
        # 2. Budget Calculation
        n_split = len(split_indices)
        n_clone = len(clone_indices)
        
        cost_per_split = 2 
        cost_per_clone = 1
        
        projected_add = n_split * cost_per_split + n_clone * cost_per_clone
        remaining_budget = self.max_num_atoms - current_atoms
        
        # 3. Budget Saturation / Truncation
        if remaining_budget <= 0:
            split_indices = torch.tensor([], device=model.device, dtype=torch.long)
            clone_indices = torch.tensor([], device=model.device, dtype=torch.long)
        elif projected_add > remaining_budget:
            split_cost = n_split * cost_per_split
            if split_cost > remaining_budget:
                allowed_splits = remaining_budget // cost_per_split
                split_indices = split_indices[:allowed_splits]
                clone_indices = torch.tensor([], device=model.device, dtype=torch.long)
            else:
                remaining_after_split = remaining_budget - split_cost
                allowed_clones = remaining_after_split // cost_per_clone
                clone_indices = clone_indices[:allowed_clones]
                
        # 4. Execute Operations
        # IMPORTANT: Capture old params BEFORE modification
        old_params = self._get_param_dict(model)
        any_changes = False
        
        # Execute Split
        if len(split_indices) > 0:
            model.split_atoms_by_indices(split_indices)
            stats["split"] = len(split_indices)
            any_changes = True

        # Execute Clone
        if len(clone_indices) > 0:
            num_cloned = model.clone_atoms_by_indices(clone_indices)
            stats["cloned"] = num_cloned
            if num_cloned > 0:
                any_changes = True
        
        # Execute Prune and capture keep_indices
        keep_indices = None
        if do_prune:
            prune_mask = self.get_prune_mask(model)
            num_to_prune = (~prune_mask).sum().item()
            if num_to_prune > 0:
                keep_indices = model.remove_atoms(prune_mask)
                stats["pruned"] = num_to_prune
                any_changes = True

        # 5. Update Optimizer with CORRECT index mapping
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
            "phi_vector": model._phi_vector,  # REFACTORED: 2D
            "gamma": model._gamma,
        }
    
    def _update_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        old_params: Dict[str, nn.Parameter],
        keep_indices: Optional[torch.Tensor] = None,
    ):
        """
        Update optimizer state with CORRECT index mapping after pruning.
        
        CRITICAL FIX: When atoms [0, 2, 4] survive pruning, we must copy
        momentum states from indices [0, 2, 4], NOT [0, 1, 2]!
        
        The old implementation did:
            new_exp_avg[:n_copy] = old_exp_avg[:n_copy]
        
        This is WRONG because it copies consecutive indices, not the 
        surviving indices. This caused "zombie atoms" with incorrect momentum.
        
        Args:
            model: The AudioGS model (already modified)
            optimizer: The Adam optimizer to update
            old_params: Parameter dict from BEFORE modification
            keep_indices: Tensor of indices that survived pruning (from remove_atoms)
        """
        new_params = self._get_param_dict(model)
        
        # 1. Capture current Learning Rates from the old optimizer
        old_param_id_to_lr = {}
        for group in optimizer.param_groups:
            for p in group['params']:
                old_param_id_to_lr[id(p)] = group['lr']
        
        name_to_lr = {}
        for name, old_tensor in old_params.items():
            name_to_lr[name] = old_param_id_to_lr.get(id(old_tensor), 0.001)
        
        # 2. Backup old optimizer states (Momentum)
        old_states = {}
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p in optimizer.state:
                    old_states[id(p)] = optimizer.state[p]
        
        old_param_states = {}
        for name, old_param in old_params.items():
            if id(old_param) in old_states:
                old_param_states[name] = old_states[id(old_param)]
        
        # 3. Rebuild Optimizer Groups (IN-PLACE)
        optimizer.param_groups = []
        optimizer.state = {}
        
        for name, param in new_params.items():
            lr = name_to_lr.get(name, 0.001)
            optimizer.add_param_group({"params": [param], "lr": lr})
            
            # Restore momentum state with CORRECT index mapping
            if name in old_param_states:
                old_state = old_param_states[name]
                if "exp_avg" in old_state:
                    old_exp_avg = old_state["exp_avg"]
                    old_exp_avg_sq = old_state["exp_avg_sq"]
                    
                    # Check if this is a 2D parameter (phi_vector)
                    is_2d = old_exp_avg.dim() == 2
                    
                    if keep_indices is not None:
                        # CRITICAL FIX: keep_indices may contain indices of newly added atoms
                        # (from clone/split) which don't exist in old_exp_avg!
                        # 
                        # Example: After clone adds 5 atoms, keep_indices might be [0, 2, 100, 101, 102]
                        # but old_exp_avg only has 100 elements. Accessing old_exp_avg[100] crashes!
                        #
                        # Solution: Filter to only valid old indices, zero-init new atoms
                        
                        old_len = old_exp_avg.shape[0]
                        valid_mask = keep_indices < old_len
                        valid_old_indices = keep_indices[valid_mask]
                        
                        # Initialize new state tensors with zeros
                        if is_2d:
                            new_exp_avg = torch.zeros_like(param)
                            new_exp_avg_sq = torch.zeros_like(param)
                        else:
                            new_exp_avg = torch.zeros_like(param)
                            new_exp_avg_sq = torch.zeros_like(param)
                        
                        # Copy momentum only for atoms that existed in old optimizer
                        # Map: valid_old_indices[i] in old → i in new (for surviving old atoms)
                        n_valid = len(valid_old_indices)
                        if n_valid > 0:
                            if is_2d:
                                new_exp_avg[:n_valid, :] = old_exp_avg[valid_old_indices, :]
                                new_exp_avg_sq[:n_valid, :] = old_exp_avg_sq[valid_old_indices, :]
                            else:
                                new_exp_avg[:n_valid] = old_exp_avg[valid_old_indices]
                                new_exp_avg_sq[:n_valid] = old_exp_avg_sq[valid_old_indices]
                        
                        # New atoms (from clone/split) get zero momentum - already handled by zeros_like
                    else:
                        # No pruning happened - just atoms were added (clone/split only)
                        # Copy old state for original atoms, zero-init new ones
                        n_copy = min(old_exp_avg.shape[0], param.shape[0])
                        
                        if is_2d:
                            new_exp_avg = torch.zeros_like(param)
                            new_exp_avg_sq = torch.zeros_like(param)
                            new_exp_avg[:n_copy, :] = old_exp_avg[:n_copy, :]
                            new_exp_avg_sq[:n_copy, :] = old_exp_avg_sq[:n_copy, :]
                        else:
                            new_exp_avg = torch.zeros_like(param)
                            new_exp_avg_sq = torch.zeros_like(param)
                            new_exp_avg[:n_copy] = old_exp_avg[:n_copy]
                            new_exp_avg_sq[:n_copy] = old_exp_avg_sq[:n_copy]
                    
                    optimizer.state[param] = {
                        "step": old_state.get("step", torch.tensor(0)),
                        "exp_avg": new_exp_avg,
                        "exp_avg_sq": new_exp_avg_sq,
                    }


class AdaptiveDensityController(DensityController):
    """Adaptive density controller with dynamic thresholds."""
    
    def __init__(
        self,
        grad_threshold: float = 0.0002,
        sigma_split_threshold: float = 0.01,
        prune_amplitude_threshold: float = 0.001,
        clone_scale: float = 1.5,
        max_num_atoms: int = 20000,
        warmup_iters: int = 100,
        decay_factor: float = 0.99,
    ):
        super().__init__(
            grad_threshold=grad_threshold,
            sigma_split_threshold=sigma_split_threshold,
            prune_amplitude_threshold=prune_amplitude_threshold,
            clone_scale=clone_scale,
            max_num_atoms=max_num_atoms,
        )
        self.warmup_iters = warmup_iters
        self.decay_factor = decay_factor
        self.iteration = 0
        self.initial_grad_threshold = grad_threshold
        
    def update_thresholds(self, loss: float):
        self.iteration += 1
        if self.iteration > self.warmup_iters:
            self.grad_threshold = max(
                self.initial_grad_threshold * (self.decay_factor ** (self.iteration - self.warmup_iters)),
                1e-5
            )


def rebuild_optimizer_from_model(
    model: nn.Module,
    optimizer_class: type = torch.optim.Adam,
    lr_config: Optional[Dict[str, float]] = None,
    **optimizer_kwargs,
) -> torch.optim.Optimizer:
    """Helper to create initial optimizer."""
    if lr_config is None:
        lr_config = {}
    
    param_groups = model.get_optimizer_param_groups(lr_config)
    return optimizer_class(param_groups, **optimizer_kwargs)