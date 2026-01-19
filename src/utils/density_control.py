"""
Density Control for Audio Gaussian Splatting.
Revised: Implements 'Saturation' logic and Preserves Learning Rate Schedule.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import math


class DensityController:
    """
    Manages density control operations for AudioGS model.
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
        """Determine which atoms should be split vs cloned."""
        # Get average accumulated gradients
        avg_grads = model.get_average_gradients()
        
        # High gradient atoms
        high_grad_mask = avg_grads > self.grad_threshold
        
        # Get sigma values
        sigma = model.sigma
        
        # Split: high gradient AND large sigma
        split_mask = high_grad_mask & (sigma > self.sigma_split_threshold)
        
        # Clone: high gradient AND small sigma
        clone_mask = high_grad_mask & (sigma <= self.sigma_split_threshold)
        
        return split_mask, clone_mask
    
    def get_prune_mask(self, model: nn.Module) -> torch.Tensor:
        """
        Determine which atoms should be pruned (kept=True).
        
        gemini fixed: Frequency-aware pruning - high-freq atoms have stricter threshold
        to remove static atoms causing the "bright band" artifact.
        """
        amplitude = model.amplitude
        omega = model.omega
        nyquist = model.nyquist_freq
        
        # Base threshold
        base_threshold = self.prune_amplitude_threshold
        
        # High-freq atoms (>70% Nyquist) have 2× stricter threshold
        high_freq_mask = omega > 0.7 * nyquist
        adaptive_threshold = torch.where(
            high_freq_mask,
            base_threshold * 2.0,  # Stricter for high freq
            base_threshold
        )
        
        keep_mask = amplitude >= adaptive_threshold
        return keep_mask
    
    def densify_and_prune(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        do_split: bool = False,  # DISABLED by default (v2.0 fix: prevents beating artifacts)
        do_clone: bool = True,
        do_prune: bool = True,
    ) -> Dict[str, int]:
        """
        Perform density control with saturation logic.
        """
        stats = {"split": 0, "cloned": 0, "pruned": 0}
        current_atoms = model.num_atoms
        
        # 1. Pruning First
        split_mask, clone_mask = self.get_densification_mask(model)
        
        split_indices = torch.where(split_mask)[0] if do_split else torch.tensor([], device=model.device)
        clone_indices = torch.where(clone_mask)[0] if do_clone else torch.tensor([], device=model.device)
        
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
        # IMPORTANT: Capture old params BEFORE modification to track LRs and State
        old_params = self._get_param_dict(model)
        any_changes = False
        
        # Execute Split
        if len(split_indices) > 0:
            model.split_atoms_by_indices(split_indices)
            stats["split"] = len(split_indices)
            any_changes = True

        # Execute Clone
        if len(clone_indices) > 0:
            model.clone_atoms_by_indices(clone_indices)
            stats["cloned"] = len(clone_indices)
            any_changes = True
            
        # Execute Prune
        if do_prune:
            prune_mask = self.get_prune_mask(model)
            num_to_prune = (~prune_mask).sum().item()
            if num_to_prune > 0:
                model.remove_atoms(prune_mask)
                stats["pruned"] = num_to_prune
                any_changes = True

        # 5. Update Optimizer
        if any_changes:
            self._update_optimizer_state(model, optimizer, old_params)
            model.reset_gradient_accumulators()
        
        return stats
    
    def _get_param_dict(self, model: nn.Module) -> Dict[str, nn.Parameter]:
        return {
            "amplitude_logit": model._amplitude_logit,
            "tau": model._tau,
            "omega_logit": model._omega_logit,
            "sigma_logit": model._sigma_logit,
            "phi": model._phi,
            "gamma": model._gamma,
        }
    
    def _update_optimizer_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        old_params: Dict[str, nn.Parameter],
    ):
        """Update optimizer state preserving momentum AND current learning rates."""
        new_params = self._get_param_dict(model)
        
        # 1. Capture current Learning Rates from the old optimizer
        # Map: id(old_tensor) -> current_lr
        old_param_id_to_lr = {}
        for group in optimizer.param_groups:
            for p in group['params']:
                old_param_id_to_lr[id(p)] = group['lr']

        # Map: param_name -> current_lr
        # This ensures we respect the Scheduler's decay
        name_to_lr = {}
        for name, old_tensor in old_params.items():
            if id(old_tensor) in old_param_id_to_lr:
                name_to_lr[name] = old_param_id_to_lr[id(old_tensor)]
            else:
                # Fallback (should typically not happen if init is correct)
                name_to_lr[name] = 0.001 

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
        
        # 3. Rebuild Optimizer Groups (IN-PLACE modification)
        optimizer.param_groups = []
        optimizer.state = {}
        
        for name, param in new_params.items():
            # Use the CAPTURED lr, not a hardcoded one
            lr = name_to_lr.get(name, 0.001)
            optimizer.add_param_group({"params": [param], "lr": lr})
            
            # Restore momentum state
            if name in old_param_states:
                old_state = old_param_states[name]
                if "exp_avg" in old_state:
                    n_copy = min(old_state["exp_avg"].shape[0], param.shape[0])
                    
                    new_exp_avg = torch.zeros_like(param)
                    new_exp_avg_sq = torch.zeros_like(param)
                    
                    new_exp_avg[:n_copy] = old_state["exp_avg"][:n_copy]
                    new_exp_avg_sq[:n_copy] = old_state["exp_avg_sq"][:n_copy]
                    
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
        lr_config = {} # Rely on defaults in model
    
    param_groups = model.get_optimizer_param_groups(lr_config)
    return optimizer_class(param_groups, **optimizer_kwargs)