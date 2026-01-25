"""
Shared configuration loading utilities for AudioGS.

Provides a single source of truth for loading and accessing config,
ensuring consistent behavior across all entry points.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


# =============================================================================
# Shared Constants (single source of truth)
# =============================================================================
# These values MUST be consistent with src/models/atom.py SIGMA_OFFSET
SIGMA_OFFSET = 0.001  # 1ms minimum sigma (matches CUDA kernel clamp)

# Default initialization values (used when config is absent)
DEFAULT_INIT_CONFIG = {
    "constant_q_cycles": 3.5,
    "sigma_min": 0.002,  # 2ms
    "sigma_max": 0.050,  # 50ms
    "omega_min_hz": 50.0,
    "constant_q_use_2pi": True,
}


@dataclass
class InitConfig:
    """
    Initialization configuration for AudioGS models.
    
    This dataclass ensures consistent initialization across all entry points.
    """
    constant_q_cycles: float = 3.5
    sigma_min: float = 0.002
    sigma_max: float = 0.050
    omega_min_hz: float = 50.0
    constant_q_use_2pi: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]] = None) -> 'InitConfig':
        """Create from dictionary, using defaults for missing values."""
        if config_dict is None:
            config_dict = {}
        return cls(
            constant_q_cycles=config_dict.get("constant_q_cycles", DEFAULT_INIT_CONFIG["constant_q_cycles"]),
            sigma_min=config_dict.get("sigma_min", DEFAULT_INIT_CONFIG["sigma_min"]),
            sigma_max=config_dict.get("sigma_max", DEFAULT_INIT_CONFIG["sigma_max"]),
            omega_min_hz=config_dict.get("omega_min_hz", DEFAULT_INIT_CONFIG["omega_min_hz"]),
            constant_q_use_2pi=config_dict.get(
                "constant_q_use_2pi", DEFAULT_INIT_CONFIG["constant_q_use_2pi"]
            ),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for passing to model methods."""
        return {
            "constant_q_cycles": self.constant_q_cycles,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "omega_min_hz": self.omega_min_hz,
            "constant_q_use_2pi": self.constant_q_use_2pi,
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary with configuration values
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_init_config(config: Optional[Dict[str, Any]] = None) -> InitConfig:
    """
    Extract initialization config from full config dict.
    
    Args:
        config: Full config dictionary (may contain 'initialization' key)
        
    Returns:
        InitConfig with proper defaults
    """
    if config is None:
        return InitConfig()
    
    init_section = config.get("initialization", {})
    return InitConfig.from_dict(init_section)


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate from src/utils/config.py to project root
    return Path(__file__).parent.parent.parent


def get_default_config_path() -> Path:
    """Get path to default config file."""
    return get_project_root() / "configs" / "atom_fitting_config.yaml"
