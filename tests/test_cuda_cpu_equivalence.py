"""
Tests for AudioGS Code Maintenance Fixes.

Tests cover:
- (A) CUDA/CPU forward equivalence and gradient finiteness
- (B) Inverse transform roundtrips for densification
- (D) Config initialization wiring
- init_random config wiring

Run: pytest tests/test_cuda_cpu_equivalence.py -v
"""
import torch
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "cuda_gabor"))

# Import helpers from atom.py
from src.models.atom import (
    AudioGSModel, 
    SIGMA_OFFSET,
    _inv_softplus, 
    _sigma_real_to_logit, 
    _omega_real_to_logit
)

# Import shared config utilities
from src.utils.config import InitConfig, get_init_config, DEFAULT_INIT_CONFIG

# Try to import CUDA renderer
try:
    from cuda_gabor import GaborRendererCUDA, CUDA_EXT_AVAILABLE
except ImportError:
    CUDA_EXT_AVAILABLE = False
    GaborRendererCUDA = None

# Import CPU renderer from proper package location (Issue 1 fix)
from src.renderers.cpu_renderer import render_pytorch


# =============================================================================
# Test: Inverse Transform Helpers (Issue B)
# =============================================================================
class TestInverseTransforms:
    """Test inverse transform helpers for densification."""
    
    def test_inv_softplus_roundtrip(self):
        """Test inv_softplus correctly inverts softplus."""
        x = torch.linspace(-5, 5, 100)
        y = torch.nn.functional.softplus(x)
        x_recovered = _inv_softplus(y)
        
        assert torch.allclose(x, x_recovered, atol=1e-4), \
            f"inv_softplus roundtrip failed: max error = {(x - x_recovered).abs().max()}"
    
    def test_inv_softplus_large_values(self):
        """Test inv_softplus with large values (numerical stability)."""
        y = torch.tensor([30.0, 50.0, 100.0])
        x_recovered = _inv_softplus(y)
        y_check = torch.nn.functional.softplus(x_recovered)
        
        assert torch.allclose(y, y_check, atol=1e-3), \
            f"inv_softplus large value test failed"
    
    def test_sigma_logit_roundtrip(self):
        """Test sigma real->logit->real roundtrip."""
        # Test various sigma values above minimum
        sigma_real = torch.tensor([0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
        logits = _sigma_real_to_logit(sigma_real)
        sigma_recovered = torch.nn.functional.softplus(logits) + SIGMA_OFFSET
        
        assert torch.allclose(sigma_real, sigma_recovered, atol=1e-5), \
            f"Sigma roundtrip failed: {sigma_real} != {sigma_recovered}"
    
    def test_omega_logit_roundtrip(self):
        """Test omega real->logit->real roundtrip using 0.99*nyquist."""
        nyquist = 12000.0
        # Test various omega values within valid range
        omega_real = torch.tensor([100.0, 500.0, 2000.0, 5000.0, 10000.0])
        
        logits = _omega_real_to_logit(omega_real, nyquist)
        omega_recovered = torch.sigmoid(logits) * 0.99 * nyquist
        
        assert torch.allclose(omega_real, omega_recovered, atol=1.0), \
            f"Omega roundtrip failed: {omega_real} != {omega_recovered}"


# =============================================================================
# Test: CPU Renderer Import (Issue 1)
# =============================================================================
class TestCpuRendererImport:
    """Test that CPU renderer can be imported correctly."""
    
    def test_import_from_renderers_package(self):
        """Verify render_pytorch can be imported from src.renderers."""
        from src.renderers import render_pytorch as rp
        assert callable(rp), "render_pytorch should be callable"
    
    def test_cpu_renderer_basic(self):
        """Test CPU renderer produces output."""
        device = torch.device("cpu")
        output = render_pytorch(
            amplitude=torch.tensor([0.5], device=device),
            tau=torch.tensor([0.1], device=device),
            omega=torch.tensor([440.0], device=device),
            sigma=torch.tensor([0.01], device=device),
            phi=torch.tensor([0.0], device=device),
            gamma=torch.tensor([0.0], device=device),
            num_samples=1000,
            sample_rate=24000,
            device=device
        )
        assert output.shape == (1000,), f"Expected shape (1000,), got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains non-finite values"


# =============================================================================
# Test: CUDA/CPU Equivalence (Issues A, C)
# =============================================================================
class TestCudaCpuEquivalence:
    """Test CUDA and CPU renderers produce equivalent outputs."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def sample_atoms(self, device):
        """Create a small set of test atoms."""
        return {
            "amplitude": torch.tensor([0.5, 0.3, 0.8], device=device),
            "tau": torch.tensor([0.1, 0.25, 0.4], device=device),
            "omega": torch.tensor([440.0, 880.0, 220.0], device=device),
            "sigma": torch.tensor([0.01, 0.005, 0.02], device=device),
            "phi": torch.tensor([0.0, 1.57, 3.14], device=device),
            "gamma": torch.tensor([0.0, 0.0, 0.0], device=device),
        }
    
    @pytest.mark.skipif(not CUDA_EXT_AVAILABLE, reason="CUDA extension not available")
    def test_forward_equivalence(self, device, sample_atoms):
        """Test that CUDA and CPU forward pass produce similar outputs."""
        sample_rate = 24000
        num_samples = 12000  # 0.5 seconds
        
        renderer_cuda = GaborRendererCUDA(sample_rate=sample_rate, sigma_multiplier=5.0)
        
        # CUDA forward
        output_cuda = renderer_cuda(
            sample_atoms["amplitude"],
            sample_atoms["tau"],
            sample_atoms["omega"],
            sample_atoms["sigma"],
            sample_atoms["phi"],
            sample_atoms["gamma"],
            num_samples
        )
        
        # CPU forward  
        output_cpu = render_pytorch(
            sample_atoms["amplitude"],
            sample_atoms["tau"],
            sample_atoms["omega"],
            sample_atoms["sigma"],
            sample_atoms["phi"],
            sample_atoms["gamma"],
            num_samples,
            sample_rate,
            device,
            sigma_multiplier=5.0
        )
        
        # Check outputs are close (tolerance for floating point differences)
        max_diff = (output_cuda - output_cpu).abs().max().item()
        assert max_diff < 1e-3, f"CUDA/CPU mismatch: max diff = {max_diff}"
    
    @pytest.mark.skipif(not CUDA_EXT_AVAILABLE, reason="CUDA extension not available")
    def test_gradients_finite(self, device):
        """Test that gradients are finite for small sigma (Issue A)."""
        sample_rate = 24000
        num_samples = 12000
        
        # Create atom with SMALL sigma (edge case)
        amplitude = torch.tensor([0.5], device=device, requires_grad=True)
        tau = torch.tensor([0.25], device=device, requires_grad=True)
        omega = torch.tensor([440.0], device=device, requires_grad=True)
        sigma = torch.tensor([0.001], device=device, requires_grad=True)  # 1ms - minimum
        phi = torch.tensor([0.0], device=device, requires_grad=True)
        gamma = torch.tensor([0.0], device=device, requires_grad=True)
        
        renderer = GaborRendererCUDA(sample_rate=sample_rate, sigma_multiplier=5.0)
        
        output = renderer(amplitude, tau, omega, sigma, phi, gamma, num_samples)
        loss = output.sum()
        loss.backward()
        
        # Check all gradients are finite
        for name, param in [("amplitude", amplitude), ("tau", tau), 
                            ("omega", omega), ("sigma", sigma),
                            ("phi", phi), ("gamma", gamma)]:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert torch.isfinite(param.grad).all(), \
                f"Non-finite gradient for {name}: {param.grad}"


# =============================================================================
# Test: Densification Correctness (Issue B)
# =============================================================================
class TestDensificationCorrectness:
    """Test that cloned/split atoms have correct real values."""
    
    @pytest.fixture
    def model(self):
        model = AudioGSModel(
            num_atoms=10, 
            sample_rate=24000, 
            audio_duration=1.0,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        model.init_random()
        return model
    
    def test_clone_sigma_matches_target(self, model):
        """After clone, child sigma should be parent_sigma / 2."""
        # Record parent sigmas
        parent_indices = torch.tensor([0, 3, 5], device=model.device)
        parent_sigma = model.sigma[parent_indices].clone()
        parent_omega = model.omega[parent_indices].clone()
        
        # Only test atoms that won't be filtered (omega*2 < 95% Nyquist)
        valid_mask = (parent_omega * 2) < (0.95 * model.nyquist_freq)
        if not valid_mask.any():
            pytest.skip("No valid atoms for clone test (all would exceed Nyquist)")
        
        num_before = model.num_atoms
        num_cloned = model.clone_atoms_by_indices(parent_indices)
        
        if num_cloned == 0:
            pytest.skip("No atoms were cloned (all filtered)")
        
        # Check new atoms have sigma = parent_sigma / 2
        new_sigmas = model.sigma[-num_cloned:]
        
        # Expected: sigma/2 clamped to minimum
        valid_parent_sigma = parent_sigma[valid_mask]
        expected_sigma = (valid_parent_sigma / 2).clamp(min=SIGMA_OFFSET + 1e-6)
        
        assert torch.allclose(new_sigmas, expected_sigma, atol=1e-4), \
            f"Clone sigma mismatch:\n  Got: {new_sigmas}\n  Expected: {expected_sigma}"


# =============================================================================
# Test: Config Initialization (Issues 2, 3, D)
# =============================================================================
class TestConfigInitialization:
    """Test that config fields are wired to initialization."""
    
    def test_init_random_respects_config(self):
        """Issue 2: Verify init_random uses config values correctly."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Test with default (no config)
        model_default = AudioGSModel(
            num_atoms=100, sample_rate=24000, audio_duration=1.0, device=device
        )
        model_default.init_random()
        sigma_default = model_default.sigma.clone()
        
        # Test with custom config
        model_custom = AudioGSModel(
            num_atoms=100, sample_rate=24000, audio_duration=1.0, device=device
        )
        custom_config = {
            "constant_q_cycles": 5.0,
            "sigma_min": 0.005,
            "sigma_max": 0.025,
        }
        model_custom.init_random(init_config=custom_config)
        sigma_custom = model_custom.sigma.clone()
        
        # Check custom bounds are respected
        assert sigma_custom.min() >= 0.005 - 1e-6, \
            f"sigma_min not respected in init_random: min={sigma_custom.min()}"
        assert sigma_custom.max() <= 0.025 + 1e-6, \
            f"sigma_max not respected in init_random: max={sigma_custom.max()}"
    
    def test_init_config_dataclass(self):
        """Test InitConfig dataclass works correctly."""
        # From dict
        config = InitConfig.from_dict({
            "constant_q_cycles": 4.0,
            "sigma_min": 0.003,
        })
        assert config.constant_q_cycles == 4.0
        assert config.sigma_min == 0.003
        assert config.sigma_max == 0.050  # Default when not specified
        
        # To dict roundtrip
        d = config.to_dict()
        assert d["constant_q_cycles"] == 4.0
        assert d["sigma_min"] == 0.003
    
    def test_initialize_from_audio_respects_config(self):
        """Verify that initialize_from_audio uses config values."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample_rate = 24000
        duration = 0.5
        t = torch.linspace(0, duration, int(sample_rate * duration), device=device)
        waveform = torch.sin(2 * 3.14159 * 440 * t)
        
        # Test with custom config (different bounds)
        model = AudioGSModel(
            num_atoms=100, sample_rate=sample_rate,
            audio_duration=duration, device=device
        )
        custom_config = {
            "constant_q_cycles": 4.0,
            "sigma_min": 0.005,
            "sigma_max": 0.030,
        }
        model.initialize_from_audio(waveform, init_config=custom_config)
        sigma = model.sigma.clone()
        
        # Check sigma bounds are respected
        assert sigma.min() >= 0.005 - 1e-6, \
            f"sigma_min not respected: min={sigma.min()}"
        assert sigma.max() <= 0.030 + 1e-6, \
            f"sigma_max not respected: max={sigma.max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
