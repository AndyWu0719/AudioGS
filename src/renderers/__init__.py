"""
Renderers package for AudioGS.

Provides both CUDA and CPU rendering backends.
"""
from .cpu_renderer import render_pytorch

# Try to import CUDA renderer
try:
    from cuda_gabor import GaborRendererCUDA, CUDA_EXT_AVAILABLE
except ImportError:
    CUDA_EXT_AVAILABLE = False
    GaborRendererCUDA = None

__all__ = ['render_pytorch', 'GaborRendererCUDA', 'CUDA_EXT_AVAILABLE']
