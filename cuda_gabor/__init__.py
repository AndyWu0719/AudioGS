"""
CUDA Gabor Renderer Extension

High-performance Gabor atom rendering using custom CUDA kernels.
"""

# The extension _C is built into this package directly
# Re-export from the inner cuda_gabor subpackage for API
from .cuda_gabor.gabor_render import (
    GaborRendererCUDA,
    GaborRenderFunction,
    get_cuda_gabor_renderer,
    CUDA_EXT_AVAILABLE,
)

__all__ = [
    "GaborRendererCUDA",
    "GaborRenderFunction",
    "get_cuda_gabor_renderer",
    "CUDA_EXT_AVAILABLE",
]

__version__ = "0.1.0"
