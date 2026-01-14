"""
CUDA Gabor Renderer Extension

High-performance Gabor atom rendering using custom CUDA kernels.
"""

from .gabor_render import (
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
