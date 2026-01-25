"""
Setup script for cuda_gabor extension.

Build with: pip install -e .

Issue E Fix: Multi-arch CUDA build support.
Override architectures with: TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0" pip install -e .
"""

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# =============================================================================
# CUDA Architecture Configuration
# =============================================================================
# Override with environment variable: TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0"
# If not set, uses a reasonable default covering common GPUs.
#
# Common architectures:
#   sm_70: V100
#   sm_75: T4, RTX 2080
#   sm_80: A100
#   sm_86: RTX 3090
#   sm_89: RTX 4090, L4
#   sm_90: H100
# =============================================================================

def get_cuda_arch_flags():
    """
    Get CUDA architecture flags from environment or defaults.
    
    Priority:
    1. TORCH_CUDA_ARCH_LIST environment variable (e.g., "7.0;7.5;8.0")
    2. Default multi-arch gencodes for common GPUs
    """
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    
    if arch_list:
        # Convert "7.0;7.5;8.0" to nvcc gencode flags
        arches = [a.strip().replace(".", "") for a in arch_list.split(";") if a.strip()]
        flags = []
        for arch in arches:
            flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
        print(f"[cuda_gabor] Using TORCH_CUDA_ARCH_LIST: {arch_list}")
        return flags
    else:
        # Default: cover common GPUs (V100, T4, A100, RTX 30xx, RTX 40xx, H100)
        print("[cuda_gabor] Using default multi-arch build (70, 75, 80, 86, 89, 90)")
        return [
            "-gencode=arch=compute_70,code=sm_70",  # V100
            "-gencode=arch=compute_75,code=sm_75",  # T4, RTX 2080
            "-gencode=arch=compute_80,code=sm_80",  # A100
            "-gencode=arch=compute_86,code=sm_86",  # RTX 3090
            "-gencode=arch=compute_89,code=sm_89",  # RTX 4090, L4
            "-gencode=arch=compute_90,code=sm_90",  # H100
        ]


setup(
    name="cuda_gabor",
    version="0.1.1",  # Version bump for multi-arch support
    description="CUDA-accelerated Gabor atom rendering",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            # Extension goes into the cuda_gabor package directory
            # This matches the relative import "from . import _C" in gabor_render.py
            name="cuda_gabor._C",
            sources=[
                "csrc/ext.cpp",
                "csrc/gabor_render.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                ] + get_cuda_arch_flags(),
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    # Ensure extension is built inplace for editable installs
    options={"build_ext": {"inplace": True}},
)
