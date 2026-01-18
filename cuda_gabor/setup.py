"""
Setup script for cuda_gabor extension.

Build with: pip install -e .
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda_gabor",
    version="0.1.0",
    description="CUDA-accelerated Gabor atom rendering",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            # Extension goes into the inner cuda_gabor.cuda_gabor package
            # This matches the relative import "from . import _C" in gabor_render.py
            name="cuda_gabor.cuda_gabor._C",
            sources=[
                "csrc/ext.cpp",
                "csrc/gabor_render.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-arch=sm_80",
                    "--use_fast_math",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    # Ensure extension is built inplace for editable installs
    options={"build_ext": {"inplace": True}},
)
