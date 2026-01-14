"""
Setup script for cuda_gabor extension.

Build with: pip install -e .
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda_gabor",
    version="0.1.0",
    description="CUDA-accelerated Gabor atom rendering",
    packages=["cuda_gabor"],
    ext_modules=[
        CUDAExtension(
            name="cuda_gabor._C",
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
)

