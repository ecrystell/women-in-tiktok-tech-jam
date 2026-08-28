#!/usr/bin/env python3
"""Build the small Person 2 FP16 residual/mask CUDA extension."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent

setup(
    name="person2-post-cuda-ext",
    ext_modules=[
        CUDAExtension(
            name="person2_post_cuda_ext",
            sources=[
                str(ROOT / "csrc" / "person2_post_cuda.cpp"),
                str(ROOT / "csrc" / "person2_post_cuda_kernel.cu"),
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
