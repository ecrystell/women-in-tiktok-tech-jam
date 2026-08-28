#!/usr/bin/env python3
"""Build the Person 2 cuBLASLt extension without requiring Ninja."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent

setup(
    name="person2-ffn-cuda-ext",
    ext_modules=[
        CUDAExtension(
            name="person2_ffn_cuda_ext",
            sources=[
                str(ROOT / "csrc" / "person2_ffn_cuda.cpp"),
                str(ROOT / "csrc" / "person2_ffn_cuda_kernel.cu"),
            ],
            libraries=["cublasLt"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
