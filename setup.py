import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


root = os.path.dirname(os.path.abspath(__file__))
extensions_dir = os.path.join(root, "maskrcnn_benchmark", "csrc")

sources = [os.path.join(extensions_dir, "vision.cpp")]
sources += glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))

extension = CppExtension
define_macros = []
extra_compile_args = {"cxx": []}

if torch.cuda.is_available():
    extension = CUDAExtension
    sources += glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    define_macros.append(("WITH_CUDA", None))
    extra_compile_args["nvcc"] = [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]

setup(
    name="maskrcnn_benchmark",
    version="0.0.1",
    packages=find_packages(exclude=("configs", "tests")),
    ext_modules=[
        extension(
            "maskrcnn_benchmark._C",
            sources,
            include_dirs=[extensions_dir],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

