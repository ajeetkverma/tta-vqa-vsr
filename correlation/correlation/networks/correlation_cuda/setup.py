#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-D PYTHON_BUILD', '-fopenmp']
nvcc_args = [
    '-gencode', 'arch=compute_37,code=sm_37',
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_70,code=compute_70'
]

setup(
    name='correlation_cuda',
    ext_modules=[
        CUDAExtension('correlation_cuda', # Name of the module used in pybind
        [
            'correlation_cuda.cc', #  source files
            'correlation_cuda_kernel.cu' 
        ], extra_compile_args={
            'cxx': cxx_args, 
            'nvcc': nvcc_args}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
