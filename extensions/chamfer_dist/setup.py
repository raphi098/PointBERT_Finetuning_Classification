# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-10 10:04:25
# @Email:  cshzxie@gmail.com

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='chamfer_cuda_extension',
      version='2.0.0',
      packages=find_packages(), 
      ext_modules=[
          CUDAExtension(name='chamfer_cuda._ext', 
                        sources=[
              'chamfer_cuda/src/chamfer_binding.cpp',
              'chamfer_cuda/src/chamfer.cu',
          ],
              extra_compile_args={
        'cxx': ['-O2'],
        'nvcc': [
            '-O2',
            '-DCUDA_HAS_FP16=1',
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ],}),
      ],
      cmdclass={'build_ext': BuildExtension})
