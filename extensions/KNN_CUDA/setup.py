from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='knn_cuda_extension',
    version='0.2',
    description='pytorch version knn support cuda.',
    author='Shuaipeng Li',
    author_email='sli@mail.bnu.edu.cn',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='knn_cuda._ext',
            sources=[
                'knn_cuda/src/knn_binding.cpp', ## Important c++ binding and cu file are not allowed to have the same name
                'knn_cuda/src/knn.cu',
            ],
            # include_dirs=['knn_cuda/csrc/cuda'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
