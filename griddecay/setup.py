import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

parent_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='griddecay',
    packages=['griddecay'],
    ext_modules=[
        CUDAExtension(
            name='griddecay._C',
            sources=[os.path.join(parent_dir, path) for path in ['grid_offset.cpp']],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
