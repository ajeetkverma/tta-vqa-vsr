from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='correlation',
    ext_modules=[
        cpp_extension.CppExtension(
            'correlation_cpp', # Name of the module used in pybind
            ['correlation.cpp'], #  source files
            extra_compile_args={'cxx': ['-fopenmp', '-D PYTHON_BUILD']},
            extra_link_args=['-lgomp'])
    ],
    author='K|Lens GmbH',
    author_email='samim.taray@k-lens.de',
    install_requires=['torch>=1.1', 'numpy'],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)
