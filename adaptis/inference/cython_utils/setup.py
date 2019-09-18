from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("utils", ["utils.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-O3'],
              language='c++11'),
]

setup(
    ext_modules=cythonize(extensions),
)