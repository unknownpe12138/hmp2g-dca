from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform

extensions = [
    Extension("UTIL.shm_pool", ["UTIL/shm_pool.pyx"],
              include_dirs=[np.get_include()]),
    Extension("UTIL.tensor_ops_c", ["UTIL/tensor_ops_c.pyx"],
              include_dirs=[np.get_include()]),
    Extension("MISSION.dca.cython_func", ["MISSION/dca/cython_func.pyx"],
              include_dirs=[np.get_include()]),
    Extension("ALGORITHM.conc_4hist.cython_func",
              ["ALGORITHM/conc_4hist/cython_func.pyx"],
              include_dirs=[np.get_include()]),
    Extension("ALGORITHM.common.cython_func",
              ["ALGORITHM/common/cython_func.pyx"],
              include_dirs=[np.get_include()]),
]

setup(
    name="hmp2g-dca",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': '3'}),
)
