from setuptools import Extension, setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import os

import platform
system = platform.system()


if system == 'Linux':
    setup(
        cmdclass={'build_ext': build_ext},
        ext_modules = cythonize([Extension("c_et199",
                                           ["c_et199.pyx", "et199_api.c"],
                                           #libraries=["ET199"],
                                           extra_objects=["libET199.a"],
                                           include_dirs=[numpy.get_include()],
                                           )]),
    )
else:
    extra_compile_args=['/W3', '/GX', '/O2', '/D "WIN32"', '/D "NDEBUG"', '/D "_CONSOLE"', '/D "_MBCS"', '/YX', '/FD', '/c']

    setup(
        cmdclass={'build_ext': build_ext},
        ext_modules = cythonize([Extension("c_et199",
                                           ["c_et199.pyx", "win_et199_api.c"],
                                           extra_objects=["ET199_64.lib"],
                                           include_dirs=[numpy.get_include()],
                                           extra_compile_args=extra_compile_args,
                                           )]),
    )
