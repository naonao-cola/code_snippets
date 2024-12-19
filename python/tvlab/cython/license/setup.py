import os
from setuptools import Extension, setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import platform

exec(open('../../tvlab/utils/license_hook.py').read())

system = platform.system()
proc = platform.processor()


extension = Extension("license", ['license.py'])

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extension, annotate=True)
)
