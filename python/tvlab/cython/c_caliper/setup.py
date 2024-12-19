import os
from setuptools import Extension, setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import platform

exec(open('../../tvlab/utils/license_hook.py').read())

system = platform.system()
proc = platform.processor()

add_flags = []
if 'x86' in proc:
    add_flags += ['-mno-avx512f']

if system in ['Linux', 'Darwin']:
    conda_include = os.path.join(os.environ['CONDA_PREFIX'], "include")
    lib_dir = "./"
    libraries=["opencv_features2d", "opencv_core", "opencv_imgcodecs", "opencv_imgproc", "opencv_dnn"]
else: # windows
    lib_dir = os.path.join(os.environ['CONDA_PREFIX'], "Library", "lib")
    conda_include = os.path.join(os.environ['CONDA_PREFIX'], "Library", "include")
    libraries=["opencv_features2d460", "opencv_core460", "opencv_imgcodecs460", "opencv_imgproc460", "opencv_dnn460"]

def register_hook(fname):
    import os.path as osp
    out_path = 'license_check_' + fname
    with open(fname, 'rt', encoding="utf-8") as fp:
        code = fp.read()

    code_lines = code.splitlines()

    with open(out_path, 'wt', encoding="utf-8") as fp:
        fp.write(code_lines[0]+'\n')
        fp.write(LICENSE_CHECK_IMPORT_CODE)
        fp.writelines('\n'.join(code_lines[1:]))
    return out_path


extension = Extension("c_caliper", [register_hook('c_caliper.pyx'), 'caliper.cpp'],
        include_dirs=[numpy.get_include(), conda_include], libraries=libraries,
        library_dirs=[lib_dir],
        extra_compile_args=['-O4', '-march=native', '-fopenmp', '-Wall'] + add_flags)

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extension, annotate=True)
)
