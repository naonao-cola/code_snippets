from setuptools import Extension, setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import os
import platform
system = platform.system()
proc = platform.processor()

exec(open('../../tvlab/utils/license_hook.py').read())

add_flags = []
if 'x86' in proc:
    add_flags += ['-mno-avx512f']

if system in ['Linux', 'Darwin']:
    cuda_include = os.path.join(os.environ['CONDA_PREFIX'], "include")
    lib_dir = "./"
    libraries=["opencv_features2d", "opencv_core", "opencv_imgcodecs", "opencv_imgproc", "opencv_dnn"]
    extra_compile_args=['-O3', '-march=native', '-fopenmp', '-Wall'] + add_flags
else: # windows
    lib_dir = os.path.join(os.environ['CONDA_PREFIX'], "Library", "lib")
    cuda_include = os.path.join(os.environ['CONDA_PREFIX'], "Library", "include")
    libraries=["opencv_features2d460", "opencv_core460", "opencv_imgcodecs460", "opencv_imgproc460", "opencv_dnn460"]
    extra_compile_args=['/MP', '/GS-', '/Qpar', '/GL', '/W0', '/Gy', '/Gm-', '/O2', '/fp:fast', '/fp:except-',
            '/errorReport:prompt', '/GF', '/WX-', '/Zc:forScope-', '/GR-', '/arch:AVX2', '/Gr', '/Oy', '/Oi', '/MT', '/std:c++14']
eigen_include = os.path.join(cuda_include, "eigen3")

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

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules = cythonize([Extension("cshape_based_matching",
                                       [register_hook("cshape_based_matching.pyx"),
                                        "sbm_algo.cpp", "line2Dup.cpp", "cuda_icp/icp.cpp",
                                        "cuda_icp/scene/common.cpp", "cuda_icp/scene/edge_scene/edge_scene.cpp"],
                                       libraries=libraries,
                                       library_dirs=[lib_dir],
                                       include_dirs=[numpy.get_include(), cuda_include, eigen_include, "./MIPP", "./cuda_icp"],
                                       extra_compile_args=extra_compile_args,
                                       )]),
)
#extra_compile_args=['-O3', '-march=native', '-mno-avx512f', '-fopenmp', '-Wall'],
