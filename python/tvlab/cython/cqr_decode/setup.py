from setuptools import Extension, setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import os
import platform
proc = platform.processor()

exec(open('../../tvlab/utils/license_hook.py').read())

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

add_flags = []
if 'x86' in proc:
    add_flags += ['-mno-avx512f']

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules = cythonize([Extension("cqr_decode",
                                       [register_hook("cqr_decode.pyx"), "decode.c",  "identify.c",
                                        "quirc.c", "version_db.c"],
                                       include_dirs=[numpy.get_include()],
                                       extra_compile_args=['-O3', '-march=native', '-fopenmp', '-Wall'] + add_flags,
                                       )]),
)
