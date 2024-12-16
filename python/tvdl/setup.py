import os
import os.path as osp
import glob
import setuptools
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import sys
import shutil

exec(open('tvdl/utils/license_hook.py').read())

def get_extensions():
    this_dir = osp.dirname(osp.abspath(__file__))
    build_dir = osp.join(this_dir, 'build')
    os.makedirs(build_dir, exist_ok=True)
    # copy tvdl to build/tvdl_py
    out_dir = osp.join(build_dir, 'src/tvdl')
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.copytree(osp.join(this_dir, "tvdl"), out_dir)
    # get py list
    py_list = glob.glob(osp.join(out_dir, '**', '*.py'), recursive=True)
    py_list = [py for py in py_list if 'license_hook' not in py and '__pycache__' not in py]
    code_py_list = [py for py in py_list if '__init__' not in py]
    init_py_list = [py for py in py_list if '__init__' in py]
    # register_hook
    register_license_check_hook(code_py_list)
    return code_py_list, init_py_list


BSO = False
for arg in sys.argv[:]:
    if 'BSO' in arg:
        BSO = True
        sys.argv.remove(arg)

exec(open('tvdl/version.py').read())

def get_packages():
    impl_list = glob.glob('./tvdl/**/impl', recursive=True)
    pkgs = ['tvdl'] + [p[2:].replace(osp.sep, '.') for p in impl_list]
    print('pkgs:', pkgs)
    return pkgs

def get_package_data():
    impl_list = glob.glob('./tvdl/**/impl', recursive=True)
    init_list = glob.glob('./tvdl/**/', recursive=True)
    models_list = glob.glob('./tvdl/**/models', recursive=True)
    pkg_data = {"tvdl": [osp.join(p[7:], '*.so') for p in impl_list] +
                        [osp.join(p[7:], '*.pyd') for p in impl_list] +
                        [osp.join(p[7:], '__init__.py') for p in init_list if '__pycache__' not in p] +
                        [osp.join(p[7:], '*.dll') for p in impl_list] +
                        [osp.join(p[7:], '*.pth') for p in models_list]
                        }
    print('pkg_data:', pkg_data)
    return pkg_data

if BSO:
    code_py_list, init_py_list = get_extensions()
    pkg_data = get_package_data()
    cython_ext_modules = cythonize(code_py_list,
                              build_dir="build",
                              compiler_directives={'language_level': 3})
    setuptools.setup(
        name="tvdl",
        version=__version__,
        author="Allen",
        author_email="allen@turingvision.com",
        description="dl module",
        cmdclass={'build_ext': build_ext},
        packages=get_packages(),
        package_data=pkg_data,
        ext_modules=cython_ext_modules,
        long_description="DeepLearning models.",
        long_description_content_type="text/markdown",
        url="",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        zip_safe=False,
    )
else:
    setuptools.setup(
        name="tvdl",
        version=__version__,
        author="Allen",
        author_email="allen@turingvision.com",
        description="dl module",
        long_description="DeepLearning models.",
        long_description_content_type="text/markdown",
        url="",
        packages=setuptools.find_packages(),
        package_data=get_package_data(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        zip_safe=False,
    )
