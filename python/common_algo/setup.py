import setuptools

exec(open('common_algo/version.py').read())

setuptools.setup(
    name="common_algo",
    version=__version__,
    author="Xiaobo",
    author_email="xiaobo0619@thundersoft.com",
    description="common_algo",
    long_description="Common DeepLearning model interface for training, inference and package.",
    long_description_content_type="text/markdown",
    url="ssh://xiaobo0619@192.168.87.197:29418/IV/industry/algo/tscv-product/common_algo",
    packages=setuptools.find_packages(),
    package_data={
        "":["configs/*.yaml"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
