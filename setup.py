import os
import subprocess
import sys
from distutils.spawn import find_executable

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext


setup(
    name="transformers4rec",
    version="0.01",
    packages=find_packages(),
    url="https://github.com/NVIDIA/NVTabular",
    author="NVIDIA Corporation",
    license="Apache 2.0",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    install_requires=["tqdm>=4.36.1"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering",
    ],
)
