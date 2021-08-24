import codecs
import itertools
import os

from setuptools import find_packages, setup


def read_requirements(filename):
    base = os.path.abspath(os.path.dirname(__file__))

    parsed = []
    with codecs.open(os.path.join(base, filename), "rb", "utf-8") as f:
        parsed.append(f.read())

    return parsed


requirements = {
    "base": read_requirements("requirements/base.txt"),
    "tensorflow": read_requirements("requirements/tensorflow.txt"),
    "pytorch": read_requirements("requirements/pytorch.txt"),
    # "nvtabular": read_requirements("requirements/nvtabular.txt"),
    "dev": read_requirements("requirements/dev.txt"),
}

setup(
    name="transformers4rec",
    version="0.01",
    packages=find_packages(),
    url="https://github.com/NVIDIA/NVTabular",
    author="NVIDIA Corporation",
    license="Apache 2.0",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    install_requires=requirements["base"],
    test_suite="tests",
    extras_require={**requirements, "all": list(itertools.chain(*list(requirements.values())))},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering",
    ],
)
