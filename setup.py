import io
import os
import re
from os import path

from setuptools import find_packages
from setuptools import setup


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="jax_nca",
    version="0.1.2",
    url="https://github.com/shyamsn97/jax-nca",
    license='MIT',

    author="Shyam Sudhakaran",
    author_email="shyamsnair@protonmail.com",

    description="Neural Cellular Automata (https://distill.pub/2020/growing-ca/ -- Mordvintsev, et al., \"Growing Neural Cellular Automata\", Distill, 2020) implemented in JAX",

    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=find_packages(exclude=('tests',)),

    install_requires=[
        'numpy',
        'jax',
        'flax',
        'matplotlib',
        'einops',
        'tensorflow',
        'tensorboardX',
        'optax',
        'tqdm',
        'pandas',
        'pillow',
        'ipycanvas', 
        'orjson',
        'opencv-python'
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)
