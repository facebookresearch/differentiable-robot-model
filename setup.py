# Copyright (c) Facebook, Inc. and its affiliates.
######################################################################
# \file setup.py
# \author Franziska Meier
#######################################################################
"""Installation function for the differentiable-robot-model project."""

import pathlib
from setuptools import setup

# current directory
HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()
DESC = ('A pytorch library that implements differentiable and learnable robot models, '
        'which allows users to learn parameters of analytical robot models, '
        'and/or propagate gradients through analytical robot computations such as forward kinematics.')

REQUIRES_PYTHON = '>=3.6.0'
VERSION = (HERE / "version.txt").read_text().strip()

install_requires = [line.rstrip() for line in open("requirements.txt", "r")]

# run setup
setup(
    name='differentiable_robot_model',
    version=VERSION,
    description=DESC,
    long_description=README,
    long_description_content_type="text/markdown",
    author='Franziska Meier',
    author_email='fmeier@fb.com',
    python_requires=REQUIRES_PYTHON,
    url="https://github.com/facebookresearch/differentiable-robot-model",
    keywords='analytical robot models, differentiable, optimization',
    packages=['differentiable_robot_model'],
    install_requires=install_requires,
    include_package_data=True,
    license="MIT",
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)

