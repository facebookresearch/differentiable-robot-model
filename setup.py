# Copyright (c) Facebook, Inc. and its affiliates.
######################################################################
# \file setup.py
# \author Franziska Meier
#######################################################################
from setuptools import setup, find_packages

install_requires = ["numpy", "torch", "pyquaternion", "hydra-core", "urdf_parser_py", "jupyter", "tqdm", "matplotlib"]

setup(
    name="differentiable_robot_model",
    author="Facebook AI Research",
    author_email="",
    version=1.0,
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)
