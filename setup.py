# Copyright (c) Facebook, Inc. and its affiliates.
######################################################################
# \file setup.py
# \author Franziska Meier
#######################################################################
"""Installation for the differentiable-robot-model project."""

import pathlib
import os
from setuptools import setup

# current directory
HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()
DESC = ('A pytorch library that implements differentiable and learnable robot models, '
        'which allows users to learn parameters of analytical robot models, '
        'and/or propagate gradients through analytical robot computations such as forward kinematics.')

REQUIRES_PYTHON = '>=3.6.0'
VERSION = "0.1.1-6"

data_files = []
datadir = "diff_robot_data"

hh = str(HERE) + "/" + datadir
print("folder with datafiles: %s", hh)

for root, dirs, files in os.walk(hh):
  for fn in files:
    ext = os.path.splitext(fn)[1][1:]
    if ext and ext in 'yaml png gif jpg urdf sdf obj txt mtl dae off stl STL xml '.split(
    ):
      fn = root + "/" + fn
      data_files.append(fn[1 + len(hh):])

print("found resource files: %i" % len(data_files))
for n in data_files:
  print("-- %s" % n)

install_requires = ['torch >= 1.6', 'pyquaternion >= 0.9.9', 'hydra-core >= 1.0.3', 'urdf_parser_py >= 0.0.3']
# run setup
setup(
    name='differentiable-robot-model',
    version=VERSION,
    description=DESC,
    long_description=README,
    long_description_content_type="text/markdown",
    author='Franziska Meier',
    author_email='fmeier@fb.com',
    python_requires=REQUIRES_PYTHON,
    url="https://github.com/facebookresearch/differentiable-robot-model",
    keywords='robotics, differentiable, optimization',
    packages=['differentiable_robot_model', 'diff_robot_data'],
    install_requires=install_requires,
    license="MIT",
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
    package_data={'diff_robot_data': data_files}
)

