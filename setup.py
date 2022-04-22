# Copyright (c) Facebook, Inc. and its affiliates.
######################################################################
# \file setup.py
# \author Franziska Meier
#######################################################################
"""Installation for the differentiable-robot-model project."""

import pathlib
import os
import subprocess
from setuptools import setup

REQUIRES_PYTHON = ">=3.6.0"

# current directory
HERE = pathlib.Path(__file__).parent

# description
README = (HERE / "README.md").read_text()
DESC = (
    "A pytorch library that implements differentiable and learnable robot models, "
    "which allows users to learn parameters of analytical robot models, "
    "and/or propagate gradients through analytical robot computations such as forward kinematics."
)

# resolve version
latest_tag = (
    subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"])
    .decode("utf-8")
    .strip("\n")
)
version_num = latest_tag.strip("v")

branch_name = (
    subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    .decode("utf-8")
    .strip("\n")
)
branch_hash = abs(hash(branch_name)) % (10**4)

rev_num = (
    subprocess.check_output(["git", "rev-list", f"{latest_tag}..HEAD", "--count"])
    .decode("utf-8")
    .strip("\n")
)

VERSION = version_num
if int(rev_num) > 0:
    VERSION = f"{version_num}a{rev_num}.dev{branch_hash}"

# resource files
data_files = []
datadir = "diff_robot_data"

hh = str(HERE) + "/" + datadir
print("folder with datafiles: %s", hh)

for root, dirs, files in os.walk(hh):
    for fn in files:
        ext = os.path.splitext(fn)[1][1:]
        if (
            ext
            and ext
            in "yaml png gif jpg urdf sdf obj txt mtl dae off stl STL xml ".split()
        ):
            fn = root + "/" + fn
            data_files.append(fn[1 + len(hh) :])

print("found resource files: %i" % len(data_files))
for n in data_files:
    print("-- %s" % n)

# dependencies
install_requires = [
    "torch >= 1.6",
    "pyquaternion >= 0.9.9",
    "hydra-core >= 1.0.3",
    "urdf_parser_py >= 0.0.3",
    "Sphinx >= 3.5.4",
    "recommonmark >= 0.7.1",
]

# run setup
setup(
    name="differentiable-robot-model",
    version=VERSION,
    description=DESC,
    long_description=README,
    long_description_content_type="text/markdown",
    author="Franziska Meier",
    author_email="fmeier@fb.com",
    python_requires=REQUIRES_PYTHON,
    url="https://github.com/facebookresearch/differentiable-robot-model",
    keywords="robotics, differentiable, optimization",
    packages=["differentiable_robot_model", "diff_robot_data"],
    install_requires=install_requires,
    license="MIT",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    package_data={"diff_robot_data": data_files},
)
