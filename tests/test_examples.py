import os
import sys
import subprocess

import pytest


@pytest.mark.parametrize(
    "relpath",
    [
        "../examples/learn_dynamics_iiwa.py",
        "../examples/learn_kinematics_of_toy.py",
    ],
)
def test_examples(relpath):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    py_file = os.path.abspath(os.path.join(curr_dir, relpath))
    cmd = [sys.executable, py_file]
    subprocess.check_call(cmd)