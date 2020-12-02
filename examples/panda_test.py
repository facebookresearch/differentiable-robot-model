# Copyright (c) Facebook, Inc. and its affiliates.
import os
import random
import torch
import numpy as np
import pytest
from hydra.experimental import compose as hydra_compose
from hydra.experimental import initialize_config_dir

import pybullet as p
import robot_data
from differentiable_robot_model.differentiable_robot_model import DifferentiableRobotModel

robot_description_folder = robot_data.__path__[0]

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)
torch.set_default_tensor_type(torch.DoubleTensor)

rel_urdf_path = "panda_description/urdf/panda.urdf"
urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
dof = 9
print("DOF+++++++++++++++++++++++++++++++++++++++++")
pc_id = p.connect(p.DIRECT)
robot_id = p.loadURDF(
    urdf_path,
    basePosition=[0, 0, 0],
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE,
)

p.setGravity(0, 0, -9.81)
JOINT_DAMPING = 0.5

print("JOINT INFO")
num_joints = p.getNumJoints(robot_id)
for i in range(num_joints):
    print(p.getJointInfo(robot_id, i))
print("JOINT INFO")
