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

rel_urdf_path = "panda_description/urdf/panda_no_gripper.urdf"
urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
dof = 7
print("DOF+++++++++++++++++++++++++++++++++++++++++")
pc_id = p.connect(p.DIRECT)
robot_id = p.loadURDF(
    urdf_path,
    basePosition=[0, 0, 0],
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE,
)

p.setGravity(0, 0, -9.81)
JOINT_DAMPING = 0.0

# for link_idx in range(7):
#     p.changeDynamics(
#         robot_id,
#         link_idx,
#         linearDamping=0.0,
#         angularDamping=0.0,
#         jointDamping=JOINT_DAMPING,
#     )
#     p.changeDynamics(robot_id, link_idx, maxJointVelocity=200)
#
# Set all seeds to ensure reproducibility
random.seed(0)
np.random.seed(1)
torch.manual_seed(0)

# Load configuration
abs_config_dir = os.path.abspath("../conf")
with initialize_config_dir(config_dir=abs_config_dir):
    # compose from config.yaml, this composes a bunch of defaults in:
    cfg = hydra_compose(config_name="torch_robot_model_gt_panda.yaml")

num_joints = p.getNumJoints(robot_id)
for i in range(num_joints):
    print(p.getJointInfo(robot_id, i))

test_angles = [0.1]*7
test_velocities = [0.1]*7
test_accelerations = [0.1]*7
gt_robot_model = DifferentiableRobotModel(**cfg.model)

for i in range(7):
    p.resetJointState(
        bodyUniqueId=robot_id,
        jointIndex=i,
        targetValue=test_angles[i],
        targetVelocity=test_velocities[i],
    )

bullet_torques = p.calculateInverseDynamics(
    robot_id, test_angles, test_velocities, test_accelerations
)

model_torques = gt_robot_model.compute_inverse_dynamics(
    torch.Tensor(test_angles).reshape(1, dof),
    torch.Tensor(test_velocities).reshape(1, dof),
    torch.Tensor(test_accelerations).reshape(1, dof),
    include_gravity=True,
)

# ee_id = 7
# model_jac_lin, model_jac_ang = gt_robot_model.compute_endeffector_jacobian(
#     torch.Tensor(test_angles).reshape(1, 7), "panda_virtual_ee_link"
# )
#
# bullet_jac_lin, bullet_jac_ang = p.calculateJacobian(
#     bodyUniqueId=robot_id,
#     linkIndex=ee_id,
#     localPosition=[0, 0, 0],
#     objPositions=test_angles,
#     objVelocities=test_velocities,
#     objAccelerations=[0] * 7,
# )
#
# print(model_jac_lin)
# print()
# print(np.asarray(bullet_jac_lin))

print('hello')

