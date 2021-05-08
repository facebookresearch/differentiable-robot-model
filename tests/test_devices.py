# Copyright (c) Facebook, Inc. and its affiliates.
import os

import torch
import pytest

import diff_robot_data
from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel,
)

rel_urdf_path = "2link_robot.urdf"
robot_description_folder = diff_robot_data.__path__[0]
urdf_path = os.path.join(robot_description_folder, rel_urdf_path)


@pytest.fixture(params=["cuda", "cpu"])
def robot_model(request):
    device = request.param
    if not torch.cuda.is_available():
        device = "cpu"
    return DifferentiableRobotModel(urdf_path, device=device)


@pytest.mark.parametrize(
    "default_tensor_type", [torch.cuda.FloatTensor, torch.FloatTensor]
)
def test_robot_model(robot_model, default_tensor_type):
    # Set default tensor type
    if not torch.cuda.is_available():
        default_tensor_type = torch.FloatTensor

    torch.set_default_tensor_type(default_tensor_type)

    # Method arguments
    n_dofs = robot_model._n_dofs
    rand_n_dofs = torch.rand([1, n_dofs], device=robot_model._device)
    ee_name = "endEffector"

    # Run robot model methods
    robot_model.update_kinematic_state(rand_n_dofs, rand_n_dofs)
    robot_model.compute_forward_kinematics(rand_n_dofs, ee_name)
    robot_model.compute_inverse_dynamics(rand_n_dofs, rand_n_dofs, rand_n_dofs)
    robot_model.compute_non_linear_effects(rand_n_dofs, rand_n_dofs)
    robot_model.compute_lagrangian_inertia_matrix(rand_n_dofs)
    robot_model.compute_forward_dynamics(rand_n_dofs, rand_n_dofs, rand_n_dofs)
    robot_model.compute_endeffector_jacobian(rand_n_dofs, ee_name)
