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

rel_urdf_path = "fetch_description/urdf/fetch_arm.urdf"
urdf_path = os.path.join(robot_description_folder, rel_urdf_path)

pc_id = p.connect(p.DIRECT)
robot_id = p.loadURDF(
    urdf_path,
    basePosition=[0, 0, 0],
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE,
)

p.setGravity(0, 0, -9.81)
JOINT_DAMPING = 0.5

# need to be careful with joint damping to zero, because in pybullet the forward dynamics (used for simulation)
# does use joint damping, but the inverse dynamics call does not use joint damping
for link_idx in range(8):
    p.changeDynamics(
        robot_id,
        link_idx,
        linearDamping=0.0,
        angularDamping=0.0,
        jointDamping=JOINT_DAMPING,
    )
    p.changeDynamics(robot_id, link_idx, maxJointVelocity=200)


def sample_test_case(robot_model, zero_vel=False, zero_acc=False):
    limits_per_joint = robot_model.get_joint_limits()
    joint_lower_bounds = [joint["lower"] for joint in limits_per_joint]
    joint_upper_bounds = [joint["upper"] for joint in limits_per_joint]
    joint_velocity_limits = [joint["velocity"] for joint in limits_per_joint]
    joint_angles = []
    joint_velocities = []
    joint_accelerations = []

    for i in range(robot_model._n_dofs):
        joint_angles.append(
            np.random.uniform(low=joint_lower_bounds[i], high=joint_upper_bounds[i])
        )

        if zero_vel:
            joint_velocities.append(0.0)

        else:
            joint_velocities.append(
                np.random.uniform(
                    low=-joint_velocity_limits[i], high=joint_velocity_limits[i]
                )
            )

        if zero_acc:
            joint_accelerations.append(0.0)
        else:
            joint_accelerations.append(
                np.random.uniform(
                    low=-joint_velocity_limits[i] * 2.0,
                    high=joint_velocity_limits[i] * 2.0,
                )
            )

    return {
        "joint_angles": joint_angles,
        "joint_velocities": joint_velocities,
        "joint_accelerations": joint_accelerations,
    }


def generate_test_cases(setup_dict):
    robot_model = setup_dict["robot_model"]
    num_test_cases = 3
    test_cases = []

    for i in range(num_test_cases):
        test_cases.append(sample_test_case(robot_model, zero_vel=True, zero_acc=True))

    for i in range(num_test_cases):
        test_cases.append(sample_test_case(robot_model, zero_vel=False, zero_acc=True))

    for i in range(num_test_cases):
        test_cases.append(sample_test_case(robot_model, zero_vel=False, zero_acc=False))

    return test_cases


@pytest.fixture
def setup_dict():
    """
    if model is "ground_truth":
        tensorType = 'torch.DoubleTensor'
        torch.set_default_tensor_type(tensorType)
    else:
        tensorType = 'torch.FloatTensor'
        torch.set_default_tensor_type(tensorType)
    """
    # Set all seeds to ensure reproducibility
    random.seed(0)
    np.random.seed(1)
    torch.manual_seed(0)

    # Load configuration
    abs_config_dir = os.path.abspath("conf")
    with initialize_config_dir(config_dir=abs_config_dir):
        # compose from config.yaml, this composes a bunch of defaults in:
        cfg = hydra_compose(config_name="fetch_robot_model.yaml")
    robot_model = DifferentiableRobotModel(**cfg.model)
    test_case = sample_test_case(robot_model)

    return {"robot_model": robot_model, "test_case": test_case}


class TestRobotModel:
    def test_ee_jacobian(self, request, setup_dict):
        robot_model = setup_dict["robot_model"]
        test_case = setup_dict["test_case"]
        n_dofs = robot_model._n_dofs
        ee_id = 10

        test_angles, test_velocities = (
            test_case["joint_angles"],
            test_case["joint_velocities"],
        )

        test_angles[7] = 0.0
        test_velocities[7] = 0.0

        model_jac_lin, model_jac_ang = robot_model.compute_endeffector_jacobian(
            torch.Tensor(test_angles).reshape(1, n_dofs), "virtual_ee_link"
        )

        bullet_jac_lin, bullet_jac_ang = p.calculateJacobian(
            bodyUniqueId=robot_id,
            linkIndex=ee_id,
            localPosition=[0, 0, 0],
            objPositions=test_angles,
            objVelocities=test_velocities,
            objAccelerations=[0] * n_dofs,
        )
        assert np.allclose(
            model_jac_lin.detach().numpy(), np.asarray(bullet_jac_lin), atol=1e-7
        )
        assert np.allclose(
            model_jac_ang.detach().numpy(), np.asarray(bullet_jac_ang), atol=1e-7
        )

    def test_end_effector_state(self, request, setup_dict):

        robot_model = setup_dict["robot_model"]
        test_case = setup_dict["test_case"]
        n_dofs = robot_model._n_dofs
        ee_id = 10
        test_angles, test_velocities = (
            test_case["joint_angles"],
            test_case["joint_velocities"],
        )
        controlled_joints = [0, 1, 2, 3, 4, 5, 6, 8, 9]

        for i, joint_idx in enumerate(controlled_joints):
            p.resetJointState(
                bodyUniqueId=robot_id,
                jointIndex=joint_idx,
                targetValue=test_angles[i],
                targetVelocity=test_velocities[i],
            )
        bullet_ee_state = p.getLinkState(robot_id, ee_id)

        model_ee_state = robot_model.compute_forward_kinematics(
            torch.Tensor(test_angles).reshape(1, 9), "virtual_ee_link"
        )

        assert np.allclose(
            model_ee_state[0].detach().numpy(),
            np.asarray(bullet_ee_state[0]),
            atol=1e-7,
        )
        assert np.allclose(
            model_ee_state[1].detach().numpy(),
            np.asarray(bullet_ee_state[1]),
            atol=1e-7,
        )

