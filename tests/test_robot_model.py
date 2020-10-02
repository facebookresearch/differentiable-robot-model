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

rel_urdf_path = "kuka_iiwa/urdf/iiwa7.urdf"
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

    for i in range(7):
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

    test_case = {
        "joint_angles": [0, 0.9, 0, 0, -1, 0, 0],
        "joint_velocities": [0.0] * 7,
        "joint_accelerations": [0.0] * 7,
    }
    test_cases.append(test_case)

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
        cfg = hydra_compose(config_name="torch_robot_model_gt.yaml")
    robot_model = DifferentiableRobotModel(**cfg.model)
    test_case = sample_test_case(robot_model)

    return {"robot_model": robot_model, "test_case": test_case}


class TestRobotModel:
    def test_ee_jacobian(self, request, setup_dict):
        robot_model = setup_dict["robot_model"]
        test_case = setup_dict["test_case"]
        ee_id = 7

        test_angles, test_velocities = (
            test_case["joint_angles"],
            test_case["joint_velocities"],
        )

        model_jac_lin, model_jac_ang = robot_model.compute_endeffector_jacobian(
            torch.Tensor(test_angles).reshape(1, 7), "iiwa_link_ee"
        )

        bullet_jac_lin, bullet_jac_ang = p.calculateJacobian(
            bodyUniqueId=robot_id,
            linkIndex=ee_id,
            localPosition=[0, 0, 0],
            objPositions=test_angles,
            objVelocities=test_velocities,
            objAccelerations=[0] * 7,
        )
        assert np.allclose(
            model_jac_lin.detach().numpy(), np.asarray(bullet_jac_lin), atol=1e-7
        )
        assert np.allclose(
            model_jac_ang.detach().numpy(), np.asarray(bullet_jac_ang), atol=1e-7
        )

    def test_mass_computation(self, request, setup_dict):
        robot_model = setup_dict["robot_model"]
        test_case = setup_dict["test_case"]
        test_angles, test_velocities = (
            test_case["joint_angles"],
            test_case["joint_velocities"],
        )

        for i in range(7):
            p.resetJointState(
                bodyUniqueId=robot_id,
                jointIndex=i,
                targetValue=test_angles[i],
                targetVelocity=test_velocities[i],
            )

        bullet_mass = np.array(p.calculateMassMatrix(robot_id, test_angles))
        inertia_mat = robot_model.compute_lagrangian_inertia_matrix(
            torch.Tensor(test_angles).reshape(1, 7)
        )

        assert np.allclose(
            inertia_mat.detach().squeeze().numpy(), bullet_mass, atol=1e-7
        )

    def test_end_effector_state(self, request, setup_dict):

        robot_model = setup_dict["robot_model"]
        test_case = setup_dict["test_case"]
        ee_id = 7
        test_angles, test_velocities = (
            test_case["joint_angles"],
            test_case["joint_velocities"],
        )

        for i in range(7):
            p.resetJointState(
                bodyUniqueId=robot_id,
                jointIndex=i,
                targetValue=test_angles[i],
                targetVelocity=test_velocities[i],
            )
        bullet_ee_state = p.getLinkState(robot_id, ee_id)

        model_ee_state = robot_model.compute_forward_kinematics(
            torch.Tensor(test_angles).reshape(1, 7), "iiwa_link_ee"
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

    def test_inverse_dynamics(self, request, setup_dict):

        robot_model = setup_dict["robot_model"]
        test_case = setup_dict["test_case"]

        test_angles, test_velocities = (
            test_case["joint_angles"],
            test_case["joint_velocities"],
        )
        test_accelerations = test_case["joint_accelerations"]

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

        model_torques = robot_model.compute_inverse_dynamics(
            torch.Tensor(test_angles).reshape(1, 7),
            torch.Tensor(test_velocities).reshape(1, 7),
            torch.Tensor(test_accelerations).reshape(1, 7),
            include_gravity=True,
        )
        if JOINT_DAMPING != 0.0:
            # if we have non-zero joint damping, we'll have to subtract the damping term from our predicted torques,
            # because pybullet does not include damping/viscous friction in their inverse dynamics call
            damping_const = torch.zeros(1, robot_model._n_dofs)
            qd = torch.Tensor(test_velocities).reshape(1, 7)
            for i in range(robot_model._n_dofs):
                idx = robot_model._controlled_joints[i]
                damping_const[:, i] = robot_model._bodies[idx].get_joint_damping_const()
            damping_term = damping_const.repeat(1, 1) * qd
            model_torques -= damping_term

        assert np.allclose(
            model_torques.detach().squeeze().numpy(),
            np.asarray(bullet_torques),
            atol=1e-7,
        )

    def test_forward_dynamics(self, request, setup_dict):
        robot_model = setup_dict["robot_model"]
        test_case = setup_dict["test_case"]
        test_angles, test_velocities = (
            test_case["joint_angles"],
            test_case["joint_velocities"],
        )
        test_accelerations = test_case["joint_accelerations"]
        dt = 1.0 / 240.0
        n_dofs = 7
        controlled_joints = range(n_dofs)
        # activating torque control
        p.setJointMotorControlArray(
            bodyIndex=robot_id,
            jointIndices=controlled_joints,
            controlMode=p.VELOCITY_CONTROL,
            forces=np.zeros(n_dofs),
        )

        # set simulation to be in state test_angles/test_velocities
        for i in range(7):
            p.resetJointState(
                bodyUniqueId=robot_id,
                jointIndex=i,
                targetValue=test_angles[i],
                targetVelocity=test_velocities[i],
            )

        # let's get the torque that achieves the test_accelerations from the current state
        bullet_tau = np.array(
            p.calculateInverseDynamics(
                robot_id, test_angles, test_velocities, test_accelerations
            )
        )

        p.setJointMotorControlArray(
            bodyIndex=robot_id,
            jointIndices=controlled_joints,
            controlMode=p.TORQUE_CONTROL,
            forces=bullet_tau,
        )

        p.stepSimulation()

        cur_joint_states = p.getJointStates(robot_id, controlled_joints)
        q = [cur_joint_states[i][0] for i in range(n_dofs)]
        qd = [cur_joint_states[i][1] for i in range(n_dofs)]

        qdd = (np.array(qd) - np.array(test_velocities)) / dt

        model_qdd = robot_model.compute_forward_dynamics(
            torch.Tensor(test_angles).reshape(1, 7),
            torch.Tensor(test_velocities).reshape(1, 7),
            torch.Tensor(bullet_tau).reshape(1, 7),
            include_gravity=True,
        )

        model_qdd = np.asarray(model_qdd.detach().squeeze())
        if JOINT_DAMPING == 0.0:
            # we can only test this if joint damping is zero,
            # if it is non-zero the pybullet forward dynamics and inverse dynamics call will not be exactly the
            # "inverse" of each other
            assert np.allclose(
                model_qdd, np.asarray(test_accelerations), atol=1e-7
            )  # if atol = 1e-3 it doesnt pass
        assert np.allclose(model_qdd, qdd, atol=1e-7)  # if atol = 1e-3 it doesnt pass
