# Copyright (c) Facebook, Inc. and its affiliates.
import os
import random
import torch
import numpy as np
import pytest

import pybullet as p
import diff_robot_data
from differentiable_robot_model.differentiable_robot_model import DifferentiableRobotModel, LearnableRigidBodyConfig

robot_description_folder = diff_robot_data.__path__[0]

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)
torch.set_default_tensor_type(torch.DoubleTensor)

rel_urdf_path = "kuka_iiwa/urdf/iiwa7.urdf"
urdf_path = os.path.join(robot_description_folder, rel_urdf_path)

# Setup pybullet client
pc_id = p.connect(p.DIRECT)

robot_id = p.loadURDF(
    urdf_path,
    basePosition=[0, 0, 0],
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE,
    physicsClientId=pc_id
)

p.setGravity(0, 0, -9.81, physicsClientId=pc_id)


def sample_test_case(robot_model, zero_vel=False, zero_acc=False):
    limits_per_joint = robot_model.get_joint_limits()
    joint_lower_bounds = [joint["lower"] for joint in limits_per_joint]
    joint_upper_bounds = [joint["upper"] for joint in limits_per_joint]
    joint_velocity_limits = [joint["velocity"] for joint in limits_per_joint]
    joint_angles = []
    joint_velocities = []
    joint_accelerations = []

    for i in range(len(limits_per_joint)):
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
    robot_model = DifferentiableRobotModel(urdf_path, LearnableRigidBodyConfig(), "differentiable_allegro_hand")
    test_case = sample_test_case(robot_model)

    # Update pybullet joint damping
    NUM_JOINTS = p.getNumJoints(robot_id)
    for link_idx in range(NUM_JOINTS):
        joint_damping = robot_model._bodies[link_idx+1].joint_damping
        p.changeDynamics(
            robot_id,
            link_idx,
            linearDamping=0.0,
            angularDamping=0.0,
            jointDamping=joint_damping,
            physicsClientId=pc_id
        )
        p.changeDynamics(robot_id, link_idx, maxJointVelocity=200, physicsClientId=pc_id)


    return {
        "robot_model": robot_model, 
        "test_case": test_case,
        "num_dofs": len(robot_model.get_joint_limits()),
    }


@pytest.mark.parametrize("ee_link_idx, ee_link_name", [
    (7, "iiwa_link_ee"), 
])
class TestRobotModel:
    def test_end_effector_state(self, request, setup_dict, ee_link_idx, ee_link_name):
        robot_model = setup_dict["robot_model"]
        test_case = setup_dict["test_case"]
        num_dofs = setup_dict["num_dofs"]

        test_angles, test_velocities = (
            test_case["joint_angles"],
            test_case["joint_velocities"],
        )

        for i in range(num_dofs):
            p.resetJointState(
                bodyUniqueId=robot_id,
                jointIndex=i,
                targetValue=test_angles[i],
                targetVelocity=test_velocities[i],
                physicsClientId=pc_id
            )
        bullet_ee_state = p.getLinkState(robot_id, ee_link_idx, physicsClientId = pc_id)

        model_ee_state = robot_model.compute_forward_kinematics(
            torch.Tensor(test_angles).reshape(1, num_dofs), ee_link_name
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

    def test_ee_jacobian(self, request, setup_dict, ee_link_idx, ee_link_name):
        robot_model = setup_dict["robot_model"]
        test_case = setup_dict["test_case"]
        num_dofs = setup_dict["num_dofs"]

        test_angles, test_velocities = (
            test_case["joint_angles"],
            test_case["joint_velocities"],
        )

        model_jac_lin, model_jac_ang = robot_model.compute_endeffector_jacobian(
            torch.Tensor(test_angles).reshape(1, num_dofs), ee_link_name
        )

        bullet_jac_lin, bullet_jac_ang = p.calculateJacobian(
            bodyUniqueId=robot_id,
            linkIndex=ee_link_idx,
            localPosition=[0, 0, 0],
            objPositions=test_angles,
            objVelocities=test_velocities,
            objAccelerations=[0] * num_dofs,
            physicsClientId=pc_id
        )
        assert np.allclose(
            model_jac_lin.detach().numpy(), np.asarray(bullet_jac_lin), atol=1e-7
        )
        assert np.allclose(
            model_jac_ang.detach().numpy(), np.asarray(bullet_jac_ang), atol=1e-7
        )

    def test_inverse_dynamics(self, request, setup_dict, ee_link_idx, ee_link_name):
        robot_model = setup_dict["robot_model"]
        test_case = setup_dict["test_case"]
        num_dofs = setup_dict["num_dofs"]

        test_angles, test_velocities = (
            test_case["joint_angles"],
            test_case["joint_velocities"],
        )
        test_accelerations = test_case["joint_accelerations"]

        for i in range(num_dofs):
            p.resetJointState(
                bodyUniqueId=robot_id,
                jointIndex=i,
                targetValue=test_angles[i],
                targetVelocity=test_velocities[i],
                physicsClientId=pc_id
            )

        bullet_torques = p.calculateInverseDynamics(
            robot_id, test_angles, test_velocities, test_accelerations, physicsClientId = pc_id
        )

        model_torques = robot_model.compute_inverse_dynamics(
            torch.Tensor(test_angles).reshape(1, num_dofs),
            torch.Tensor(test_velocities).reshape(1, num_dofs),
            torch.Tensor(test_accelerations).reshape(1, num_dofs),
            include_gravity=True,
            use_damping=False
        )

        assert np.allclose(
            model_torques.detach().squeeze().numpy(),
            np.asarray(bullet_torques),
            atol=1e-7,
        )

    def test_mass_computation(self, request, setup_dict, ee_link_idx, ee_link_name):
        robot_model = setup_dict["robot_model"]
        test_case = setup_dict["test_case"]
        num_dofs = setup_dict["num_dofs"]

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
                physicsClientId=pc_id
            )
        bullet_mass = np.array(p.calculateMassMatrix(robot_id, test_angles,physicsClientId = pc_id))
        inertia_mat = robot_model.compute_lagrangian_inertia_matrix(
            torch.Tensor(test_angles).reshape(1, num_dofs)
        )

        assert np.allclose(
            inertia_mat.detach().squeeze().numpy(), bullet_mass, atol=1e-7
        )

    def test_forward_dynamics(self, request, setup_dict, ee_link_idx, ee_link_name):
        robot_model = setup_dict["robot_model"]
        test_case = setup_dict["test_case"]
        num_dofs = setup_dict["num_dofs"]

        test_angles, test_velocities = (
            test_case["joint_angles"],
            test_case["joint_velocities"],
        )
        test_accelerations = test_case["joint_accelerations"]
        dt = 1.0 / 240.0
        controlled_joints = range(num_dofs)
        # activating torque control
        p.setJointMotorControlArray(
            bodyIndex=robot_id,
            jointIndices=controlled_joints,
            controlMode=p.VELOCITY_CONTROL,
            forces=np.zeros(num_dofs),
            physicsClientId=pc_id
        )

        # set simulation to be in state test_angles/test_velocities
        for i in range(num_dofs):
            p.resetJointState(
                bodyUniqueId=robot_id,
                jointIndex=i,
                targetValue=test_angles[i],
                targetVelocity=test_velocities[i],
                physicsClientId=pc_id
            )

        # let's get the torque that achieves the test_accelerations from the current state
        bullet_tau = np.array(
            p.calculateInverseDynamics(
                robot_id, test_angles, test_velocities, test_accelerations, physicsClientId=pc_id
            )
        )

        p.setJointMotorControlArray(
            bodyIndex=robot_id,
            jointIndices=controlled_joints,
            controlMode=p.TORQUE_CONTROL,
            forces=bullet_tau,
            physicsClientId=pc_id
        )

        p.stepSimulation(physicsClientId = pc_id)

        cur_joint_states = p.getJointStates(robot_id, controlled_joints)
        q = [cur_joint_states[i][0] for i in range(num_dofs)]
        qd = [cur_joint_states[i][1] for i in range(num_dofs)]

        qdd = (np.array(qd) - np.array(test_velocities)) / dt

        model_qdd = robot_model.compute_forward_dynamics(
            torch.Tensor(test_angles).reshape(1, num_dofs),
            torch.Tensor(test_velocities).reshape(1, num_dofs),
            torch.Tensor(bullet_tau).reshape(1, num_dofs),
            include_gravity=True,
            use_damping=True
        )

        model_qdd = np.asarray(model_qdd.detach().squeeze())
        assert np.allclose(model_qdd, qdd, atol=1e-7)  # if atol = 1e-3 it doesnt pass
