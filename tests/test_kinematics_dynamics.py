# Copyright (c) Facebook, Inc. and its affiliates.
import os
import random
from dataclasses import dataclass

import torch
import numpy as np
import pytest

import pybullet as p
import diff_robot_data
from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel,
)

torch.set_default_tensor_type(torch.FloatTensor)

# (rel_urdf_path, test_link_list)
test_data = [
    # Fetch arm
    (
        "fetch_description/urdf/fetch_arm_no_gripper_small_damping.urdf",
        [(7, "virtual_ee_link")],
    ),
    # Toy
    ("2link_robot.urdf", [(2, "endEffector")]),
    # Kuka iiwa
    ("kuka_iiwa/urdf/iiwa7.urdf", [(7, "iiwa_link_ee")]),
    # Franka_panda
    ("panda_description/urdf/panda_no_gripper.urdf", [(7, "panda_virtual_ee_link")]),
    # Allegro hand
    (
        "allegro/urdf/allegro_hand_description_left_small_damping.urdf",
        [
            (4, "link_11.0_tip"),
            (9, "link_7.0_tip"),
            (14, "link_3.0_tip"),
            (19, "link_15.0_tip"),
        ],
    ),
    # Trifinger Edu
    (
        "trifinger_edu_description/trifinger_edu.urdf",
        [
            (5, "finger_tip_link_0"),
            (10, "finger_tip_link_120"),
            (15, "finger_tip_link_240"),
        ],
    ),
    # Kinova
    ("kinova_description/urdf/jaco_clean.urdf", [(8, "j2n6s300_link_ee")]),
]

# Note: test batch sizes that coincide with other data dims to ensure errors don't occur during reshaping ops
batch_shapes = [
    tuple(),
    (1,),
    (3,),  # same dim size as so3 variables
    (6,),  # same dim size as se3 variables
    (7,),  # same dim size as 7dof robot states
]

################
# Dataclasses
################


@dataclass
class MetaTestInfo:
    urdf_path: str
    link_list: list
    zero_vel: bool
    zero_acc: bool


@dataclass
class PybulletInstance:
    pc_id: int
    robot_id: int
    num_joints: int


@dataclass
class SampledTestCase:
    joint_pos: list
    joint_vel: list
    joint_acc: list


################
# Arrange
################


@pytest.fixture(params=test_data)
def test_info(request):
    rel_urdf_path = request.param[0]
    robot_description_folder = diff_robot_data.__path__[0]
    urdf_path = os.path.join(robot_description_folder, rel_urdf_path)

    return MetaTestInfo(
        urdf_path=urdf_path, link_list=request.param[1], zero_vel=False, zero_acc=False
    )


@pytest.fixture(params=batch_shapes)
def batch_shape(request):
    return request.param


# Setup pybullet
@pytest.fixture
def sim(test_info):
    pc_id = p.connect(p.DIRECT)

    robot_id = p.loadURDF(
        test_info.urdf_path,
        basePosition=[0, 0, 0],
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
        physicsClientId=pc_id,
    )

    p.setGravity(0, 0, -9.81, physicsClientId=pc_id)
    num_joints = p.getNumJoints(robot_id, physicsClientId=pc_id)

    return PybulletInstance(
        pc_id=pc_id,
        robot_id=robot_id,
        num_joints=num_joints,
    )


# Setup differentiable robot model
@pytest.fixture
def robot_model(test_info):
    return DifferentiableRobotModel(test_info.urdf_path)


# Setup test
@pytest.fixture()
def setup_dict(request, test_info, sim, robot_model, batch_shape):
    # Get num dofs
    num_dofs = len(robot_model.get_joint_limits())

    # Update pybullet joint damping
    for link_idx in range(sim.num_joints):
        joint_damping = robot_model._bodies[link_idx + 1].get_joint_damping_const()
        p.changeDynamics(
            sim.robot_id,
            link_idx,
            linearDamping=0.0,
            angularDamping=0.0,
            jointDamping=joint_damping,
            physicsClientId=sim.pc_id,
        )
        p.changeDynamics(
            sim.robot_id, link_idx, maxJointVelocity=200, physicsClientId=sim.pc_id
        )

    # Set all seeds to ensure reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Sample test cases
    limits_per_joint = robot_model.get_joint_limits()
    joint_lower_bounds = np.asarray([joint["lower"] for joint in limits_per_joint])
    joint_upper_bounds = np.asarray([joint["upper"] for joint in limits_per_joint])
    joint_velocity_limits = np.asarray(
        [0.01 * joint["velocity"] for joint in limits_per_joint]
    )
    # NOTE: sample low velocities since PyBullet inhibits unknown clipping for large damping forces
    #       (encountered with the allegro hand urdf)

    sample_shape = batch_shape + (num_dofs,)

    joint_angles = np.random.uniform(
        low=joint_lower_bounds, high=joint_upper_bounds, size=sample_shape
    )
    joint_velocities = np.random.uniform(
        low=-joint_velocity_limits, high=joint_velocity_limits, size=sample_shape
    )
    if test_info.zero_acc:
        joint_accelerations = np.zeros(sample_shape)
    else:
        joint_accelerations = 10.0 * np.random.uniform(
            low=-joint_velocity_limits, high=joint_velocity_limits, size=sample_shape
        )

    return {
        "robot_model": robot_model,
        "sim": sim,
        "num_dofs": num_dofs,
        "test_case": SampledTestCase(
            joint_pos=joint_angles.tolist(),
            joint_vel=joint_velocities.tolist(),
            joint_acc=joint_accelerations.tolist(),
        ),
    }


################
# Act
################

# Helper functions
def extract_setup_dict(setup_dict):
    robot_model = setup_dict["robot_model"]
    sim = setup_dict["sim"]
    num_dofs = setup_dict["num_dofs"]
    test_case = setup_dict["test_case"]

    return robot_model, sim, num_dofs, test_case


def set_pybullet_state(sim, robot_model, num_dofs, angles, velocities):
    for i in range(num_dofs):
        j_idx = (
            robot_model._controlled_joints[i] - 1
        )  # pybullet link idx starts at -1 for base link
        p.resetJointState(
            bodyUniqueId=sim.robot_id,
            jointIndex=j_idx,
            targetValue=angles[i],
            targetVelocity=velocities[i],
            physicsClientId=sim.pc_id,
        )


# Main test class
class TestRobotModel:
    @pytest.mark.parametrize("recursive", [True, False])
    def test_end_effector_state(self, request, setup_dict, test_info, recursive):
        robot_model, sim, num_dofs, test_case = extract_setup_dict(setup_dict)

        for ee_link_idx, ee_link_name in test_info.link_list:
            # Bullet sim
            def get_bullet_ee_state(joint_pos, joint_vel):
                set_pybullet_state(sim, robot_model, num_dofs, joint_pos, joint_vel)
                return p.getLinkState(
                    sim.robot_id, ee_link_idx, physicsClientId=sim.pc_id
                )

            if type(test_case.joint_pos[0]) is list:
                ret = [
                    get_bullet_ee_state(joint_pos, joint_vel)
                    for joint_pos, joint_vel in zip(
                        test_case.joint_pos, test_case.joint_vel
                    )
                ]
                bullet_ee_pos = [r[0] for r in ret]
                bullet_ee_quat = [r[1] for r in ret]
            else:
                ret = get_bullet_ee_state(test_case.joint_pos, test_case.joint_vel)
                bullet_ee_pos = ret[0]
                bullet_ee_quat = ret[1]

            # Differentiable model
            model_ee_pos, model_ee_quat = robot_model.compute_forward_kinematics(
                torch.Tensor(test_case.joint_pos), ee_link_name, recursive=recursive
            )

            # Compare
            assert np.allclose(
                model_ee_pos.detach().numpy(),
                np.asarray(bullet_ee_pos),
                atol=1e-6,
            )
            assert np.allclose(
                model_ee_quat.detach().numpy(),
                np.asarray(bullet_ee_quat),
                atol=1e-6,
            )

    def test_ee_jacobian(self, request, setup_dict, test_info):
        robot_model, sim, num_dofs, test_case = extract_setup_dict(setup_dict)

        for ee_link_idx, ee_link_name in test_info.link_list:
            # Bullet sim
            def get_bullet_jacobian(joint_pos, joint_vel):
                set_pybullet_state(sim, robot_model, num_dofs, joint_pos, joint_vel)

                return p.calculateJacobian(
                    bodyUniqueId=sim.robot_id,
                    linkIndex=ee_link_idx,
                    localPosition=[0, 0, 0],
                    objPositions=joint_pos,
                    objVelocities=joint_vel,
                    objAccelerations=[0] * num_dofs,
                    physicsClientId=sim.pc_id,
                )

            if type(test_case.joint_pos[0]) is list:
                ret = [
                    get_bullet_jacobian(joint_pos, joint_vel)
                    for joint_pos, joint_vel in zip(
                        test_case.joint_pos, test_case.joint_vel
                    )
                ]
                bullet_jac_lin = [r[0] for r in ret]
                bullet_jac_ang = [r[1] for r in ret]
            else:
                bullet_jac_lin, bullet_jac_ang = get_bullet_jacobian(
                    test_case.joint_pos, test_case.joint_vel
                )

            # Differentiable model
            model_jac_lin, model_jac_ang = robot_model.compute_endeffector_jacobian(
                torch.Tensor(test_case.joint_pos), ee_link_name
            )

            # Compare
            assert np.allclose(
                model_jac_lin.detach().numpy(),
                np.asarray(bullet_jac_lin),
                atol=1e-6,
            )
            assert np.allclose(
                model_jac_ang.detach().numpy(),
                np.asarray(bullet_jac_ang),
                atol=1e-6,
            )

    @pytest.mark.parametrize("use_damping", [True, False])
    def test_inverse_dynamics(self, request, setup_dict, use_damping):
        robot_model, sim, num_dofs, test_case = extract_setup_dict(setup_dict)

        # Bullet sim
        def get_bullet_invdyn(joint_pos, joint_vel, joint_acc):
            set_pybullet_state(sim, robot_model, num_dofs, joint_pos, joint_vel)
            return p.calculateInverseDynamics(
                sim.robot_id,
                joint_pos,
                joint_vel,
                joint_acc,
                physicsClientId=sim.pc_id,
            )

        if type(test_case.joint_pos[0]) is list:
            bullet_torques = [
                get_bullet_invdyn(joint_pos, joint_vel, joint_acc)
                for joint_pos, joint_vel, joint_acc in zip(
                    test_case.joint_pos, test_case.joint_vel, test_case.joint_acc
                )
            ]
        else:
            bullet_torques = get_bullet_invdyn(
                test_case.joint_pos, test_case.joint_vel, test_case.joint_acc
            )

        # Differentiable model
        model_torques = robot_model.compute_inverse_dynamics(
            torch.Tensor(test_case.joint_pos),
            torch.Tensor(test_case.joint_vel),
            torch.Tensor(test_case.joint_acc),
            include_gravity=True,
            use_damping=use_damping,
        )

        if use_damping:
            # if we have non-zero joint damping, we'll have to subtract the damping term from our predicted torques,
            # because pybullet does not include damping/viscous friction in their inverse dynamics call
            damping_const = torch.zeros(num_dofs)
            qd = torch.Tensor(test_case.joint_vel)
            for i in range(robot_model._n_dofs):
                idx = robot_model._controlled_joints[i]
                damping_const[i] = robot_model._bodies[idx].get_joint_damping_const()
            damping_term = damping_const * qd
            model_torques -= damping_term

        # Compare
        assert np.allclose(
            model_torques.detach().numpy(),
            np.asarray(bullet_torques),
            atol=1e-5,
        )

    def test_mass_computation(self, request, setup_dict):
        robot_model, sim, num_dofs, test_case = extract_setup_dict(setup_dict)

        # Bullet sim
        def get_bullet_mass(joint_pos, joint_vel):
            set_pybullet_state(sim, robot_model, num_dofs, joint_pos, joint_vel)
            return np.array(
                p.calculateMassMatrix(
                    sim.robot_id, joint_pos, physicsClientId=sim.pc_id
                )
            )

        if type(test_case.joint_pos[0]) is list:
            bullet_mass = [
                get_bullet_mass(joint_pos, joint_vel)
                for joint_pos, joint_vel in zip(
                    test_case.joint_pos, test_case.joint_vel
                )
            ]
        else:
            bullet_mass = get_bullet_mass(test_case.joint_pos, test_case.joint_vel)

        # Differentiable model
        inertia_mat = robot_model.compute_lagrangian_inertia_matrix(
            torch.Tensor(test_case.joint_pos)
        )

        # Compare (Dynamics scales differ a lot between different robots so rtol is used)
        assert np.allclose(
            inertia_mat.detach().numpy(), bullet_mass, rtol=1e-3, atol=1e-5
        )

    @pytest.mark.parametrize("use_damping", [True, False])
    def test_forward_dynamics(self, request, setup_dict, use_damping):
        robot_model, sim, num_dofs, test_case = extract_setup_dict(setup_dict)

        # Bullet sim
        dt = 1.0 / 240.0
        controlled_joints = [i - 1 for i in robot_model._controlled_joints]

        def get_bullet_fd(joint_pos, joint_vel, joint_acc):
            # Update joint damping
            for link_idx in range(sim.num_joints):
                if use_damping:
                    joint_damping = robot_model._bodies[
                        link_idx + 1
                    ].get_joint_damping_const()
                else:
                    joint_damping = 0.0

                p.changeDynamics(
                    sim.robot_id,
                    link_idx,
                    linearDamping=0.0,
                    angularDamping=0.0,
                    jointDamping=joint_damping,
                    physicsClientId=sim.pc_id,
                )

            p.setJointMotorControlArray(  # activating torque control
                bodyIndex=sim.robot_id,
                jointIndices=controlled_joints,
                controlMode=p.VELOCITY_CONTROL,
                forces=np.zeros(num_dofs),
                physicsClientId=sim.pc_id,
            )

            set_pybullet_state(sim, robot_model, num_dofs, joint_pos, joint_vel)

            bullet_tau = np.array(  # torque that achieves joint_acc from current state
                p.calculateInverseDynamics(
                    sim.robot_id,
                    joint_pos,
                    joint_vel,
                    joint_acc,
                    physicsClientId=sim.pc_id,
                )
            )

            p.setJointMotorControlArray(
                bodyIndex=sim.robot_id,
                jointIndices=controlled_joints,
                controlMode=p.TORQUE_CONTROL,
                forces=bullet_tau,
                physicsClientId=sim.pc_id,
            )

            p.stepSimulation(physicsClientId=sim.pc_id)

            cur_joint_states = p.getJointStates(
                sim.robot_id, controlled_joints, physicsClientId=sim.pc_id
            )
            q = [cur_joint_states[i][0] for i in range(num_dofs)]
            qd = [cur_joint_states[i][1] for i in range(num_dofs)]

            qdd = (np.array(qd) - np.array(joint_vel)) / dt

            return bullet_tau, qdd

        if type(test_case.joint_pos[0]) is list:
            ret = [
                get_bullet_fd(joint_pos, joint_vel, joint_acc)
                for joint_pos, joint_vel, joint_acc in zip(
                    test_case.joint_pos, test_case.joint_vel, test_case.joint_acc
                )
            ]
            bullet_tau = [r[0] for r in ret]
            qdd = [r[1] for r in ret]
        else:
            bullet_tau, qdd = get_bullet_fd(
                test_case.joint_pos, test_case.joint_vel, test_case.joint_acc
            )

        # Differentiable model
        model_qdd = robot_model.compute_forward_dynamics(
            torch.Tensor(test_case.joint_pos),
            torch.Tensor(test_case.joint_vel),
            torch.Tensor(bullet_tau),
            include_gravity=True,
            use_damping=use_damping,
        )

        # Compare (Dynamics scales differ a lot between different robots so rtol is used)
        model_qdd = np.asarray(model_qdd.detach())
        assert np.allclose(model_qdd, qdd, rtol=1e-2, atol=1e-3)

        if not use_damping:
            # we can only test this if joint damping is zero,
            # if it is non-zero the pybullet forward dynamics and inverse dynamics call will not be exactly the
            # "inverse" of each other
            assert np.allclose(
                model_qdd, np.asarray(test_case.joint_acc), rtol=1e-2, atol=1e-3
            )
