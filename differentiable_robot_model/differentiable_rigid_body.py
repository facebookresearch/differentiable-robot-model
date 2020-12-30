# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from . import utils
from .coordinate_transform import (
    CoordinateTransform,
    z_rot,
    y_rot,
    x_rot,
)

import hydra


class DifferentiableRigidBody(torch.nn.Module):
    """
    Differentiable Representation of a link
    """

    def __init__(self, rigid_body_params, device="cpu"):

        super().__init__()

        self._device = device
        self.joint_id = rigid_body_params["joint_id"]
        self.name = rigid_body_params["link_name"]

        # dynamics parameters
        self.mass = rigid_body_params["mass"]
        self.com = rigid_body_params["com"]
        self.inertia_mat = rigid_body_params["inertia_mat"]
        self.joint_damping = rigid_body_params["joint_damping"]

        # kinematics parameters
        self.trans = rigid_body_params["trans"]
        self.rot_angles = rigid_body_params["rot_angles"]
        self.joint_limits = rigid_body_params["joint_limits"]

        # local joint axis (w.r.t. joint coordinate frame):
        self.joint_axis = rigid_body_params["joint_axis"]

        self.joint_pose = CoordinateTransform()
        self.joint_pose.set_translation(torch.reshape(self.trans, (1, 3)))

        # local velocities and accelerations (w.r.t. joint coordinate frame):
        # in spatial vector terminology: linear velocity v
        self.joint_lin_vel = torch.zeros((1, 3))  # .to(self._device)
        # in spatial vector terminology: angular velocity w
        self.joint_ang_vel = torch.zeros((1, 3))  # .to(self._device)
        # in spatial vector terminology: linear acceleration vd
        self.joint_lin_acc = torch.zeros((1, 3))  # .to(self._device)
        # in spatial vector terminology: angular acceleration wd
        self.joint_ang_acc = torch.zeros((1, 3))  # .to(self._device)

        self.update_joint_state(torch.zeros(1, 1), torch.zeros(1, 1))
        self.update_joint_acc(torch.zeros(1, 1))

        self.pose = CoordinateTransform()

        # I have different vectors for angular/linear motion/force, but they usually always appear as a pair
        # meaning we usually always compute both angular/linear components.
        # Maybe worthwile thinking of a structure for this - in math notation we would use the notion of spatial vectors
        # drake uses some form of spatial vector implementation
        self.lin_vel = torch.zeros((1, 3)).to(self._device)
        self.ang_vel = torch.zeros((1, 3)).to(self._device)
        self.lin_acc = torch.zeros((1, 3)).to(self._device)
        self.ang_acc = torch.zeros((1, 3)).to(self._device)

        # in spatial vector terminology this is the "linear force f"
        self.lin_force = torch.zeros((1, 3)).to(self._device)
        # in spatial vector terminology this is the "couple n"
        self.ang_force = torch.zeros((1, 3)).to(self._device)

        return

    def update_joint_state(self, q, qd):
        batch_size = q.shape[0]

        self.joint_ang_vel = qd @ self.joint_axis

        roll = self.rot_angles[0]
        pitch = self.rot_angles[1]
        yaw = self.rot_angles[2]

        fixed_rotation = (z_rot(yaw) @ y_rot(pitch)) @ x_rot(roll)

        # when we update the joint angle, we also need to update the transformation
        self.joint_pose.set_translation(torch.reshape(self.trans, (1, 3)))
        if self.joint_axis[0, 0] == 1:
            rot = x_rot(q)
        elif self.joint_axis[0, 1] == 1:
            rot = y_rot(q)
        else:
            rot = z_rot(q)

        self.joint_pose.set_rotation(fixed_rotation.repeat(batch_size, 1, 1) @ rot)
        return

    def update_joint_acc(self, qdd):
        # local z axis (w.r.t. joint coordinate frame):
        self.joint_ang_acc = qdd @ self.joint_axis
        return

    def multiply_inertia_with_motion_vec(self, lin, ang):

        mass, com, inertia_mat = self._get_dynamics_parameters_values()

        mcom = com * mass
        com_skew_symm_mat = utils.vector3_to_skew_symm_matrix(com)
        inertia = inertia_mat + mass * (
            com_skew_symm_mat @ com_skew_symm_mat.transpose(-2, -1)
        )

        batch_size = lin.shape[0]

        new_lin_force = mass * lin - utils.cross_product(
            mcom.repeat(batch_size, 1), ang
        )
        new_ang_force = (inertia.repeat(batch_size, 1, 1) @ ang.unsqueeze(2)).squeeze(
            2
        ) + utils.cross_product(mcom.repeat(batch_size, 1), lin)

        return new_lin_force, new_ang_force

    def _get_dynamics_parameters_values(self):
        return self.mass, self.com, self.inertia_mat

    def get_joint_limits(self):
        return self.joint_limits

    def get_joint_damping_const(self):
        return self.joint_damping


class LearnableRigidBody(DifferentiableRigidBody):
    r"""

    Learnable Representation of a link

    """

    def __init__(self, learnable_rigid_body_config, gt_rigid_body_params, device="cpu"):

        super().__init__(rigid_body_params=gt_rigid_body_params, device=device)

        # we overwrite dynamics parameters
        if "mass" in learnable_rigid_body_config.learnable_dynamics_params:
            self.mass_fn = hydra.utils.instantiate(
                learnable_rigid_body_config.mass_parametrization, device=device
            )
        else:
            self.mass_fn = lambda: self.mass

        if "com" in learnable_rigid_body_config.learnable_dynamics_params:
            self.com_fn = hydra.utils.instantiate(
                learnable_rigid_body_config.com_parametrization, device=device
            )
        else:
            self.com_fn = lambda: self.com

        if "inertia_mat" in learnable_rigid_body_config.learnable_dynamics_params:
            self.inertia_mat_fn = hydra.utils.instantiate(learnable_rigid_body_config.inertia_parametrization)
        else:
            self.inertia_mat_fn = lambda: self.inertia_mat

        self.joint_damping = gt_rigid_body_params["joint_damping"]

        # kinematics parameters
        if "trans" in learnable_rigid_body_config.learnable_kinematics_params:
            self.trans = torch.nn.Parameter(
                torch.rand_like(gt_rigid_body_params["trans"])
            )
            self.joint_pose.set_translation(torch.reshape(self.trans, (1, 3)))

        if "rot_angles" in learnable_rigid_body_config.learnable_kinematics_params:
            self.rot_angles = torch.nn.Parameter(gt_rigid_body_params["rot_angles"])

        return

    def _get_dynamics_parameters_values(self):
        return self.mass_fn(), self.com_fn(), self.inertia_mat_fn()
