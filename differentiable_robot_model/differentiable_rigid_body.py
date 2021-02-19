# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from .spatial_vector_algebra import (
    CoordinateTransform,
    z_rot,
    y_rot,
    x_rot,
)

from .spatial_vector_algebra import SpatialForceVec, SpatialMotionVec
from .spatial_vector_algebra import DifferentiableSpatialRigidBodyInertia, LearnableSpatialRigidBodyInertia

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
        self.joint_damping = rigid_body_params["joint_damping"]
        self.inertia = DifferentiableSpatialRigidBodyInertia(rigid_body_params)

        # kinematics parameters
        self.trans = rigid_body_params["trans"]
        self.rot_angles = rigid_body_params["rot_angles"]
        self.joint_limits = rigid_body_params["joint_limits"]

        # local joint axis (w.r.t. joint coordinate frame):
        self.joint_axis = rigid_body_params["joint_axis"]

        self.joint_pose = CoordinateTransform()
        self.joint_pose.set_translation(torch.reshape(self.trans, (1, 3)))

        # local velocities and accelerations (w.r.t. joint coordinate frame):
        self.joint_vel = SpatialMotionVec()
        self.joint_acc = SpatialMotionVec()

        self.update_joint_state(torch.zeros(1, 1), torch.zeros(1, 1))
        self.update_joint_acc(torch.zeros(1, 1))

        self.pose = CoordinateTransform()

        self.body_vel = SpatialMotionVec()
        self.body_acc = SpatialMotionVec()

        self.force = SpatialForceVec()

        return

    def update_joint_state(self, q, qd):
        batch_size = q.shape[0]

        joint_ang_vel = qd @ self.joint_axis
        self.joint_vel = SpatialMotionVec(torch.zeros_like(joint_ang_vel), joint_ang_vel)

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
        joint_ang_acc = qdd @ self.joint_axis
        self.joint_acc = SpatialMotionVec(torch.zeros_like(joint_ang_acc), joint_ang_acc)
        return

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

        self.inertia = LearnableSpatialRigidBodyInertia(learnable_rigid_body_config, gt_rigid_body_params)
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

