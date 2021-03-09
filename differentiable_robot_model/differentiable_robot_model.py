# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List, Tuple, Dict, Optional
import os

import torch

from .differentiable_rigid_body import (
    DifferentiableRigidBody,
    LearnableRigidBody,
)
from .spatial_vector_algebra import SpatialMotionVec, SpatialForceVec
from .urdf_utils import URDFRobotModel

import diff_robot_data
robot_description_folder = diff_robot_data.__path__[0]


class DifferentiableRobotModel(torch.nn.Module):
    """
    Differentiable Robot Model
    """

    def __init__(
        self, urdf_path: str, learnable_rigid_body_config=None, name=""
    ):

        super().__init__()

        self.name = name

        self._device = "cpu"

        self._urdf_model = URDFRobotModel(
            urdf_path=urdf_path, device=self._device
        )
        self._bodies = torch.nn.ModuleList()
        self._n_dofs = 0
        self._controlled_joints = []

        # here we're making the joint a part of the rigid body
        # while urdfs model joints and rigid bodies separately
        # joint is at the beginning of a link
        self._name_to_idx_map = dict()

        for (i, link) in enumerate(self._urdf_model.robot.links):

            rigid_body_params = self._urdf_model.get_body_parameters_from_urdf(i, link)

            if (learnable_rigid_body_config is not None) and (link.name in learnable_rigid_body_config.learnable_links):
                body = LearnableRigidBody(
                    learnable_rigid_body_config=learnable_rigid_body_config,
                    gt_rigid_body_params=rigid_body_params,
                    device=self._device,
                )
            else:
                body = DifferentiableRigidBody(
                    rigid_body_params=rigid_body_params, device=self._device
                )

            body.joint_idx = None
            if rigid_body_params["joint_type"] != "fixed":
                body.joint_idx = self._n_dofs
                self._n_dofs += 1
                self._controlled_joints.append(i)

            self._bodies.append(body)
            self._name_to_idx_map[body.name] = i

    def update_kinematic_state(self, q: torch.Tensor, qd: torch.Tensor) -> None:
        r"""

        Updates the kinematic state of the robot
        Args:
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]

        Returns:

        """
        assert q.ndim == 2
        assert qd.ndim == 2
        assert q.shape[1] == self._n_dofs
        assert qd.shape[1] == self._n_dofs

        batch_size = q.shape[0]

        # update the state of the joints
        for i in range(q.shape[1]):
            idx = self._controlled_joints[i]
            self._bodies[idx].update_joint_state(
                q[:, i].unsqueeze(1), qd[:, i].unsqueeze(1)
            )

        # we assume a non-moving base
        parent_body = self._bodies[0]
        parent_body.vel = SpatialMotionVec(torch.zeros((batch_size, 3)), torch.zeros((batch_size, 3)))

        # propagate the new joint state through the kinematic chain to update bodies position/velocities
        for i in range(1, len(self._bodies)):

            body = self._bodies[i]
            parent_name = self._urdf_model.get_name_of_parent_body(body.name)
            # find the joint that has this link as child
            parent_body = self._bodies[self._name_to_idx_map[parent_name]]

            # transformation operator from child link to parent link
            childToParentT = body.joint_pose
            # transformation operator from parent link to child link
            parentToChildT = childToParentT.inverse()

            # the position and orientation of the body in world coordinates, with origin at the joint
            body.pose = parent_body.pose.multiply_transform(childToParentT)

            # we rotate the velocity of the parent's body into the child frame
            new_vel = parent_body.vel.transform(parentToChildT)

            # this body's angular velocity is combination of the velocity experienced at it's parent's link
            # + the velocity created by this body's joint
            body.vel = body.joint_vel.add_motion_vec(new_vel)
        return

    def compute_forward_kinematics(
        self, q: torch.Tensor, link_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""

        Args:
            q: joint angles [batch_size x n_dofs]
            link_name: name of link

        Returns: translation and rotation of the link frame

        """
        assert q.ndim == 2
        qd = torch.zeros_like(q)
        self.update_kinematic_state(q, qd)

        pose = self._bodies[self._name_to_idx_map[link_name]].pose
        pos = pose.translation()
        rot = pose.get_quaternion()
        return pos, rot

    def iterative_newton_euler(self, base_acc: SpatialMotionVec) -> None:
        r"""

        Args:
            base_acc: spatial acceleration of base (for fixed manipulators this is zero)
        """

        body = self._bodies[0]
        body.acc = base_acc

        # forward pass to propagate accelerations from root to end-effector link
        for i in range(1, len(self._bodies)):
            body = self._bodies[i]
            parent_name = self._urdf_model.get_name_of_parent_body(body.name)

            parent_body = self._bodies[self._name_to_idx_map[parent_name]]

            # get the inverse of the current joint pose
            inv_pose = body.joint_pose.inverse()

            # transform spatial acceleration of parent body into this body's frame
            acc_parent_body = parent_body.acc.transform(inv_pose)
            # body velocity cross joint vel
            tmp = body.vel.cross_motion_vec(body.joint_vel)
            body.acc = acc_parent_body.add_motion_vec(body.joint_acc).add_motion_vec(tmp)

        child_body = self._bodies[-1]

        # backward pass to propagate forces up (from endeffector to root body)
        for i in range(len(self._bodies) - 2, 0, -1):
            body = self._bodies[i]
            joint_pose = child_body.joint_pose

            # pose x children_force
            child_body_force = child_body.force.transform(joint_pose)

            icxacc = body.inertia.multiply_motion_vec(body.acc)
            icxvel = body.inertia.multiply_motion_vec(body.vel)
            tmp_force = body.vel.cross_force_vec(icxvel)

            body.force = icxacc.add_force_vec(tmp_force).add_force_vec(child_body_force)
            child_body = body
        return

    def compute_inverse_dynamics(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        qdd_des: torch.Tensor,
        include_gravity: Optional[bool] = True,
    ) -> torch.Tensor:
        r"""

        Args:
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]
            qdd_des: desired joint accelerations [batch_size x n_dofs]
            include_gravity: when False, we assume gravity compensation is already taken care off

        Returns: forces to achieve desired accelerations

        """
        assert q.ndim == 2
        assert qd.ndim == 2
        assert qdd_des.ndim == 2
        assert q.shape[1] == self._n_dofs
        assert qd.shape[1] == self._n_dofs
        assert qdd_des.shape[1] == self._n_dofs

        batch_size = qdd_des.shape[0]
        force = torch.zeros_like(qdd_des)

        # we set the current state of the robot
        self.update_kinematic_state(q, qd)

        # we set the acceleration of all controlled joints to the desired accelerations
        for i in range(self._n_dofs):
            idx = self._controlled_joints[i]
            self._bodies[idx].update_joint_acc(qdd_des[:, i].unsqueeze(1))

        # forces at the base are either 0, or gravity
        base_ang_acc = q.new_zeros((batch_size, 3))
        base_lin_acc = q.new_zeros((batch_size, 3))
        if include_gravity:
            base_lin_acc[:, 2] = 9.81 * torch.ones(batch_size)

        # we propagate the base forces
        self.iterative_newton_euler(SpatialMotionVec(base_lin_acc, base_ang_acc))

        # we extract the relevant forces for all controlled joints
        for i in range(qdd_des.shape[1]):
            idx = self._controlled_joints[i]
            rot_axis = torch.zeros((batch_size, 3)).to(self._device)
            rot_axis[:, 2] = torch.ones(batch_size).to(self._device)
            force[:, i] += (
                self._bodies[idx].force.ang.unsqueeze(1) @ rot_axis.unsqueeze(2)
            ).squeeze()

        # we add forces to counteract damping
        damping_const = torch.zeros(1, self._n_dofs)
        for i in range(self._n_dofs):
            idx = self._controlled_joints[i]
            damping_const[:, i] = self._bodies[idx].get_joint_damping_const()
        force += damping_const.repeat(batch_size, 1) * qd

        return force

    def compute_non_linear_effects(
        self, q: torch.Tensor, qd: torch.Tensor, include_gravity: Optional[bool] = True
    ) -> torch.Tensor:
        r"""

        Compute the non-linear effects (Coriolis, centrifugal, gravitational, and damping effects).

        Args:
            q: joint angles [batch_size x n_dofs]
            qd: [batch_size x n_dofs]
            include_gravity: set to False if your robot has gravity compensation

        Returns:

        """
        zero_qdd = q.new_zeros(q.shape)
        return self.compute_inverse_dynamics(q, qd, zero_qdd, include_gravity)

    def compute_lagrangian_inertia_matrix(
        self, q: torch.Tensor, include_gravity: Optional[bool] = True
    ) -> torch.Tensor:
        r"""

        Args:
            q: joint angles [batch_size x n_dofs]
            include_gravity: set to False if your robot has gravity compensation

        Returns:

        """
        assert q.shape[1] == self._n_dofs
        batch_size = q.shape[0]
        identity_tensor = torch.eye(q.shape[1]).unsqueeze(0).repeat(batch_size, 1, 1)
        zero_qd = q.new_zeros(q.shape)
        zero_qdd = q.new_zeros(q.shape)
        if include_gravity:
            gravity_term = self.compute_inverse_dynamics(
                q, zero_qd, zero_qdd, include_gravity
            )
        else:
            gravity_term = q.new_zeros(q.shape)

        H = torch.stack(
            [
                (
                    self.compute_inverse_dynamics(
                        q, zero_qd, identity_tensor[:, :, j], include_gravity
                    )
                    - gravity_term
                )
                for j in range(self._n_dofs)
            ],
            dim=2,
        )
        return H

    def compute_forward_dynamics_old(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        f: torch.Tensor,
        include_gravity: Optional[bool] = True,
    ) -> torch.Tensor:
        r"""
        Computes next qdd by solving the Euler-Lagrange equation
        qdd = H^{-1} (F - Cv - G - damping_term)

        Args:
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]
            f: forces to be applied [batch_size x n_dofs]
            include_gravity: set to False if your robot has gravity compensation

        Returns: accelerations that are the result of applying forces f in state q, qd

        """

        nle = self.compute_non_linear_effects(
            q=q, qd=qd, include_gravity=include_gravity
        )
        inertia_mat = self.compute_lagrangian_inertia_matrix(
            q=q, include_gravity=include_gravity
        )

        # Solve H qdd = F - Cv - G - damping_term
        qdd = torch.solve(f.unsqueeze(2) - nle.unsqueeze(2), inertia_mat)[0].squeeze(2)

        return qdd

    def compute_forward_dynamics(
            self,
            q: torch.Tensor,
            qd: torch.Tensor,
            f: torch.Tensor,
            include_gravity: Optional[bool] = True,
            use_damping: Optional[bool] = False,
    ) -> torch.Tensor:
        r"""
        Computes next qdd via the articulated body algorithm (see Featherstones Rigid body dynamics page 132)

        Args:
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]
            f: forces to be applied [batch_size x n_dofs]
            include_gravity: set to False if your robot has gravity compensation

        Returns: accelerations that are the result of applying forces f in state q, qd

        """
        qdd = torch.zeros_like(q)
        batch_size = q.shape[0]

        if use_damping:
            damping_const = torch.zeros(1, self._n_dofs)
            for i in range(self._n_dofs):
                idx = self._controlled_joints[i]
                damping_const[:, i] = self._bodies[idx].get_joint_damping_const()
            f -= damping_const.repeat(batch_size, 1) * qd

        # we set the current state of the robot
        self.update_kinematic_state(q, qd)

        # forces at the base are either 0, or gravity
        base_ang_acc = q.new_zeros((batch_size, 3))
        base_lin_acc = q.new_zeros((batch_size, 3))
        if include_gravity:
            base_lin_acc[:, 2] = 9.81 * torch.ones(batch_size)

        base_acc = SpatialMotionVec(base_lin_acc, base_ang_acc)

        body = self._bodies[0]
        body.acc = base_acc

        for i in range(1, len(self._bodies)):
            body = self._bodies[i]

            # body velocity cross joint vel
            body.c = body.vel.cross_motion_vec(body.joint_vel)
            icxvel = body.inertia.multiply_motion_vec(body.vel)
            body.pA = body.vel.cross_force_vec(icxvel)
            # IA is 6x6, we repeat it for each item in the batch, as the inertia matrix is shared across the whole batch
            body.IA = body.inertia.get_spatial_mat().repeat((batch_size, 1, 1))

        for i in range(len(self._bodies) - 2, 0, -1):
            body = self._bodies[i]

            S = SpatialMotionVec(lin_motion=torch.zeros((batch_size, 3)),
                                 ang_motion=body.joint_axis.repeat((batch_size, 1)))
            body.S = S
            # we take the first inertia matrix, since it doesn't matter
            Utmp = body.IA[0].matmul(S.get_vector().transpose(-2, -1)).transpose(-2, -1)
            body.U = SpatialForceVec(lin_force=Utmp[:, 3:],
                                     ang_force=Utmp[:, :3])
            body.d = S.dot(body.U)
            body.u = f[:, body.joint_idx] - body.pA.dot(S)

            parent_name = self._urdf_model.get_name_of_parent_body(body.name)
            parent_idx = self._name_to_idx_map[parent_name]

            if parent_idx > 0:
                parent_body = self._bodies[parent_idx]
                U = body.U.get_vector()
                Ud = U/body.d.view(batch_size, 1)
                c = body.c.get_vector()

                # IA is of size [batch_size x 6 x 6]
                IA = body.IA - torch.bmm(U.view(batch_size, 6, 1), Ud.view(batch_size, 1, 6))

                tmp = torch.bmm(IA, c.view(batch_size, 6, 1)).squeeze(dim=2)
                tmps = SpatialForceVec(lin_force=tmp[:, 3:],
                                       ang_force=tmp[:, :3])
                ud = body.u/body.d
                uu = body.U.multiply(ud)
                pa = body.pA.add_force_vec(tmps).add_force_vec(uu)

                joint_pose = body.joint_pose

                # transform is of shape 6x6 and shared across all items in a batch
                transform_mat = joint_pose.to_matrix().repeat((batch_size, 1, 1))
                parent_body.IA += torch.bmm(transform_mat.transpose(-2, -1), IA).bmm(transform_mat)
                parent_body.pA = parent_body.pA.add_force_vec(pa.transform(joint_pose))

        base_acc = SpatialMotionVec(lin_motion=base_lin_acc, ang_motion=base_ang_acc)

        body = self._bodies[0]
        body.acc = base_acc

        # forward pass to propagate accelerations from root to end-effector link
        # Todo: -1 is a fix for now to skip final virtual ee link
        for i in range(1, len(self._bodies)-1):
            joint_idx = self._controlled_joints.index(i)
            body = self._bodies[i]
            parent_name = self._urdf_model.get_name_of_parent_body(body.name)
            parent_idx = self._name_to_idx_map[parent_name]
            parent_body = self._bodies[parent_idx]

            # get the inverse of the current joint pose
            inv_pose = body.joint_pose.inverse()

            # transform spatial acceleration of parent body into this body's frame
            acc_parent_body = parent_body.acc.transform(inv_pose)
            # body velocity cross joint vel
            body.acc = acc_parent_body.add_motion_vec(body.c)
            qdd[:, joint_idx] = (1.0/body.d) * (body.u - body.U.dot(body.acc))
            body.acc = body.acc.add_motion_vec(body.S.multiply(qdd[:, joint_idx]))

        return qdd

    def compute_endeffector_jacobian(
        self, q: torch.Tensor, link_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""

        Args:
            link_name: name of link name for the jacobian
            q: joint angles [batch_size x n_dofs]

        Returns: linear and angular jacobian

        """
        self.compute_forward_kinematics(q, link_name)

        e_pose = self._bodies[self._name_to_idx_map[link_name]].pose
        p_e = e_pose.translation()[0]

        lin_jac, ang_jac = torch.zeros([3, self._n_dofs]), torch.zeros(
            [3, self._n_dofs]
        )

        # any joints larger than this joint, will have 0 in the jacobian
        parent_name = self._urdf_model.get_name_of_parent_body(link_name)
        parent_joint_id = self._urdf_model.find_joint_of_body(parent_name)

        for i, idx in enumerate(self._controlled_joints):
            if (idx -1) > parent_joint_id:
                continue
            pose = self._bodies[idx].pose
            axis = self._bodies[idx].joint_axis
            axis_idx = int(torch.where(axis[0])[0])
            p_i, z_i = pose.translation()[0], pose.rotation()[0, :, axis_idx]
            lin_jac[:, i] = torch.cross(z_i, p_e - p_i)
            ang_jac[:, i] = z_i

        return lin_jac, ang_jac

    def get_joint_limits(self) -> List[Dict[str, torch.Tensor]]:
        r"""

        Returns: list of joint limit dict, containing joint position, velocity and effort limits

        """
        limits = []
        for idx in self._controlled_joints:
            limits.append(self._bodies[idx].get_joint_limits())
        return limits

    def get_link_names(self) -> List[str]:
        r"""

        Returns: a list containing names for all links

        """

        link_names = []
        for i in range(len(self._bodies)):
            link_names.append(self._bodies[i].name)
        return link_names

    def print_link_names(self) -> None:
        r"""

        print the names of all links

        """
        for i in range(len(self._bodies)):
            print(self._bodies[i].name)

    def print_learnable_params(self) -> None:
        r"""

        print the name and value of all learnable parameters

        """
        for name, param in self.named_parameters():
            print(f"{name}: {param}")


class LearnableRigidBodyConfig:
    def __init__(self, learnable_links=[], learnable_kinematics_params=[], learnable_dynamics_params=[]):
        self.learnable_links = learnable_links
        self.learnable_kinematics_params = learnable_kinematics_params
        self.learnable_dynamics_params = learnable_dynamics_params


class DifferentiableKUKAiiwa(DifferentiableRobotModel):
    def __init__(self):
        rel_urdf_path = "kuka_iiwa/urdf/iiwa7.urdf"
        self.urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
        self.learnable_rigid_body_config = LearnableRigidBodyConfig()
        self.name = "differentiable_kuka_iiwa"
        super().__init__(self.urdf_path, self.learnable_rigid_body_config, self.name)


class DifferentiableFrankaPanda(DifferentiableRobotModel):
    def __init__(self):
        rel_urdf_path = "panda_description/urdf/panda_no_gripper.urdf"
        self.urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
        self.learnable_rigid_body_config = LearnableRigidBodyConfig()
        self.name = "differentiable_franka_panda"
        super().__init__(self.urdf_path, self.learnable_rigid_body_config, self.name)


class DifferentiableTwoLinkRobot(DifferentiableRobotModel):
    def __init__(self):
        rel_urdf_path ="2link_robot.urdf"
        self.urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
        self.learnable_rigid_body_config = LearnableRigidBodyConfig()
        self.name = "diff_2d_robot"
        super().__init__(self.urdf_path, self.learnable_rigid_body_config, self.name)
