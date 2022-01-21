"""
Spatial vector algebra
====================================
TODO
"""
from __future__ import annotations
from typing import Optional
import torch
import math
from . import utils
from .utils import cross_product


def x_rot(angle):
    if len(angle.shape) == 0:
        angle = angle.unsqueeze(0)
    angle = utils.convert_into_at_least_2d_pytorch_tensor(angle).squeeze(1)
    batch_size = angle.shape[0]
    R = torch.zeros((batch_size, 3, 3), device=angle.device)
    R[:, 0, 0] = torch.ones(batch_size)
    R[:, 1, 1] = torch.cos(angle)
    R[:, 1, 2] = -torch.sin(angle)
    R[:, 2, 1] = torch.sin(angle)
    R[:, 2, 2] = torch.cos(angle)
    return R


def y_rot(angle):
    if len(angle.shape) == 0:
        angle = angle.unsqueeze(0)
    angle = utils.convert_into_at_least_2d_pytorch_tensor(angle).squeeze(1)
    batch_size = angle.shape[0]
    R = torch.zeros((batch_size, 3, 3), device=angle.device)
    R[:, 0, 0] = torch.cos(angle)
    R[:, 0, 2] = torch.sin(angle)
    R[:, 1, 1] = torch.ones(batch_size)
    R[:, 2, 0] = -torch.sin(angle)
    R[:, 2, 2] = torch.cos(angle)
    return R


def z_rot(angle):
    if len(angle.shape) == 0:
        angle = angle.unsqueeze(0)
    angle = utils.convert_into_at_least_2d_pytorch_tensor(angle).squeeze(1)
    batch_size = angle.shape[0]
    R = torch.zeros((batch_size, 3, 3), device=angle.device)
    R[:, 0, 0] = torch.cos(angle)
    R[:, 0, 1] = -torch.sin(angle)
    R[:, 1, 0] = torch.sin(angle)
    R[:, 1, 1] = torch.cos(angle)
    R[:, 2, 2] = torch.ones(batch_size)
    return R


class CoordinateTransform(object):
    def __init__(self, rot=None, trans=None, device="cpu"):
        self._device = torch.device(device)

        if rot is None:
            self._rot = torch.eye(3, device=self._device)
        else:
            self._rot = rot
        if len(self._rot.shape) == 2:
            self._rot = self._rot.unsqueeze(0)

        if trans is None:
            self._trans = torch.zeros(3, device=self._device)
        else:
            self._trans = trans
        if len(self._trans.shape) == 1:
            self._trans = self._trans.unsqueeze(0)

    def set_translation(self, t):
        self._trans = t
        if len(self._trans.shape) == 1:
            self._trans = self._trans.unsqueeze(0)
        return

    def set_rotation(self, rot):
        self._rot = rot
        if len(self._rot.shape) == 2:
            self._rot = self._rot.unsqueeze(0)
        return

    def rotation(self):
        return self._rot

    def translation(self):
        return self._trans

    def inverse(self):
        rot_transpose = self._rot.transpose(-2, -1)
        return CoordinateTransform(
            rot_transpose, -(rot_transpose @ self._trans.unsqueeze(2)).squeeze(2)
        )

    def multiply_transform(self, coordinate_transform):
        new_rot = self._rot @ coordinate_transform.rotation()
        new_trans = (
            self._rot @ coordinate_transform.translation().unsqueeze(2)
        ).squeeze(2) + self._trans
        return CoordinateTransform(new_rot, new_trans)

    def trans_cross_rot(self):
        return utils.vector3_to_skew_symm_matrix(self._trans) @ self._rot

    def get_quaternion(self):
        batch_size = self._rot.shape[0]
        M = torch.zeros((batch_size, 4, 4)).to(self._rot.device)
        M[:, :3, :3] = self._rot
        M[:, :3, 3] = self._trans
        M[:, 3, 3] = 1
        q = torch.empty((batch_size, 4)).to(self._rot.device)
        t = torch.einsum("bii->b", M)  # torch.trace(M)
        for n in range(batch_size):
            tn = t[n]
            if tn > M[n, 3, 3]:
                q[n, 3] = tn
                q[n, 2] = M[n, 1, 0] - M[n, 0, 1]
                q[n, 1] = M[n, 0, 2] - M[n, 2, 0]
                q[n, 0] = M[n, 2, 1] - M[n, 1, 2]
            else:
                i, j, k = 0, 1, 2
                if M[n, 1, 1] > M[n, 0, 0]:
                    i, j, k = 1, 2, 0
                if M[n, 2, 2] > M[n, i, i]:
                    i, j, k = 2, 0, 1
                tn = M[n, i, i] - (M[n, j, j] + M[n, k, k]) + M[n, 3, 3]
                q[n, i] = tn
                q[n, j] = M[n, i, j] + M[n, j, i]
                q[n, k] = M[n, k, i] + M[n, i, k]
                q[n, 3] = M[n, k, j] - M[n, j, k]
                # q = q[[3, 0, 1, 2]]
            q[n, :] *= 0.5 / math.sqrt(tn * M[n, 3, 3])
        return q

    def to_matrix(self):
        batch_size = self._rot.shape[0]

        mat = torch.zeros((batch_size, 6, 6), device=self._device)
        t = torch.zeros((batch_size, 3, 3), device=self._device)
        t[:, 0, 1] = -self._trans[:, 2]
        t[:, 0, 2] = self._trans[:, 1]
        t[:, 1, 0] = self._trans[:, 2]
        t[:, 1, 2] = -self._trans[:, 0]
        t[:, 2, 0] = -self._trans[:, 1]
        t[:, 2, 1] = self._trans[:, 0]
        _Erx = self._rot.transpose(-2, -1).matmul(t)

        mat[:, :3, :3] = self._rot.transpose(-2, -1)
        mat[:, 3:, 0:3] = -_Erx
        mat[:, 3:, 3:] = self._rot.transpose(-2, -1)
        return mat

    def to_matrix_transpose(self):
        batch_size = self._rot.shape[0]

        mat = torch.zeros((batch_size, 6, 6), device=self._device)
        t = torch.zeros((batch_size, 3, 3), device=self._device)
        t[:, 0, 1] = -self._trans[:, 2]
        t[:, 0, 2] = self._trans[:, 1]
        t[:, 1, 0] = self._trans[:, 2]
        t[:, 1, 2] = -self._trans[:, 0]
        t[:, 2, 0] = -self._trans[:, 1]
        t[:, 2, 1] = self._trans[:, 0]
        _Erx = self._rot.matmul(t)

        mat[:, :3, :3] = self._rot.transpose(-1, -2)
        mat[:, 3:, 0:3] = -_Erx.transpose(-1, -2)
        mat[:, 3:, 3:] = self._rot.transpose(-1, -2)
        return mat


class SpatialMotionVec(object):
    def __init__(
        self,
        lin_motion: Optional[torch.Tensor] = None,
        ang_motion: Optional[torch.Tensor] = None,
        device=None,
    ):
        if lin_motion is None or ang_motion is None:
            assert (
                device is not None
            ), "Cannot initialize with default values without specifying device."
            device = torch.device(device)
        self.lin = (
            lin_motion if lin_motion is not None else torch.zeros((1, 3), device=device)
        )
        self.ang = (
            ang_motion if ang_motion is not None else torch.zeros((1, 3), device=device)
        )

    def add_motion_vec(self, smv: SpatialMotionVec) -> SpatialMotionVec:
        r"""
        Args:
            smv: spatial motion vector
        Returns:
            the sum of motion vectors
        """

        return SpatialMotionVec(self.lin + smv.lin, self.ang + smv.ang)

    def cross_motion_vec(self, smv: SpatialMotionVec) -> SpatialMotionVec:
        r"""
        Args:
            smv: spatial motion vector
        Returns:
            the cross product between motion vectors
        """
        new_ang = cross_product(self.ang, smv.ang)
        new_lin = cross_product(self.ang, smv.lin) + cross_product(self.lin, smv.ang)
        return SpatialMotionVec(new_lin, new_ang)

    def cross_force_vec(self, sfv: SpatialForceVec) -> SpatialForceVec:
        r"""
        Args:
            sfv: spatial force vector
        Returns:
            the cross product between motion (self) and force vector
        """
        new_ang = cross_product(self.ang, sfv.ang) + cross_product(self.lin, sfv.lin)
        new_lin = cross_product(self.ang, sfv.lin)
        return SpatialForceVec(new_lin, new_ang)

    def transform(self, transform: CoordinateTransform) -> SpatialMotionVec:
        r"""
        Args:
            transform: a coordinate transform object
        Returns:
            the motion vector (self) transformed by the coordinate transform
        """
        new_ang = (transform.rotation() @ self.ang.unsqueeze(2)).squeeze(2)
        new_lin = (transform.trans_cross_rot() @ self.ang.unsqueeze(2)).squeeze(2)
        new_lin += (transform.rotation() @ self.lin.unsqueeze(2)).squeeze(2)
        return SpatialMotionVec(new_lin, new_ang)

    def get_vector(self):
        return torch.cat([self.ang, self.lin], dim=1)

    def multiply(self, v):
        batch_size = self.lin.shape[0]
        return SpatialForceVec(
            self.lin * v.view(batch_size, 1), self.ang * v.view(batch_size, 1)
        )

    def dot(self, smv):
        tmp1 = torch.sum(self.ang * smv.ang, dim=-1)
        tmp2 = torch.sum(self.lin * smv.lin, dim=-1)
        return tmp1 + tmp2


class SpatialForceVec(object):
    def __init__(
        self,
        lin_force: Optional[torch.Tensor] = None,
        ang_force: Optional[torch.Tensor] = None,
        device=None,
    ):
        if lin_force is None or ang_force is None:
            assert (
                device is not None
            ), "Cannot initialize with default values without specifying device."
            device = torch.device(device)
        self.lin = (
            lin_force if lin_force is not None else torch.zeros((1, 3), device=device)
        )
        self.ang = (
            ang_force if ang_force is not None else torch.zeros((1, 3), device=device)
        )

    def add_force_vec(self, sfv: SpatialForceVec) -> SpatialForceVec:
        r"""
        Args:
            sfv: spatial force vector
        Returns:
            the sum of force vectors
        """
        return SpatialForceVec(self.lin + sfv.lin, self.ang + sfv.ang)

    def transform(self, transform: CoordinateTransform) -> SpatialForceVec:
        r"""
        Args:
            transform: a coordinate transform object
        Returns:
            the force vector (self) transformed by the coordinate transform
        """
        new_lin = (transform.rotation() @ self.lin.unsqueeze(2)).squeeze(2)
        new_ang = (transform.trans_cross_rot() @ self.lin.unsqueeze(2)).squeeze(2)
        new_ang += (transform.rotation() @ self.ang.unsqueeze(2)).squeeze(2)
        return SpatialForceVec(new_lin, new_ang)

    def get_vector(self):
        return torch.cat([self.ang, self.lin], dim=1)

    def multiply(self, v):
        batch_size = self.lin.shape[0]
        return SpatialForceVec(
            self.lin * v.view(batch_size, 1), self.ang * v.view(batch_size, 1)
        )

    def dot(self, smv):
        tmp1 = torch.sum(self.ang * smv.ang, dim=-1)
        tmp2 = torch.sum(self.lin * smv.lin, dim=-1)
        return tmp1 + tmp2


class DifferentiableSpatialRigidBodyInertia(torch.nn.Module):
    def __init__(self, rigid_body_params, device="cpu"):
        super().__init__()
        # lambda functions are a "hack" to make this compatible with the learnable variants
        self.mass = lambda: rigid_body_params["mass"]
        self.com = lambda: rigid_body_params["com"]
        self.inertia_mat = lambda: rigid_body_params["inertia_mat"]

        self._device = torch.device(device)

    def _get_parameter_values(self):
        return self.mass(), self.com(), self.inertia_mat()

    def multiply_motion_vec(self, smv):
        mass, com, inertia_mat = self._get_parameter_values()
        mcom = com * mass
        com_skew_symm_mat = utils.vector3_to_skew_symm_matrix(com)
        inertia = inertia_mat + mass * (
            com_skew_symm_mat @ com_skew_symm_mat.transpose(-2, -1)
        )

        batch_size = smv.lin.shape[0]

        new_lin_force = mass * smv.lin - utils.cross_product(
            mcom.repeat(batch_size, 1), smv.ang
        )
        new_ang_force = (
            inertia.repeat(batch_size, 1, 1) @ smv.ang.unsqueeze(2)
        ).squeeze(2) + utils.cross_product(mcom.repeat(batch_size, 1), smv.lin)

        return SpatialForceVec(new_lin_force, new_ang_force)

    def get_spatial_mat(self):
        mass, com, inertia_mat = self._get_parameter_values()
        mcom = mass * com
        com_skew_symm_mat = utils.vector3_to_skew_symm_matrix(com)
        inertia = inertia_mat + mass * (
            com_skew_symm_mat @ com_skew_symm_mat.transpose(-2, -1)
        )
        mat = torch.zeros((6, 6), device=self._device)
        mat[:3, :3] = inertia
        mat[3, 0] = 0
        mat[3, 1] = mcom[0, 2]
        mat[3, 2] = -mcom[0, 1]
        mat[4, 0] = -mcom[0, 2]
        mat[4, 1] = 0.0
        mat[4, 2] = mcom[0, 0]
        mat[5, 0] = mcom[0, 1]
        mat[5, 1] = -mcom[0, 0]
        mat[5, 2] = 0.0

        mat[0, 3] = 0
        mat[0, 4] = -mcom[0, 2]
        mat[0, 5] = mcom[0, 1]
        mat[1, 3] = mcom[0, 2]
        mat[1, 4] = 0.0
        mat[1, 5] = -mcom[0, 0]
        mat[2, 3] = -mcom[0, 1]
        mat[2, 4] = mcom[0, 0]
        mat[2, 5] = 0.0

        mat[3, 3] = mass
        mat[4, 4] = mass
        mat[5, 5] = mass
        return mat
