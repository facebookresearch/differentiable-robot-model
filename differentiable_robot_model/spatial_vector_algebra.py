import torch
import math
from . import utils
from .utils import cross_product


class SpatialMotionVec(object):

    def __init__(self, lin_motion=None, ang_motion=None):
        if lin_motion is None:
            self.lin = torch.zeros((1, 3))
        else:
            self.lin = lin_motion
        if ang_motion is None:
            self.ang = torch.zeros((1, 3))
        else:
            self.ang = ang_motion

    def add_motion_vec(self, smv):
        return SpatialMotionVec(self.lin + smv.lin, self.ang + smv.ang)

    def cross_motion_vec(self, smv):
        new_ang = cross_product(self.ang, smv.ang)
        new_lin = cross_product(self.ang, smv.lin) + cross_product(self.lin, smv.ang)
        return SpatialMotionVec(new_lin, new_ang)

    def cross_force_vec(self, sfv):
        new_ang = cross_product(self.ang, sfv.ang) + cross_product(self.lin, sfv.lin)
        new_lin = cross_product(self.ang, sfv.lin)
        return SpatialForceVec(new_lin, new_ang)

    def transform(self, transform):
        new_ang = (transform.rotation() @ self.ang.unsqueeze(2)).squeeze(2)
        new_lin = (transform.trans_cross_rot() @ self.ang.unsqueeze(2)).squeeze(2)
        new_lin += (transform.rotation() @ self.lin.unsqueeze(2)).squeeze(2)
        return SpatialMotionVec(new_lin, new_ang)


class SpatialForceVec(object):
    def __init__(self, lin_force=None, ang_force=None):
        if lin_force is None:
            self.lin = torch.zeros((1, 3))
        else:
            self.lin = lin_force
        if ang_force is None:
            self.ang = torch.zeros((1, 3))
        else:
            self.ang = ang_force

    def add_force_vec(self, sfv):
        return SpatialForceVec(self.lin + sfv.lin, self.ang + sfv.ang)

    def transform(self, transform):
        new_lin = (transform.rotation() @ self.lin.unsqueeze(2)).squeeze(2)
        new_ang = (transform.trans_cross_rot() @ self.lin.unsqueeze(2)).squeeze(2)
        new_ang += (transform.rotation() @ self.ang.unsqueeze(2)).squeeze(2)
        return SpatialForceVec(new_lin, new_ang)


class DifferentiableSpatialRigidBodyInertia(torch.nn.Module):

    def __init__(self, rigid_body_params):
        super().__init__()
        self.mass = rigid_body_params["mass"]
        self.com = rigid_body_params["com"]
        self.inertia_mat = rigid_body_params["inertia_mat"]

    def _get_parameter_values(self):
        return self.mass, self.com, self.inertia_mat

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
        new_ang_force = (inertia.repeat(batch_size, 1, 1) @ smv.ang.unsqueeze(2)).squeeze(
            2
        ) + utils.cross_product(mcom.repeat(batch_size, 1), smv.lin)

        return SpatialForceVec(new_lin_force, new_ang_force)

def x_rot(angle):
    if len(angle.shape) == 0:
        angle = angle.unsqueeze(0)
    angle = utils.convert_into_at_least_2d_pytorch_tensor(angle).squeeze(1)
    batch_size = angle.shape[0]
    R = torch.zeros((batch_size, 3, 3))
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
    R = torch.zeros((batch_size, 3, 3))
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
    R = torch.zeros((batch_size, 3, 3))
    R[:, 0, 0] = torch.cos(angle)
    R[:, 0, 1] = -torch.sin(angle)
    R[:, 1, 0] = torch.sin(angle)
    R[:, 1, 1] = torch.cos(angle)
    R[:, 2, 2] = torch.ones(batch_size)
    return R


class CoordinateTransform(object):
    def __init__(self, rot=None, trans=None):

        if rot is None:
            self._rot = torch.eye(3)
        else:
            self._rot = rot
        if len(self._rot.shape) == 2:
            self._rot = self._rot.unsqueeze(0)

        if trans is None:
            self._trans = torch.zeros(3)
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
        return CoordinateTransform(rot_transpose, -(rot_transpose @ self._trans.unsqueeze(2)).squeeze(2))

    def multiply_transform(self, coordinate_transform):
        new_rot = self._rot @ coordinate_transform.rotation()
        new_trans = (self._rot @ coordinate_transform.translation().unsqueeze(2)).squeeze(2) + self._trans
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
        t = torch.einsum('bii->b', M) #torch.trace(M)
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
                #q = q[[3, 0, 1, 2]]
            q[n, :] *= 0.5 / math.sqrt(tn * M[n, 3, 3])
        return q

