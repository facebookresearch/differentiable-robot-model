# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import numpy as np
import math

from . import se3_so3_util, utils

torch.set_default_tensor_type(torch.DoubleTensor)


class NonNegativeScalarNet(torch.nn.Module):
    def __init__(self, bias=0.0, init_param_std=1.0, init_param=None, is_initializing_params=True):
        self._bias = bias
        super().__init__()
        if (init_param is None) or (not is_initializing_params):
            init_param_value = torch.empty(1, 1).normal_(mean=0.0, std=init_param_std)
        else:
            init_param_value = torch.sqrt(init_param - self._bias)
        self.l = torch.nn.Parameter(init_param_value.squeeze())
        self.l.requires_grad = True

    def forward(self):
        non_negative_scalar = ((self.l * self.l) + self._bias).squeeze()
        return non_negative_scalar


class NoStructureNet(torch.nn.Module):
    def __init__(
        self,
        dim1,
        dim2,
        init_param_std=1.0,
        is_using_positive_initial_guest=False,
        init_param=None,
        is_initializing_params=True,
    ):
        self._dim1 = dim1
        self._dim2 = dim2
        super().__init__()
        if (init_param is None) or (not is_initializing_params):
            if is_using_positive_initial_guest:
                init_param_value = torch.abs(
                    torch.empty(1, self._dim1, self._dim2).normal_(
                        mean=0.0, std=init_param_std
                    )
                )
            else:
                init_param_value = torch.empty(1, self._dim1, self._dim2).normal_(
                    mean=0.0, std=init_param_std
                )
        else:
            init_param_value = init_param
        self.params = torch.nn.Parameter(init_param_value.squeeze())
        self.params.requires_grad = True

    def forward(self):
        return self.params.squeeze()


class MCoM3DNet(NoStructureNet):
    def __init__(self, init_param=None, is_initializing_params=True):
        super().__init__(
            dim1=1,
            dim2=3,
            is_using_positive_initial_guest=True,
            init_param=init_param,
            is_initializing_params=is_initializing_params,
        )


class SymmMatNet(torch.nn.Module):
    """
    Symmetric Matrix Networks
    """

    def __init__(self, qdim):
        self._qdim = qdim
        super().__init__()

    def forward(self, l):
        """

        :param l: vector containing lower triangular and diagonal components of the output symmetric matrix SM
        :return: Symmetric matrix SM
        """
        batch_size = l.size(0)
        SM = l.new_zeros(batch_size, self._qdim, self._qdim)
        L_tril = l.new_zeros(batch_size, self._qdim, self._qdim)
        if self._qdim > 1:
            l_tril = l[:, self._qdim :]
            L_tril = utils.bfill_lowertriangle(L_tril, l_tril)
        l_diag = l[:, : self._qdim]
        SM = utils.bfill_diagonal(SM, l_diag)
        SM += L_tril + L_tril.transpose(-2, -1)
        return SM


class CholeskyNet(torch.nn.Module):
    """
    Symmetric Positive Definite Matrix Networks via Cholesky Decomposition
    """

    def __init__(self, qdim, bias):
        self._qdim = qdim
        self._bias = bias
        super().__init__()

    def get_raw_l(self, raw_l_input):
        """
        Return vector raw_l, which is the non-zero elements of lower-triangular matrix L in Cholesky decomposition,
        WITHOUT adding positive bias (yet) to the components of raw_l that corresponds to the diagonal components of L.
        """
        return raw_l_input  # identity mapping

    def get_l(self, raw_l_input):
        raw_l = self.get_raw_l(raw_l_input)
        l = raw_l.new_zeros(raw_l.shape)
        l[
        :, : self._qdim
        ] += (
            self._bias
        )  # add bias to ensure positive definiteness of the resulting inertia matrix
        l += raw_l
        return l

    def get_L(self, l):
        batch_size = l.size(0)
        L = l.new_zeros(batch_size, self._qdim, self._qdim)
        if self._qdim > 1:
            l_tril = l[:, self._qdim :]
            L = utils.bfill_lowertriangle(L, l_tril)
        l_diag = l[:, : self._qdim]
        L = utils.bfill_diagonal(L, l_diag)
        return L

    def get_symm_pos_semi_def_matrix_and_l(self, raw_l_input):
        """
        :param raw_l_input: please see definition of get_raw_l()
        :return: Symmetric positive semi-definite matrix SPSD and the vector l (please see definition of get_l())
        """
        l = self.get_l(raw_l_input)
        L = self.get_L(l)
        SPSD = L @ L.transpose(-2, -1)
        return SPSD, l

class TriangParam3DInertiaMatrixNet(torch.nn.Module):
    """
    3D inertia matrix with triangular parameterized principal moments of inertia
    """

    def __init__(
        self, bias, init_param_std=0.01, init_param=None, is_initializing_params=True
    ):
        self._qdim = 3
        self._bias = bias
        super().__init__()

        if (init_param is None) or (not is_initializing_params):
            init_inertia_ori_axis_angle_param_value = torch.empty(1, 3).normal_(
                mean=0.0, std=init_param_std
            )
            init_J1_param_value = None
            init_J2_param_value = None
            init_alpha_param_param_value = None
        else:
            init_param = init_param.squeeze().numpy()

            [R, J_diag, _] = np.linalg.svd(init_param, full_matrices=True)
            if (
                np.linalg.det(R) < 0.0
            ):  # make sure this is really a member of SO(3), not just O(3)
                R[:, 0] = -R[:, 0]
            init_inertia_ori_axis_angle_param_value = (
                se3_so3_util.getVec3FromSkewSymMat(se3_so3_util.logMapSO3(R))
            )
            init_J1_param_value = J_diag[0]
            init_J2_param_value = J_diag[1]
            init_alpha_param_value = np.arccos(
                (
                    (J_diag[0] * J_diag[0])
                    + (J_diag[1] * J_diag[1])
                    - (J_diag[2] * J_diag[2])
                )
                / (2.0 * J_diag[0] * J_diag[1])
            )
            init_alpha_div_pi_param_value = init_alpha_param_value / math.pi
            # inverse sigmoid:
            init_alpha_param_param_value = np.log(
                init_alpha_div_pi_param_value / (1.0 - init_alpha_div_pi_param_value)
            )

            init_inertia_ori_axis_angle_param_value = torch.tensor(
                init_inertia_ori_axis_angle_param_value, dtype=torch.float32
            )
            init_J1_param_value = torch.tensor(init_J1_param_value, dtype=torch.float32)
            init_J2_param_value = torch.tensor(init_J2_param_value, dtype=torch.float32)
            assert (
                init_J1_param_value > bias
            ), "Please set bias value smaller, such that this condition is satisfied!"
            assert (
                init_J2_param_value > bias
            ), "Please set bias value smaller, such that this condition is satisfied!"
            init_alpha_param_param_value = torch.tensor(
                init_alpha_param_param_value, dtype=torch.float32
            )

        self.inertia_ori_axis_angle = torch.nn.Parameter(
            init_inertia_ori_axis_angle_param_value.squeeze()
        )
        self.inertia_ori_axis_angle.requires_grad = True

        self.J1net = NonNegativeScalarNet(
            bias=bias,
            init_param_std=0.1,
            init_param=init_J1_param_value,
            is_initializing_params=is_initializing_params,
        )
        self.J2net = NonNegativeScalarNet(
            bias=bias,
            init_param_std=0.1,
            init_param=init_J2_param_value,
            is_initializing_params=is_initializing_params,
        )
        self.alpha_param_net = NoStructureNet(
            dim1=1,
            dim2=1,
            init_param_std=init_param_std,
            init_param=init_alpha_param_param_value,
            is_initializing_params=is_initializing_params,
        )

        self.J = None
        self.R = None
        self.inertia_mat = None

    def forward(self):
        alpha = math.pi * torch.sigmoid(
            self.alpha_param_net().squeeze()
        )  # 0 < alpha < pi
        J1 = self.J1net().squeeze()
        J2 = self.J2net().squeeze()
        J3 = torch.sqrt((J1 * J1) + (J2 * J2) - (2.0 * J1 * J2 * torch.cos(alpha)))

        self.J = torch.zeros((3, 3))
        self.J[0, 0] = J1
        self.J[1, 1] = J2
        self.J[2, 2] = J3

        self.R = utils.exp_map_so3(self.inertia_ori_axis_angle)

        self.inertia_mat = self.R @ (self.J @ self.R.t())

        # if (np.isnan(self.inertia_mat.detach().numpy()).any()):
        #     print(self.inertia_mat)

        return self.inertia_mat


class CovParameterized3DInertiaMatrixNet(CholeskyNet):
    """
    Inertia matrix parameterized by density-weighted covariance of a rigid body
    (please see the paper "Linear Matrix Inequalities for Physically-Consistent Inertial Parameter Identification:
     A Statistical Perspective on the Mass Distribution" by Wensing et al. (2017), section IV.A and IV.B)
    """

    def __init__(
        self,
        bias=1.0e-7,
        init_param_std=0.01,
        init_param=None,
        is_initializing_params=True,
    ):
        super().__init__(qdim=3, bias=0)
        self.spd_3d_cov_inertia_mat_diag_bias = bias
        if (init_param is None) or (not is_initializing_params):
            init_param_value = torch.empty(1, 6).normal_(mean=0.0, std=init_param_std)
        else:
            init_inertia_matrix = init_param.squeeze()
            init_spd_3d_cov_inertia_matrix = init_param.new_zeros((3, 3))
            init_spd_3d_cov_inertia_matrix[0, 0] = 0.5 * (
                -init_inertia_matrix[0, 0]
                + init_inertia_matrix[1, 1]
                + init_inertia_matrix[2, 2]
            )
            init_spd_3d_cov_inertia_matrix[1, 1] = 0.5 * (
                init_inertia_matrix[0, 0]
                - init_inertia_matrix[1, 1]
                + init_inertia_matrix[2, 2]
            )
            init_spd_3d_cov_inertia_matrix[2, 2] = 0.5 * (
                init_inertia_matrix[0, 0]
                + init_inertia_matrix[1, 1]
                - init_inertia_matrix[2, 2]
            )
            init_spd_3d_cov_inertia_matrix[1, 0] = -init_inertia_matrix[1, 0]
            init_spd_3d_cov_inertia_matrix[2, 0] = -init_inertia_matrix[2, 0]
            init_spd_3d_cov_inertia_matrix[2, 1] = -init_inertia_matrix[2, 1]
            init_spd_3d_cov_inertia_matrix[0, 1] = init_spd_3d_cov_inertia_matrix[1, 0]
            init_spd_3d_cov_inertia_matrix[0, 2] = init_spd_3d_cov_inertia_matrix[2, 0]
            init_spd_3d_cov_inertia_matrix[1, 2] = init_spd_3d_cov_inertia_matrix[2, 1]
            L = torch.tensor(
                np.linalg.cholesky(
                    init_spd_3d_cov_inertia_matrix.numpy()
                    - (self.spd_3d_cov_inertia_mat_diag_bias * np.eye(3))
                )
            )
            diag_indices = np.diag_indices(
                min(
                    init_spd_3d_cov_inertia_matrix.size(-2),
                    init_spd_3d_cov_inertia_matrix.size(-1),
                )
            )
            tril_indices = np.tril_indices(
                init_spd_3d_cov_inertia_matrix.size(-2),
                k=-1,
                m=init_spd_3d_cov_inertia_matrix.size(-1),
            )
            dim0_indices = np.hstack([diag_indices[0], tril_indices[0]])
            dim1_indices = np.hstack([diag_indices[1], tril_indices[1]])
            init_param_value = L[dim0_indices, dim1_indices].reshape(
                (1, dim0_indices.shape[0])
            )
        self.l = torch.nn.Parameter(init_param_value.squeeze())
        self.l.requires_grad = True

    def forward(self):
        raw_l_input = self.l.unsqueeze(0)
        [spsd_3d_cov_inertia_matrix, _] = super().get_symm_pos_semi_def_matrix_and_l(
            raw_l_input=raw_l_input
        )
        spsd_3d_cov_inertia_matrix = spsd_3d_cov_inertia_matrix.squeeze()
        spd_3d_cov_inertia_matrix = (
            spsd_3d_cov_inertia_matrix
            + (self.spd_3d_cov_inertia_mat_diag_bias * torch.eye(3))
        )
        inertia_matrix = spd_3d_cov_inertia_matrix.new_zeros((3, 3))
        inertia_matrix[0, 0] = (
            spd_3d_cov_inertia_matrix[1, 1] + spd_3d_cov_inertia_matrix[2, 2]
        )
        inertia_matrix[1, 1] = (
            spd_3d_cov_inertia_matrix[0, 0] + spd_3d_cov_inertia_matrix[2, 2]
        )
        inertia_matrix[2, 2] = (
            spd_3d_cov_inertia_matrix[0, 0] + spd_3d_cov_inertia_matrix[1, 1]
        )
        inertia_matrix[1, 0] = -spd_3d_cov_inertia_matrix[1, 0]
        inertia_matrix[2, 0] = -spd_3d_cov_inertia_matrix[2, 0]
        inertia_matrix[2, 1] = -spd_3d_cov_inertia_matrix[2, 1]
        inertia_matrix[0, 1] = inertia_matrix[1, 0]
        inertia_matrix[0, 2] = inertia_matrix[2, 0]
        inertia_matrix[1, 2] = inertia_matrix[2, 1]
        return inertia_matrix


class SymmPosDef3DInertiaMatrixNet(CholeskyNet):
    def __init__(
        self, bias, init_param_std=0.01, init_param=None, is_initializing_params=True
    ):
        super().__init__(qdim=3, bias=0)
        self.spd_3d_inertia_mat_diag_bias = bias
        if (init_param is None) or (not is_initializing_params):
            init_param_value = torch.empty(1, 6).normal_(mean=0.0, std=init_param_std)
        else:
            L = torch.tensor(
                np.linalg.cholesky(
                    init_param.squeeze().numpy()
                    - (self.spd_3d_inertia_mat_diag_bias * np.eye(3))
                )
            )
            diag_indices = np.diag_indices(
                min(init_param.size(-2), init_param.size(-1))
            )
            tril_indices = np.tril_indices(
                init_param.size(-2), k=-1, m=init_param.size(-1)
            )
            dim0_indices = np.hstack([diag_indices[0], tril_indices[0]])
            dim1_indices = np.hstack([diag_indices[1], tril_indices[1]])
            init_param_value = L[dim0_indices, dim1_indices].reshape(
                (1, dim0_indices.shape[0])
            )
        self.l = torch.nn.Parameter(init_param_value.squeeze())
        self.l.requires_grad = True

    def forward(self):
        raw_l_input = self.l.unsqueeze(0)
        [spsd_3d_inertia_matrix, _] = super().get_symm_pos_semi_def_matrix_and_l(
            raw_l_input=raw_l_input
        )
        spd_3d_inertia_matrix = (
            spsd_3d_inertia_matrix.squeeze()
            + (self.spd_3d_inertia_mat_diag_bias * torch.eye(3))
        )
        return spd_3d_inertia_matrix


class Symm3DInertiaMatrixNet(SymmMatNet):
    def __init__(
        self, init_param_std=0.01, init_param=None, is_initializing_params=True
    ):
        super().__init__(qdim=3)
        if (init_param is None) or (not is_initializing_params):
            init_param_value = torch.empty(1, 6).normal_(mean=0.0, std=init_param_std)
        else:
            diag_indices = np.diag_indices(3)
            tril_indices = np.tril_indices(3, k=-1, m=3)
            dim0_indices = np.hstack([diag_indices[0], tril_indices[0]])
            dim1_indices = np.hstack([diag_indices[1], tril_indices[1]])
            init_param_value = init_param[0, dim0_indices, dim1_indices].reshape((1, 6))
        self.l = torch.nn.Parameter(init_param_value.squeeze())
        self.l.requires_grad = True

    def forward(self):
        return super().forward(self.l.unsqueeze(0)).squeeze()


class InertiaMatrix3DNoStructureNet(NoStructureNet):
    def __init__(self, init_param=None, is_initializing_params=True):
        super().__init__(
            dim1=3,
            dim2=3,
            init_param=init_param,
            is_initializing_params=is_initializing_params,
        )


class UnconstrainedMassValue(torch.nn.Module):
    def __init__(self, init_val=None, device="cpu"):
        super(UnconstrainedMassValue, self).__init__()
        if init_val is None:
            self.mass = torch.nn.Parameter(torch.rand(1)).to(device)
        else:
            self.mass = torch.nn.Parameter(init_val).to(device)

    def forward(self):
        return self.mass


class PositiveMassValue(torch.nn.Module):
    def __init__(self, init_val=None, device="cpu"):
        super(PositiveMassValue, self).__init__()
        self.min_mass_val = torch.tensor(0.01).to(device)

        if init_val is None:
            init_param_value = torch.sqrt(torch.rand(1) ** 2)
        else:
            init_param_value = torch.sqrt(init_val)
        self.sqrt_mass = torch.nn.Parameter(init_param_value).to(device)

    def forward(self):
        return self.sqrt_mass * self.sqrt_mass + self.min_mass_val