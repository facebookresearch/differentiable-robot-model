"""
SE3 SO3 utilities
====================================
@author: gsutanto
@comment: implemented from "A Mathematical Introduction to Robotic Manipulation"
          textbook by Murray et al., page 413-414
"""

import torch


assert_epsilon = 1.0e-3


def integrateAxisAngle(axis_angle, omega, dt):
    R_curr = expMapso3(getSkewSymMatFromVec3(axis_angle))
    R_delta = expMapso3(getSkewSymMatFromVec3(omega * dt))
    R_next = torch.matmul(R_delta, R_curr)
    axis_angle_next = getVec3FromSkewSymMat(logMapSO3(R_next))
    return axis_angle_next


def computeAngularError(source_axis_angle, target_axis_angle):
    R_source = expMapso3(getSkewSymMatFromVec3(source_axis_angle))
    R_target = expMapso3(getSkewSymMatFromVec3(target_axis_angle))
    R_delta = torch.matmul(R_target, R_source.T)
    angular_error = getVec3FromSkewSymMat(logMapSO3(R_delta))
    return angular_error


def convertAxisAngleToQuaternion(axis_angle, epsilon=1.0e-5):
    if not torch.is_tensor(axis_angle):
        axis_angle = torch.Tensor(axis_angle)
    assert axis_angle.shape[0] == 3
    angle = torch.norm(axis_angle)
    if angle > epsilon:
        axis = axis_angle / angle
        quat = axis_angle.new_zeros(4)
        quat[:3] = axis * torch.sin(angle / 2.0)
        quat[3] = torch.cos(angle / 2.0)
    else:
        quat = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=axis_angle.device, dtype=axis_angle.dtype
        )
    quat = quat / torch.norm(quat)
    return quat


def convertQuaternionToAxisAngle(quat, alpha=0.05, epsilon=1.0e-15):
    if not torch.is_tensor(quat):
        quat = torch.Tensor(quat)
    assert quat.shape[0] == 4
    assert (torch.norm(quat) > 1.0 - alpha) and (torch.norm(quat) < 1.0 + alpha)
    quat = quat / torch.norm(quat)
    angle = 2.0 * torch.acos(quat[3])
    axis = quat[:3] / (torch.sin(angle / 2.0) + epsilon)
    axis_angle = axis * angle
    return axis_angle


def getSkewSymMatFromVec3(omega):
    omega = omega.reshape(3)
    omegahat = omega.new_zeros((3, 3))
    sign_multiplier = -1
    for i in range(3):
        for j in range(i + 1, 3):
            omegahat[i, j] = sign_multiplier * omega[3 - i - j]
            omegahat[j, i] = -sign_multiplier * omega[3 - i - j]
            sign_multiplier = -sign_multiplier
    return omegahat


def getVec3FromSkewSymMat(omegahat, epsilon=1.0e-14):
    assert torch.norm(torch.diag(omegahat)) < assert_epsilon, (
        "omegahat = \n%s" % omegahat
    )
    for i in range(3):
        for j in range(i + 1, 3):
            v1 = omegahat[i, j]
            v2 = omegahat[j, i]
            err = torch.abs(v1 + v2)
            assert err < epsilon, "err = %f >= %f = epsilon" % (err, epsilon)
    omega = omegahat.new_zeros(3)
    omega[0] = 0.5 * (omegahat[2, 1] - omegahat[1, 2])
    omega[1] = 0.5 * (omegahat[0, 2] - omegahat[2, 0])
    omega[2] = 0.5 * (omegahat[1, 0] - omegahat[0, 1])
    return omega


def getKseehatFromWrench(wrench):
    assert wrench.shape[0] == 6
    v = wrench[:3]
    omega = wrench[3:6]
    omegahat = getSkewSymMatFromVec3(omega)
    kseehat = wrench.new_zeros((4, 4))
    kseehat[:3, :3] = omegahat
    kseehat[:3, 3] = v
    return kseehat


def getWrenchFromKseehat(kseehat, epsilon=1.0e-14):
    assert torch.norm(kseehat[3, :]) < assert_epsilon, "kseehat = \n%s" % kseehat
    v = kseehat[:3, 3].reshape((3, 1))
    omegahat = kseehat[:3, :3]
    omega = getVec3FromSkewSymMat(omegahat, epsilon).reshape((3, 1))
    wrench = torch.stack([v, omega])
    assert wrench.shape[0] == 6, "wrench.shape[0] = %d" % wrench.shape[0]
    return wrench.reshape((6,))


def getHomogeneousTransformMatrixFromAxes(orig, axis_x, axis_y, axis_z):
    T = torch.eye(4)
    T[:3, 0] = axis_x
    T[:3, 1] = axis_y
    T[:3, 2] = axis_z
    T[:3, 3] = orig
    return T


def getAxesFromHomogeneousTransformMatrix(T):
    assert torch.norm(T[3, :3]) < assert_epsilon
    assert torch.abs(T[3, 3] - 1.0) < assert_epsilon

    axis_x = T[:3, 0]
    axis_y = T[:3, 1]
    axis_z = T[:3, 2]
    orig = T[:3, 3]

    return orig, axis_x, axis_y, axis_z


def getInverseHomogeneousTransformMatrix(T, epsilon=1.0e-14):
    assert torch.norm(T[3, :3]) < assert_epsilon
    assert torch.abs(T[3, 3] - 1.0) < assert_epsilon
    R = T[:3, :3]
    assert (
        torch.abs(torch.abs(torch.det(R)) - 1.0) < assert_epsilon
    ), "det(R) = %f" % torch.det(R)
    p = T[:3, 3]
    Tinv = torch.eye(4, device=T.device, dtype=T.dtype)
    Rinv = R.T
    pinv = -torch.matmul(Rinv, p)
    Tinv[:3, :3] = Rinv
    Tinv[:3, 3] = pinv
    return Tinv


def logMapSO3(R, epsilon=1.0e-14):
    assert R.shape[0] == 3
    assert R.shape[1] == 3
    assert (
        torch.abs(torch.abs(torch.det(R)) - 1.0) < assert_epsilon
    ), "det(R) = %f" % torch.det(R)
    half_traceR_minus_one = (torch.trace(R) - 1.0) / 2.0
    if half_traceR_minus_one < -R.new_ones(1):
        print("Warning: half_traceR_minus_one = %f < -1.0" % half_traceR_minus_one)
        half_traceR_minus_one = -R.new_ones(1)
    if half_traceR_minus_one > 1.0:
        print("Warning: half_traceR_minus_one = %f > 1.0" % half_traceR_minus_one)
        half_traceR_minus_one = R.new_ones(1)

    theta = torch.acos(half_traceR_minus_one)
    omegahat = (R - R.T) / ((2.0 * torch.sin(theta)) + epsilon)
    return theta * omegahat


def expMapso3(omegahat, epsilon=1.0e-14):
    assert omegahat.shape[0] == 3
    assert omegahat.shape[1] == 3
    omega = getVec3FromSkewSymMat(omegahat, epsilon)

    norm_omega = torch.norm(omega)
    exp_omegahat = (
        torch.eye(3, device=omegahat.device, dtype=omegahat.dtype)
        + ((torch.sin(norm_omega) / (norm_omega + epsilon)) * omegahat)
        + (
            ((1.0 - torch.cos(norm_omega)) / (norm_omega + epsilon) ** 2)
            * torch.matmul(omegahat, omegahat)
        )
    )
    return exp_omegahat


def logMapSE3(T, epsilon=1.0e-14):
    assert T.shape[0] == 4
    assert T.shape[1] == 4
    assert torch.norm(T[3, :3]) < assert_epsilon
    assert torch.abs(T[3, 3] - 1.0) < assert_epsilon
    R = T[:3, :3]
    omegahat = logMapSO3(R, epsilon)

    omega = getVec3FromSkewSymMat(omegahat, epsilon)
    norm_omega = torch.norm(omega)

    Ainv = (
        torch.eye(3, device=T.device, dtype=T.dtype)
        - (0.5 * omegahat)
        + (
            (
                (
                    (2.0 * torch.sin(norm_omega))
                    - (norm_omega * (1.0 + torch.cos(norm_omega)))
                )
                / ((2 * (norm_omega**2) * torch.sin(norm_omega)) + epsilon)
            )
            * torch.matmul(omegahat, omegahat)
        )
    )
    p = T[:3, 3]
    kseehat = T.new_zeros((4, 4))
    kseehat[:3, :3] = omegahat
    kseehat[:3, 3] = torch.matmul(Ainv, p)
    return kseehat


def expMapse3(kseehat, epsilon=1.0e-14):
    assert kseehat.shape[0] == 4
    assert kseehat.shape[1] == 4
    assert torch.norm(kseehat[3, :]) < assert_epsilon
    omegahat = kseehat[:3, :3]
    exp_omegahat = expMapso3(omegahat, epsilon)

    omega = getVec3FromSkewSymMat(omegahat, epsilon)
    norm_omega = torch.norm(omega)

    A = (
        torch.eye(3, device=kseehat.device, dtype=kseehat.dtype)
        + (((1.0 - torch.cos(norm_omega)) / (norm_omega + epsilon) ** 2) * omegahat)
        + (
            ((norm_omega - torch.sin(norm_omega)) / ((norm_omega + epsilon) ** 3))
            * torch.matmul(omegahat, omegahat)
        )
    )
    v = kseehat[:3, 3]
    exp_kseehat = torch.eye(4, device=kseehat.device, dtype=kseehat.dtype)
    exp_kseehat[:3, :3] = exp_omegahat
    exp_kseehat[:3, 3] = torch.matmul(A, v)
    return exp_kseehat
