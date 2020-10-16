# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import numpy as np
from torch.utils.data.dataset import Dataset


class InverseDynamicsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        q = self.data['q'][index, :]
        qd = self.data['qd'][index, :]
        qdd_des = self.data['qdd_des'][index, :]
        tau = self.data['tau'][index, :]
        return [q, qd, qdd_des, tau]

    def __len__(self):
        return self.data['q'].shape[0]

    def var(self):
        return self.data['tau'].var(dim=0)


def generate_random_forward_kinematics_data(robot_model, n_data, ee_name):
    limits_per_joint = robot_model.get_joint_limits()
    ndof = robot_model._n_dofs
    joint_lower_bounds = np.asarray([joint["lower"] for joint in limits_per_joint])
    joint_upper_bounds = np.asarray([joint["upper"] for joint in limits_per_joint])
    q = torch.tensor(
        np.random.uniform(
            low=joint_lower_bounds, high=joint_upper_bounds, size=(n_data, ndof)
        )
    )
    ee_pos, _ = robot_model.compute_forward_kinematics(q=q, link_name=ee_name)

    return {"q": q, "ee_pos": ee_pos}


def generate_random_inverse_dynamics_data(robot_model, n_data):
    limits_per_joint = robot_model.get_joint_limits()
    joint_lower_bounds = np.asarray([joint['lower'] for joint in limits_per_joint])
    joint_upper_bounds = np.asarray([joint['upper'] for joint in limits_per_joint])
    joint_velocity_limits = 0.2*np.asarray([joint['velocity'] for joint in limits_per_joint])
    q = torch.tensor(np.random.uniform(low=joint_lower_bounds, high=joint_upper_bounds, size=(n_data, 7)))
    qd = torch.tensor(np.random.uniform(low=-joint_velocity_limits, high=joint_velocity_limits, size=(n_data, 7)))
    qdd_des = torch.tensor(np.random.uniform(low=-joint_velocity_limits*2.0, high=joint_velocity_limits*2.0, size=(n_data, 7)))

    torques = robot_model.compute_inverse_dynamics(q=q, qd=qd, qdd_des=qdd_des, include_gravity=True)

    return InverseDynamicsDataset(data= {"q": q, "qd": qd, "qdd_des": qdd_des, "tau": torques})


def generate_sine_motion_inverse_dynamics_data(robot_model, n_data, dt, freq):
    A = 0.7
    q_start = torch.zeros(robot_model._n_dofs)
    T = int(n_data*dt)
    t = torch.linspace(0.0, T-1, n_data)
    pi = torch.acos(torch.zeros(1)).item() * 2

    q = A*torch.sin(2*pi*freq*t).reshape(n_data, 1).repeat(1, robot_model._n_dofs) + q_start
    qd = 2*pi*freq*A*torch.cos(2*pi*freq*t).reshape(n_data, 1).repeat(1, robot_model._n_dofs)
    qdd_des = -(2*pi*freq)**2 *A* torch.sin(2*pi*freq*t).reshape(n_data, 1).repeat(1, robot_model._n_dofs)

    torques = robot_model.compute_inverse_dynamics(q=q, qd=qd, qdd_des=qdd_des, include_gravity=True)

    return InverseDynamicsDataset(data= {"q": q, "qd": qd, "qdd_des": qdd_des, "tau": torques})
