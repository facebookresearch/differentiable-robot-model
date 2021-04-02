# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import os
import torch
import random
from torch.utils.data import DataLoader
import time


from hydra.experimental import compose as hydra_compose
from hydra.experimental import initialize_config_dir

import differentiable_robot_model
from differentiable_robot_model.differentiable_robot_model import (
    DifferentiableRobotModel,
    DifferentiableKUKAiiwa,
)
from differentiable_robot_model.data_generation_utils import (
    generate_sine_motion_forward_dynamics_data,
)
import diff_robot_data

torch.set_printoptions(precision=3, sci_mode=False)

random.seed(0)
np.random.seed(1)
torch.manual_seed(0)


class NMSELoss(torch.nn.Module):
    def __init__(self, var):
        super(NMSELoss, self).__init__()
        self.var = var

    def forward(self, yp, yt):
        err = (yp - yt) ** 2
        werr = err / self.var
        return werr.mean()


abs_config_dir = os.path.abspath(
    os.path.join(differentiable_robot_model.__path__[0], "../conf")
)
with initialize_config_dir(config_dir=abs_config_dir):
    learnable_robot_model_cfg = hydra_compose(
        config_name="torch_robot_model_learnable_l4dc_constraints.yaml"
    )


# ground truth robot model (with known kinematics and dynamics parameters) - used to generate data
gt_robot_model = DifferentiableKUKAiiwa()
gt_robot_model.print_link_names()

# learnable robot model
urdf_path = os.path.join(
    diff_robot_data.__path__[0], learnable_robot_model_cfg.model.rel_urdf_path
)
learnable_robot_model = DifferentiableRobotModel(
    urdf_path,
    learnable_robot_model_cfg.model.learnable_rigid_body_config,
    learnable_robot_model_cfg.model.name,
)

tau = torch.zeros((1, 7))  # .repeat((10, 1))
q = torch.Tensor(
    [[-0.492, -0.828, -1.862, 0.163, -1.754, 0.714, -2.197]]
)  # .repeat((10, 1))
qd = torch.Tensor(
    [[4.406, -7.065, -3.089, -1.616, 7.562, -1.654, -6.038]]
)  # .repeat((10, 1))

qdd = gt_robot_model.compute_forward_dynamics(q=q, qd=qd, f=tau, use_damping=True)
tau_pred = gt_robot_model.compute_inverse_dynamics(q=q, qd=qd, qdd_des=qdd)

train_data = generate_sine_motion_forward_dynamics_data(
    gt_robot_model, n_data=10000, dt=1.0 / 250.0, freq=0.1
)
train_loader = DataLoader(dataset=train_data, batch_size=100, shuffle=False)

optimizer = torch.optim.Adam(learnable_robot_model.parameters(), lr=1e-2)
loss_fn = NMSELoss(train_data.var())
for i in range(100):
    losses = []
    for batch_idx, batch_data in enumerate(train_loader):
        q, qd, qdd, tau = batch_data
        optimizer.zero_grad()
        qdd_pred = learnable_robot_model.compute_forward_dynamics(
            q=q, qd=qd, f=tau, include_gravity=True, use_damping=True
        )
        loss = loss_fn(qdd_pred, qdd)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"i: {i} loss: {np.mean(losses)}")

learnable_robot_model.print_learnable_params()
