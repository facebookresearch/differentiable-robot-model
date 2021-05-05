# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import os
import torch
import random
from torch.utils.data import DataLoader

# potential mass parametrizations
from differentiable_robot_model.rigid_body_params import UnconstrainedMassValue, PositiveMassValue

# potential inertia matrix parametrizations
from differentiable_robot_model.rigid_body_params import (InertiaMatrix3DNoStructureNet,
                                                          CovParameterized3DInertiaMatrixNet,
                                                          Symm3DInertiaMatrixNet,
                                                          SymmPosDef3DInertiaMatrixNet,
                                                          TriangParam3DInertiaMatrixNet)
# potential center of mass parametrizations
from differentiable_robot_model.rigid_body_params import MCoM3DNet

import differentiable_robot_model
from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel,
    DifferentiableKUKAiiwa,
)
from differentiable_robot_model.data_utils import (
    generate_sine_motion_inverse_dynamics_data,
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


def run(n_epochs=10, n_data=1000, device="cpu"):

    """ setup learnable robot model """
    learnable_model_cfg = {}
    # add all links that have a learnable component, use urdf link name
    # any link that is not specified as learnable will be initialized from urdf
    learnable_model_cfg['learnable_links'] = ['iiwa_link_1', 'iiwa_link_2']
    learnable_params = {}
    learnable_params['mass'] = {'module': UnconstrainedMassValue}
    learnable_params['com'] = {'module': MCoM3DNet}
    learnable_params['inertia_mat'] = {'module': InertiaMatrix3DNoStructureNet}
    learnable_model_cfg['learnable_params'] = learnable_params

    urdf_path = os.path.join(
        diff_robot_data.__path__[0], "kuka_iiwa/urdf/iiwa7.urdf"
    )

    learnable_robot_model = DifferentiableRobotModel(
        urdf_path,
        learnable_model_cfg,
        "kuka_iiwa",
        device=device
    )

    """ generate training data via ground truth model """
    # ground truth robot model (with known kinematics and dynamics parameters) - used to generate data
    gt_robot_model = DifferentiableKUKAiiwa(device=device)
    gt_robot_model.print_link_names()

    train_data = generate_sine_motion_inverse_dynamics_data(
        gt_robot_model, n_data=n_data, dt=1.0 / 250.0, freq=0.05
    )
    train_loader = DataLoader(dataset=train_data, batch_size=100, shuffle=False)

    """ optimize learnable params """
    optimizer = torch.optim.Adam(learnable_robot_model.parameters(), lr=1e-2)
    loss_fn = NMSELoss(train_data.var())
    for i in range(n_epochs):
        losses = []
        for batch_idx, batch_data in enumerate(train_loader):
            q, qd, qdd_des, gt_tau = batch_data
            optimizer.zero_grad()
            tau_pred = learnable_robot_model.compute_inverse_dynamics(
                q=q, qd=qd, qdd_des=qdd_des, include_gravity=True
            )
            loss = loss_fn(tau_pred, gt_tau)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"i: {i} loss: {np.mean(losses)}")

    learnable_robot_model.print_learnable_params()


if __name__ == "__main__":
    run()
