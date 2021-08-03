import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "..")

# potential mass parametrizations
from differentiable_robot_model.rigid_body_params import (
    PositiveScalar,
    UnconstrainedTensor,
)

from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel,
    DifferentiableKUKAiiwa,
)
from differentiable_robot_model.data_utils import (
    generate_sine_motion_forward_dynamics_data,
)
import diff_robot_data

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


class DifferentiableJaco(DifferentiableRobotModel):
    def __init__(self, device=None, **kwargs):
        rel_urdf_path = "jaco.urdf"
        self.urdf_path = os.path.join(rel_urdf_path)
        self.learnable_rigid_body_config = None
        self.name = "differentiable_kinova_jaco"
        super().__init__(
            self.urdf_path,
            self.name,
            device=device,
            **kwargs,
        )


def run(n_epochs=100, n_data=10000, device="cpu"):
    """setup learnable robot model"""

    robot_type = "jaco"
    # robot_type = 'iiwa'

    if robot_type == "iiwa":
        urdf_path = os.path.join(
            diff_robot_data.__path__[0], "kuka_iiwa/urdf/iiwa7.urdf"
        )
        link_to_learn = "iiwa_link_1"
        gt_robot_model = DifferentiableKUKAiiwa(device=device)
    elif robot_type == "jaco":
        urdf_path = os.path.join("jaco.urdf")
        link_to_learn = "j2n6s300_link_1"
        gt_robot_model = DifferentiableJaco(device=device)

    learnable_robot_model = DifferentiableRobotModel(
        urdf_path,
        "jaco",
        device=device,
    )

    learnable_robot_model.make_link_param_learnable(
        link_to_learn, "mass", PositiveScalar()
    )
    learnable_robot_model.make_link_param_learnable(
        link_to_learn, "com", UnconstrainedTensor(dim1=1, dim2=3)
    )
    learnable_robot_model.make_link_param_learnable(
        link_to_learn, "inertia_mat", UnconstrainedTensor(dim1=3, dim2=3)
    )

    """ generate training data via ground truth model """

    torch.autograd.set_detect_anomaly(True)

    train_data = generate_sine_motion_forward_dynamics_data(
        gt_robot_model, n_data=n_data, dt=1.0 / 250.0, freq=0.1
    )

    train_loader = DataLoader(dataset=train_data, batch_size=100, shuffle=False)

    """ optimize learnable params """
    optimizer = torch.optim.Adam(learnable_robot_model.parameters(), lr=1e-2)
    loss_fn = NMSELoss(train_data.var())
    for i in range(n_epochs):
        losses = []
        for batch_idx, batch_data in enumerate(train_loader):
            q, qd, qdd, tau = batch_data
            optimizer.zero_grad()
            qdd_pred = learnable_robot_model.compute_forward_dynamics(
                q=q, qd=qd, f=tau, include_gravity=True, use_damping=True
            )
            loss = loss_fn(qdd_pred, qdd)
            loss.backward(retain_graph=True)
            optimizer.step()
            losses.append(loss.item())

        print(f"i: {i} loss: {np.mean(losses)}")

    learnable_robot_model.print_learnable_params()


if __name__ == "__main__":
    run()
