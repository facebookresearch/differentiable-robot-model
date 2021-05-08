# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import random
import os

from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel,
    DifferentiableKUKAiiwa,
)

from differentiable_robot_model.rigid_body_params import UnconstrainedTensor

from differentiable_robot_model.data_utils import (
    generate_random_forward_kinematics_data,
)
import diff_robot_data

torch.set_printoptions(precision=3, sci_mode=False)
random.seed(0)
np.random.seed(1)
torch.manual_seed(0)


def run(n_epochs=3000, n_data=100, device="cpu"):

    """setup learnable robot model"""

    urdf_path = os.path.join(diff_robot_data.__path__[0], "kuka_iiwa/urdf/iiwa7.urdf")
    learnable_robot_model = DifferentiableRobotModel(
        urdf_path, "kuka_iiwa", device=device
    )
    learnable_robot_model.make_link_param_learnable(
        "iiwa_link_1", "trans", UnconstrainedTensor(dim1=1, dim2=3)
    )
    learnable_robot_model.make_link_param_learnable(
        "iiwa_link_1", "rot_angles", UnconstrainedTensor(dim1=1, dim2=3)
    )

    """ generate training data via ground truth model """
    gt_robot_model = DifferentiableKUKAiiwa(device=device)
    train_data = generate_random_forward_kinematics_data(
        gt_robot_model, n_data=n_data, ee_name="iiwa_link_ee"
    )
    q = train_data["q"]
    gt_ee_pos = train_data["ee_pos"]

    """ optimize learnable params """
    optimizer = torch.optim.Adam(learnable_robot_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    for i in range(n_epochs):
        optimizer.zero_grad()
        ee_pos_pred, _ = learnable_robot_model.compute_forward_kinematics(
            q=q, link_name="iiwa_link_ee"
        )
        loss = loss_fn(ee_pos_pred, gt_ee_pos)
        if i % 100 == 0:
            print(f"i: {i}, loss: {loss}")
            learnable_robot_model.print_learnable_params()
        loss.backward()
        optimizer.step()

    print("parameters of the ground truth model (that's what we ideally learn)")
    print("gt trans param: {}".format(gt_robot_model._bodies[6].trans))
    print("gt trans param: {}".format(gt_robot_model._bodies[7].trans))

    print("parameters of the optimized learnable robot model")
    learnable_robot_model.print_learnable_params()


if __name__ == "__main__":
    run()
