# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import random
import os

from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel,
    DifferentiableTwoLinkRobot,
)
from differentiable_robot_model.data_utils import (
    generate_random_forward_kinematics_data,
)

from differentiable_robot_model.rigid_body_params import UnconstrainedTensor
import diff_robot_data

torch.set_printoptions(precision=3, sci_mode=False)

random.seed(0)
np.random.seed(1)
torch.manual_seed(0)


def run(n_epochs=3000, n_data=100, device="cpu"):
    rel_urdf_path = "2link_robot.urdf"
    robot_description_folder = diff_robot_data.__path__[0]
    urdf_path = os.path.join(robot_description_folder, rel_urdf_path)

    learnable_robot_model = DifferentiableRobotModel(
        urdf_path=urdf_path, name="2link", device=device
    )
    learnable_robot_model.make_link_param_learnable(
        "arm1", "trans", UnconstrainedTensor(dim1=1, dim2=3)
    )
    learnable_robot_model.make_link_param_learnable(
        "arm2", "trans", UnconstrainedTensor(dim1=1, dim2=3)
    )

    gt_robot_model = DifferentiableTwoLinkRobot(device=device)
    train_data = generate_random_forward_kinematics_data(
        gt_robot_model, n_data=n_data, ee_name="endEffector"
    )
    q = train_data["q"]
    gt_ee_pos = train_data["ee_pos"]

    optimizer = torch.optim.Adam(learnable_robot_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    for i in range(n_epochs):
        optimizer.zero_grad()
        ee_pos_pred, _ = learnable_robot_model.compute_forward_kinematics(
            q=q, link_name="endEffector"
        )
        loss = loss_fn(ee_pos_pred, gt_ee_pos)
        if i % 10 == 0:
            print(f"i: {i}, loss: {loss}")
            learnable_robot_model.print_learnable_params()

        if i == 10:
            learnable_robot_model.freeze_learnable_link_param(
                link_name="arm1", parameter_name="trans"
            )

        if i == 100:
            learnable_robot_model.unfreeze_learnable_link_param(
                link_name="arm1", parameter_name="trans"
            )
        loss.backward()
        optimizer.step()

    print("parameters of the ground truth model (that's what we ideally learn)")
    print("gt trans param: {}".format(gt_robot_model._bodies[1].trans))
    print("gt trans param: {}".format(gt_robot_model._bodies[2].trans))

    print("parameters of the optimized learnable robot model")
    learnable_robot_model.print_learnable_params()


if __name__ == "__main__":
    run()
