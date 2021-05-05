# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import random
import os

from hydra.experimental import compose as hydra_compose
from hydra.experimental import initialize_config_dir

from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel,
    DifferentiableTwoLinkRobot,
)
from differentiable_robot_model.data_utils import (
    generate_random_forward_kinematics_data,
)
import differentiable_robot_model
import diff_robot_data

torch.set_printoptions(precision=3, sci_mode=False)

random.seed(0)
np.random.seed(1)
torch.manual_seed(0)


def run(n_epochs=3000, n_data=100, device="cpu"):
    rel_urdf_path = "2link_robot.urdf"
    robot_description_folder = diff_robot_data.__path__[0]
    urdf_path = os.path.join(robot_description_folder, rel_urdf_path)

    # set up learnable params
    learnable_params = {}
    learnable_params["trans"] = {}

    # specify learnable model details
    # add all links that have a learnable component, use urdf link name
    # any link that is not specified as learnable will be initialized from urdf
    learnable_model_cfg = {}
    learnable_model_cfg["learnable_links"] = ["arm1", "arm2"]
    learnable_model_cfg["learnable_params"] = learnable_params
    learnable_robot_model = DifferentiableRobotModel(
        urdf_path=urdf_path,
        learnable_rigid_body_config=learnable_model_cfg,
        name="2link",
        device=device,
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
        if i % 100 == 0:
            print(f"i: {i}, loss: {loss}")
            learnable_robot_model.print_learnable_params()
        loss.backward()
        optimizer.step()

    print("parameters of the ground truth model (that's what we ideally learn)")
    print("gt trans param: {}".format(gt_robot_model._bodies[1].trans))
    print("gt trans param: {}".format(gt_robot_model._bodies[2].trans))

    print("parameters of the optimized learnable robot model")
    learnable_robot_model.print_learnable_params()


if __name__ == "__main__":
    run()
