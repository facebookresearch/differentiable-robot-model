# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import random
import os

from hydra.experimental import compose as hydra_compose
from hydra.experimental import initialize_config_dir

from differentiable_robot_model.differentiable_robot_model import DifferentiableRobotModel
from differentiable_robot_model.data_generation_utils import generate_random_forward_kinematics_data

torch.set_printoptions(precision=3, sci_mode=False)
random.seed(0)
np.random.seed(1)
torch.manual_seed(0)


abs_config_dir = os.path.abspath("../conf")
# we load configurations for a ground truth robot , and a learnable robot model
with initialize_config_dir(config_dir=abs_config_dir):
    gt_robot_model_cfg = hydra_compose(config_name="torch_robot_model_gt.yaml")
    # which parameters are learnable is specified in the config file
    learnable_robot_model_cfg = hydra_compose(
        config_name="torch_robot_model_learnable_kinematics.yaml"
    )


gt_robot_model = DifferentiableRobotModel(**gt_robot_model_cfg.model)
learnable_robot_model = DifferentiableRobotModel(**learnable_robot_model_cfg.model)

train_data = generate_random_forward_kinematics_data(gt_robot_model, n_data=100)
q = train_data["q"]
gt_ee_pos = train_data["ee_pos"]

optimizer = torch.optim.Adam(learnable_robot_model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()
for i in range(3000):
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
