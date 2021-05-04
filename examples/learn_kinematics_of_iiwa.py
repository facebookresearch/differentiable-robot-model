# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import random
import os

from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel,
    DifferentiableKUKAiiwa,
)

from differentiable_robot_model.rigid_body_params import UnconstrainedMassValue
from differentiable_robot_model.data_utils import (
    generate_random_forward_kinematics_data,
)
import diff_robot_data

torch.set_printoptions(precision=3, sci_mode=False)
random.seed(0)
np.random.seed(1)
torch.manual_seed(0)
torch.set_default_tensor_type(torch.FloatTensor)

gt_robot_model = DifferentiableKUKAiiwa()
urdf_path = os.path.join(
    diff_robot_data.__path__[0], "kuka_iiwa/urdf/iiwa7.urdf"
)

learnable_model_cfg = {}
# add all links that have a learnable component, use urdf link name
learnable_model_cfg['learnable_links'] = ['iiwa_link_1', 'iiwa_link_2']
learnable_params = {}
learnable_params['mass'] = {'module': UnconstrainedMassValue}
learnable_model_cfg['learnable_params'] = learnable_params

learnable_robot_model = DifferentiableRobotModel(
    urdf_path,
    learnable_model_cfg,
    "kuka_iiwa",
)

train_data = generate_random_forward_kinematics_data(
    gt_robot_model, n_data=100, ee_name="iiwa_link_ee"
)
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
