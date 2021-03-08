# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import os
import torch
import random
from torch.utils.data import DataLoader

from hydra.experimental import compose as hydra_compose
from hydra.experimental import initialize_config_dir

import differentiable_robot_model
from differentiable_robot_model.differentiable_robot_model import DifferentiableRobotModel, DifferentiableKUKAiiwa
from differentiable_robot_model.data_generation_utils import generate_sine_motion_inverse_dynamics_data
import diff_robot_data

torch.set_printoptions(precision=3, sci_mode=False)

random.seed(0)
np.random.seed(1)
torch.manual_seed(0)


abs_config_dir=os.path.abspath(os.path.join(differentiable_robot_model.__path__[0], "../conf"))
with initialize_config_dir(config_dir=abs_config_dir):
    learnable_robot_model_cfg = hydra_compose(config_name="torch_robot_model_learnable_l4dc_constraints.yaml")


# ground truth robot model (with known kinematics and dynamics parameters) - used to generate data
gt_robot_model = DifferentiableKUKAiiwa()
gt_robot_model.print_link_names()

# learnable robot model
urdf_path = os.path.join(diff_robot_data.__path__[0], learnable_robot_model_cfg.model.rel_urdf_path)
learnable_robot_model = DifferentiableRobotModel(urdf_path,
                                                 learnable_robot_model_cfg.model.learnable_rigid_body_config,
                                                 learnable_robot_model_cfg.model.name)
q = torch.zeros((1, 7))
qd = torch.zeros((1, 7))
tau = torch.zeros((1, 7))

qdd = gt_robot_model.compute_forward_dynamics_old(q=q, qd=qd, f=tau)
qdd2 = gt_robot_model.compute_forward_dynamics(q=q, qd=qd, f=tau)

print(f"qdd: {qdd}")
print(f"qdd: {qdd2}")

