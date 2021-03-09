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
from differentiable_robot_model.differentiable_robot_model import DifferentiableRobotModel, DifferentiableKUKAiiwa
from differentiable_robot_model.data_generation_utils import generate_sine_motion_inverse_dynamics_data
import diff_robot_data

torch.set_printoptions(precision=3, sci_mode=False)

random.seed(0)
np.random.seed(1)
torch.manual_seed(0)

def sample_test_case(robot_model, zero_vel=False, zero_acc=False):
    limits_per_joint = robot_model.get_joint_limits()
    joint_lower_bounds = [joint["lower"] for joint in limits_per_joint]
    joint_upper_bounds = [joint["upper"] for joint in limits_per_joint]
    joint_velocity_limits = [joint["velocity"] for joint in limits_per_joint]
    joint_angles = []
    joint_velocities = []
    joint_accelerations = []

    for i in range(7):
        joint_angles.append(
            np.random.uniform(low=joint_lower_bounds[i], high=joint_upper_bounds[i])
        )

        if zero_vel:
            joint_velocities.append(0.0)

        else:
            joint_velocities.append(
                np.random.uniform(
                    low=-joint_velocity_limits[i], high=joint_velocity_limits[i]
                )
            )

        if zero_acc:
            joint_accelerations.append(0.0)
        else:
            joint_accelerations.append(
                np.random.uniform(
                    low=-joint_velocity_limits[i] * 2.0,
                    high=joint_velocity_limits[i] * 2.0,
                )
            )

    return {
        "joint_angles": joint_angles,
        "joint_velocities": joint_velocities,
        "joint_accelerations": joint_accelerations,
    }

abs_config_dir=os.path.abspath(os.path.join(differentiable_robot_model.__path__[0], "../conf"))
with initialize_config_dir(config_dir=abs_config_dir):
    learnable_robot_model_cfg = hydra_compose(config_name="torch_robot_model_learnable_l4dc_constraints.yaml")


# ground truth robot model (with known kinematics and dynamics parameters) - used to generate data
gt_robot_model = DifferentiableKUKAiiwa()
gt_robot_model.print_link_names()

# learnable robot model
#urdf_path = os.path.join(diff_robot_data.__path__[0], learnable_robot_model_cfg.model.rel_urdf_path)
#learnable_robot_model = DifferentiableRobotModel(urdf_path,
#                                                 learnable_robot_model_cfg.model.learnable_rigid_body_config,
#                                                 learnable_robot_model_cfg.model.name)
test_case = sample_test_case(gt_robot_model)
#q = torch.zeros((1, 7))
#qd = torch.zeros((1, 7))
#tau = torch.zeros((1, 7))

#q = torch.Tensor(test_case['joint_angles']).reshape((1, 7))
#qd = torch.Tensor(test_case['joint_velocities']).reshape((1, 7))
tau = torch.zeros((1, 7))
q = torch.Tensor([[-0.492, -0.828, -1.862,  0.163, -1.754,  0.714, -2.197]])
qd = torch.Tensor([[ 4.406, -7.065, -3.089, -1.616,  7.562, -1.654, -6.038]])



#start1 = time.process_time()
#for i in range(100):
#    qdd = gt_robot_model.compute_forward_dynamics_old(q=q, qd=qd, f=tau)
#elapsed_time = time.process_time() - start1
#print(elapsed_time)

start2 = time.process_time()
for i in range(100):
    qdd2 = gt_robot_model.compute_forward_dynamics(q=q, qd=qd, f=tau)
elapsed_time = time.process_time() - start2
print(elapsed_time)
# end2 = timeit.timeit()
# print(end2 - start2)

#print(f"qdd: {qdd}")
#print(f"qdd: {qdd2}")

