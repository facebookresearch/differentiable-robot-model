import numpy as np
import torch
import os
import time

from differentiable_robot_model.robot_model import (
    DifferentiableFrankaPanda,
)
from simple_sim_wrapper import SimplePybulletWrapperPositionControl
import diff_robot_data


class KinematicTrajectoryOptimizer(torch.nn.Module):
    def __init__(self, robot_model, link_name, n_dofs, time_horizon):
        super().__init__()
        self._robot_model = robot_model
        self._time_horizon = time_horizon
        self._n_dofs = n_dofs
        limits_per_joint = robot_model.get_joint_limits()
        joint_lower_bounds = [joint["lower"] for joint in limits_per_joint]
        joint_upper_bounds = [joint["upper"] for joint in limits_per_joint]
        self._joint_limits_min = torch.tensor(joint_lower_bounds)
        self._joint_limits_max = torch.tensor(joint_upper_bounds)
        self._action_seq = torch.nn.Parameter(torch.zeros(time_horizon, n_dofs))
        self._link_name = link_name

    def forward(self, joint_state, actions=0):
        next_joint_state = joint_state + actions
        next_joint_state = torch.where(
            next_joint_state > self._joint_limits_max,
            self._joint_limits_max,
            next_joint_state,
        )
        next_joint_state = torch.where(
            next_joint_state < self._joint_limits_min,
            self._joint_limits_min,
            next_joint_state,
        )
        ee_pos, _ = self._robot_model.compute_forward_kinematics(
            next_joint_state.reshape(1, 7), self._link_name
        )
        return next_joint_state, ee_pos.squeeze()

    def rollout(self, start_joint_state):
        joint_state_traj = torch.zeros(self._time_horizon, self._n_dofs)
        ee_state_traj = torch.zeros(self._time_horizon, 3)
        joint_state, ee_state = self.forward(start_joint_state)
        joint_state_traj[0, :] = start_joint_state
        ee_state_traj[0, :] = ee_state
        for t in range(self._time_horizon - 1):
            action_param = self._action_seq[t]
            joint_state, ee_state = self.forward(joint_state.detach(), action_param)
            joint_state_traj[t + 1, :] = joint_state
            ee_state_traj[t + 1, :] = ee_state
        return ee_state_traj, joint_state_traj

    def reset_actions(self):
        self._action_seq.data = torch.Tensor(np.zeros([self.time_horizon, 7]))


class DenseGoalCost(torch.nn.Module):
    def __init__(self):
        super(DenseGoalCost, self).__init__()

    def forward(self, goal_state, trajectory):
        traj_dist_from_goal = (100 * (trajectory - goal_state)) ** 2
        return traj_dist_from_goal.mean()


rel_urdf_path = "panda_description/urdf/panda_no_gripper.urdf"
urdf_path = os.path.join(diff_robot_data.__path__[0], rel_urdf_path)

robot_model = DifferentiableFrankaPanda()
traj_wrapper = KinematicTrajectoryOptimizer(
    robot_model=robot_model,
    link_name="panda_virtual_ee_link",
    n_dofs=7,
    time_horizon=20,
)
goal_ee_pose, _ = robot_model.compute_forward_kinematics(
    q=torch.tensor([[0.0] * 7]), link_name="panda_virtual_ee_link"
)

""" optimize trajectory to move to desired endeffector goal pose"""
start_joint_config = [0.0, 0.0, 0.0, -1.5, 0.0, 1.6, 0.0]
start_joint_config = torch.tensor(start_joint_config)
action_optimizer = torch.optim.Adam(traj_wrapper.parameters(), lr=1e-3)
cost_fn = DenseGoalCost()
for i in range(100):
    action_optimizer.zero_grad()

    ee_state_traj, _ = traj_wrapper.rollout(start_joint_config)
    cost = cost_fn(goal_ee_pose, ee_state_traj)
    cost.backward()
    action_optimizer.step()
    print(f"cost: {cost.item()}")

""" execute optimized action trajectory in pybullet open-loop"""
action_seq = traj_wrapper._action_seq.clone().detach()
des_joint_state = start_joint_config.numpy()

# sim is only needed to visualize results
sim = SimplePybulletWrapperPositionControl(
    rel_urdf_path=rel_urdf_path, controlled_joints=range(7), GUI=True
)
sim.reset_joint_state(start_joint_config.numpy())
for i in range(20):
    des_joint_state = des_joint_state + action_seq[i].numpy()
    sim.step(des_joint_state)
    time.sleep(0.3)

# final endeffector position:
ee_pos, ee_rot = sim.get_link_state(link_id=7)

# final distance to desired endeffector goal
print("__________________________________________________")
print(f"final dist to goal: {np.asarray(ee_pos) - goal_ee_pose.squeeze().numpy()}")
print("__________________________________________________")
