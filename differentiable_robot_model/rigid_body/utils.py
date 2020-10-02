# Copyright (c) Facebook, Inc. and its affiliates.

import random
from contextlib import contextmanager

import numpy as np
import timeit
import torch
import operator
from functools import reduce


prod = lambda l: reduce(operator.mul, l, 1)
torch.set_default_tensor_type(torch.DoubleTensor)

def cross_product(vec3a, vec3b):
    vec3a = convert_into_at_least_2d_pytorch_tensor(vec3a)
    vec3b = convert_into_at_least_2d_pytorch_tensor(vec3b)
    skew_symm_mat_a = vector3_to_skew_symm_matrix(vec3a)
    return (skew_symm_mat_a @ vec3b.unsqueeze(2)).squeeze(2)


def bfill_lowertriangle(A: torch.Tensor, vec: torch.Tensor):
    ii, jj = np.tril_indices(A.size(-2), k=-1, m=A.size(-1))
    A[..., ii, jj] = vec
    return A


def bfill_diagonal(A: torch.Tensor, vec: torch.Tensor):
    ii, jj = np.diag_indices(min(A.size(-2), A.size(-1)))
    A[..., ii, jj] = vec
    return A


def vector3_to_skew_symm_matrix(vec3):
    vec3 = convert_into_at_least_2d_pytorch_tensor(vec3)
    batch_size = vec3.shape[0]
    skew_symm_mat = vec3.new_zeros((batch_size, 3, 3))
    skew_symm_mat[:, 0, 1] = -vec3[:, 2]
    skew_symm_mat[:, 0, 2] = vec3[:, 1]
    skew_symm_mat[:, 1, 0] = vec3[:, 2]
    skew_symm_mat[:, 1, 2] = -vec3[:, 0]
    skew_symm_mat[:, 2, 0] = -vec3[:, 1]
    skew_symm_mat[:, 2, 1] = vec3[:, 0]
    return skew_symm_mat


def torch_square(x):
    return x * x


def exp_map_so3(omega, epsilon=1.0e-14):
    omegahat = vector3_to_skew_symm_matrix(omega).squeeze()

    norm_omega = torch.norm(omega, p=2)
    exp_omegahat = (torch.eye(3) +
                    ((torch.sin(norm_omega) / (norm_omega + epsilon)) * omegahat) +
                    (((1.0 - torch.cos(norm_omega)) / (torch_square(norm_omega + epsilon))) *
                     (omegahat @ omegahat))
                    )
    return exp_omegahat


def set_rng_seed(rng_seed: int) -> None:
    random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)


def move_optimizer_to_gpu(optimizer):
    """
    Move the optimizer state to GPU, if necessary.
    After calling, any parameter specific state in the optimizer
    will be located on the same device as the parameter.
    """
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.is_cuda:
                param_state = optimizer.state[param]
                for k in param_state.keys():
                    if isinstance(param_state[k], torch.Tensor):
                        param_state[k] = param_state[k].cuda(device=param.get_device())


def require_and_zero_grads(vs):
    for v in vs:
        v.requires_grad_(True)
        try:
            v.grad.zero_()
        except AttributeError:
            pass


def compute_mse_per_dim(prediction, ground_truth, data_axis=0):
    mse_per_dim = ((prediction - ground_truth) ** 2).mean(axis=data_axis)
    return mse_per_dim


def compute_mse_loss(prediction, ground_truth, data_axis=0):
    mse_per_dim = compute_mse_per_dim(prediction, ground_truth, data_axis)
    return mse_per_dim.mean()


def compute_mse_var_nmse_per_dim(prediction, ground_truth, data_axis=0, is_adding_regularizer=True, save_filepath=None):
    mse_per_dim = compute_mse_per_dim(prediction, ground_truth, data_axis)
    var_ground_truth_per_dim = ground_truth.var(axis=data_axis)
    if (is_adding_regularizer):
        reg = 1.0e-14
    else:
        reg = 0.0
    nmse_per_dim = mse_per_dim / (var_ground_truth_per_dim + reg)
    if save_filepath is not None:
        from fair_robot_envs.env.utils.pyplot_util import subplot_ND

        traj_list = [prediction, ground_truth]
        subplot_ND(NDtraj_list=traj_list,
                   title='prediction_vs_ground_truth',
                   Y_label_list=['dim %d' % i for i in range(prediction.shape[1])],
                   fig_num=0,
                   label_list=['prediction', 'ground_truth'],
                   is_auto_line_coloring_and_styling=True,
                   save_filepath=save_filepath,
                   X_label='time index')
    return mse_per_dim, var_ground_truth_per_dim, nmse_per_dim


def compute_nmse_per_dim(prediction, ground_truth, data_axis=0, is_adding_regularizer=True, save_filepath=None):
    [_, _, nmse_per_dim] = compute_mse_var_nmse_per_dim(prediction, ground_truth,
                                                        data_axis, is_adding_regularizer, save_filepath)
    return nmse_per_dim


def compute_nmse_loss(prediction, ground_truth, data_axis=0, is_adding_regularizer=True):
    nmse_per_dim = compute_nmse_per_dim(prediction, ground_truth, data_axis, is_adding_regularizer)
    return nmse_per_dim.mean()


def convert_into_pytorch_tensor(variable):
    if isinstance(variable, torch.Tensor):
        return variable
    elif isinstance(variable, np.ndarray):
        return torch.Tensor(variable)
    else:
        return torch.Tensor(variable)


def convert_into_at_least_2d_pytorch_tensor(variable):
    tensor_var = convert_into_pytorch_tensor(variable)
    if len(tensor_var.shape) == 1:
        return tensor_var.unsqueeze(0)
    else:
        return tensor_var


def list2dict(list_dict):
    dict_list = {key: [] for key in list_dict[0]}
    for element in list_dict:
        for key in element.keys():
            dict_list[key].append(element[key])
    return dict_list


def extract_inv_dyn_dataset(dataset):
    inv_dyn_dataset = dict()
    T = dataset['joint_vel'].shape[0]
    inv_dyn_dataset['N_data'] = T-1

    keys = ['joint_pos', 'joint_vel', 'joint_acc', 'torque_applied']
    for key in keys:
        inv_dyn_dataset[key] = dataset[key][:(T-1), :]
        assert(inv_dyn_dataset[key].shape[0] == T-1)

    next_keys = ['joint_pos', 'joint_vel', 'joint_acc']
    for key in next_keys:
        inv_dyn_dataset['next_' + key] = dataset[key][1:, :]
        assert(inv_dyn_dataset['next_' + key].shape[0] == T-1)
    return inv_dyn_dataset


def combine_datasets(dataset_list):
    combined_dataset = dict()
    N_data = 0
    dt = None
    keys = dataset_list[0].keys()
    for dataset in dataset_list:
        for key in keys:
            if key == 'N_data':
                N_data += dataset['N_data']
            elif key == 'dt':
                if dt is None:
                    dt = dataset['dt']
                else:
                    assert (dt == dataset['dt'])
            else:
                if key not in combined_dataset.keys():
                    combined_dataset[key] = list()
                combined_dataset[key].append(dataset[key])
    for key in keys:
        if key == 'N_data':
            combined_dataset['N_data'] = N_data
        elif key == 'dt':
            combined_dataset['dt'] = dt
        else:
            combined_dataset[key] = np.concatenate(combined_dataset[key])
            if 'N_data' in keys:
                assert combined_dataset[key].shape[0] == N_data
    return combined_dataset


def plot_grad_flow(viz, env, named_parameters, window_name="grad_bar"):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    num_nan_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            if p.grad is not None:
                assert p.grad is not None, "Layer " + str(n) + " does not have any gradients!!!"
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
                num_nan_grads.append(torch.sum(torch.isnan(p.grad.abs()) | torch.isinf(p.grad.abs()))/prod(list(p.grad.abs().shape)))
    if len(layers) < 2:
        dictionary = dict(
            stacked=False,
            legend=['max-gradient', 'mean-gradient']
        )
    else:
        dictionary = dict(
            stacked=False,
            legend=['max-gradient', 'mean-gradient'],
            rownames=layers,
            title=window_name
        )
    viz.bar(
        X=np.array([max_grads, ave_grads]).transpose([1, 0]),
        env=env,
        win=window_name,
        opts=dictionary
    )
    return torch.stack(max_grads), torch.stack(ave_grads), torch.stack(num_nan_grads), layers


@contextmanager
def temp_require_grad(vs):
    prev_grad_status = [v.requires_grad for v in vs]
    require_and_zero_grads(vs)
    yield
    for v, status in zip(vs, prev_grad_status):
        v.requires_grad_(status)


class Timer(object):

    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self

    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start
