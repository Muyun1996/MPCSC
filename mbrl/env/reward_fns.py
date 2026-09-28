# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F

from . import termination_fns

def kuramoto(act: torch.Tensor, next_obs: torch.Tensor, reward_obj = "order_parameter") -> torch.Tensor:
    """
    Args:
        act:
        next_obs:
        reward_obj: ["variance", "fixed_target", "order_parameter"]
    Returns:
    """
    assert len(next_obs.shape) == len(act.shape) == 2

    batch_state = next_obs
    # Get the orientation of all individuals
    angle_radians = batch_state  # (3000, 50)

    if reward_obj == "variance":
        # measure the degree of synchronization by variance
        reward = - torch.var(next_obs, dim=-1, keepdim=True)
        return reward

    elif reward_obj == "fixed_target":
        zero_mse = F.mse_loss(angle_radians[:, :5], torch.zeros((angle_radians.shape[0], 5)), reduction='none')
        target_value = 2 * torch.tensor([3.141592653589793], dtype=torch.float32)
        target_value = target_value.repeat(angle_radians.shape[0], 5)
        target_mse = F.mse_loss(angle_radians[:, 5:], target_value, reduction='none')
        reward = - torch.mean(target_mse + zero_mse, dim=-1, keepdim=True)
        return reward

    elif reward_obj == "order_parameter":
        # Convert the orientations to vectors on the complex plane
        vectors = torch.stack([torch.cos(angle_radians), torch.sin(angle_radians)], dim=2)  # (3000, 50, 2)
        # Calculate the magnitude of the sum of vectors
        vector_sum = torch.sum(vectors, dim=1)  # (3000, 2)
        vector_sum_length = torch.norm(vector_sum, dim=1)  # (3000)

        node_num = batch_state.shape[1]
        if node_num > 0:
            order_parameter = vector_sum_length / node_num  # (3000)
        else:
            order_parameter = torch.zeros_like(vector_sum_length)
        reward = order_parameter.unsqueeze(1)  # (3000, 1)
        return reward

def sis(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    # measure the Infecious mean
    reward = - torch.mean(next_obs, dim=-1, keepdim=True)
    return reward

def boids(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    # Calculate the Order Parameter
    batch_state = next_obs.reshape(next_obs.shape[0], -1, 5).clone().detach()
    # batch_state : 3000 * 20 * 3

    # Get the orientation of all individuals
    angle_degrees = batch_state[:, :, 4] * 360  # (3000, 20)
    angle_radians = torch.deg2rad(angle_degrees)
    # Convert the orientations to vectors on the complex plane
    vectors = torch.stack([torch.cos(angle_radians), torch.sin(angle_radians)], dim=2)  # (3000, 20, 2)
    # Calculate the magnitude of the sum of vectors
    vector_sum = torch.sum(vectors, dim=1)  # (3000, 2)
    vector_sum_length = torch.norm(vector_sum, dim=1)  # (3000)

    boids_num = batch_state.shape[1]
    if boids_num > 0:
        order_parameter = vector_sum_length / boids_num  # (3000)
    else:
        order_parameter = torch.zeros_like(vector_sum_length)
    rewards = order_parameter.unsqueeze(1)  # (3000, 1)
    return rewards


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.cartpole(act, next_obs)).float().view(-1, 1)


def cartpole_pets(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    goal_pos = torch.tensor([0.0, 0.6]).to(next_obs.device)
    x0 = next_obs[:, :1]
    theta = next_obs[:, 1:2]
    ee_pos = torch.cat([x0 - 0.6 * theta.sin(), -0.6 * theta.cos()], dim=1)
    obs_cost = torch.exp(-torch.sum((ee_pos - goal_pos) ** 2, dim=1) / (0.6**2))
    act_cost = -0.01 * torch.sum(act**2, dim=1)
    return (obs_cost + act_cost).view(-1, 1)


def inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.inverted_pendulum(act, next_obs)).float().view(-1, 1)


def halfcheetah(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    reward_ctrl = -0.1 * act.square().sum(dim=1)
    reward_run = next_obs[:, 0] - 0.0 * next_obs[:, 2].square()
    return (reward_run + reward_ctrl).view(-1, 1)


def pusher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    goal_pos = torch.tensor([0.45, -0.05, -0.323]).to(next_obs.device)

    to_w, og_w = 0.5, 1.25
    tip_pos, obj_pos = next_obs[:, 14:17], next_obs[:, 17:20]

    tip_obj_dist = (tip_pos - obj_pos).abs().sum(axis=1)
    obj_goal_dist = (goal_pos - obj_pos).abs().sum(axis=1)
    obs_cost = to_w * tip_obj_dist + og_w * obj_goal_dist

    act_cost = 0.1 * (act**2).sum(axis=1)

    return -(obs_cost + act_cost).view(-1, 1)
