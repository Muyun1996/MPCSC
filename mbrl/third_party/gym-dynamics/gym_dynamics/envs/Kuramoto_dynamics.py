import random
import torch
import gym
from gym.spaces import Box
import numpy as np
from ..dynamics.kuramoto_dynamics import KuramotoDynamics

class KuramotoEnv(gym.Env):
    def __init__(self, net_type='ER', T=150, node_num=100, driver_node_num=10, observe_node_num=100, avg_degree=6, coupling=2, noise_std=0., seed=1111):
        self.env = KuramotoDynamics(net_type=net_type, T=T, node_num=node_num, driver_node_num=driver_node_num, observe_node_num=observe_node_num, avg_degree=avg_degree, coupling=coupling, noise_std=noise_std, seed=seed)
        self.T = self.env.T
        self.node_num = node_num
        self.driver_node_num = driver_node_num
        self.observe_node_num = observe_node_num

        self.obs_dim = self.observe_node_num
        self.action_dim = self.driver_node_num

        self.driver_node = self.env.driver_node_list
        self.adj_mat = self.env.adj_matrix.tolist()

        self.observation_space = Box(low=-10000.0, high=10000.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

    def reset(self, random_init=False):
        observation = self.env.reset(random_init=random_init)
        if torch.is_tensor(observation):
            observation = observation.detach().numpy()
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if torch.is_tensor(observation):
            observation = observation.detach().numpy()
        if torch.is_tensor(reward):
            reward = reward.detach().data.item()
        return observation, reward, done, info

    def render(self, log_fig_dir=None, is_show=False):
        self.env.render(log_fig_dir, is_show)

    def close(self):
        self.env.close()


