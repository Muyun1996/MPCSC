import gym
import torch
from gym.spaces import Box
import numpy as np
from ..dynamics.boids_dynamics import BoidsDynamics

class BoidsEnv(gym.Env):
    def __init__(self, T=150, boids_num=20, driver_boids_num=5, width=500, height=500, speed=170, dt=0.02, is_wrap=True, seed=1111):
        self.render_mode = None
        self.env = BoidsDynamics(T=T, boids_num=boids_num, driver_boids_num=driver_boids_num, width=width, height=height, speed=speed, dt=dt, is_wrap=is_wrap, seed=seed)
        self.T = self.env.T
        self.node_num = boids_num
        self.driver_node_num = driver_boids_num
        self.observe_node_num = boids_num

        self.obs_dim = 5 * self.observe_node_num
        self.action_dim = self.driver_node_num

        self.driver_node = self.env.driver_boids_list

        self.observation_space = Box(low=-10000.0, high=10000.0, shape=(self.obs_dim, ), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.action_dim, ), dtype=np.float32)


    def reset(self, random_init=False):
        # We need the following line to seed self.np_random
        observation = self.env.reset(random_init=random_init, display=True)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if torch.is_tensor(observation):
            observation = observation.detach().numpy()
        if torch.is_tensor(reward):
            reward = reward.detach().data.item()
        return observation, reward, done, info

    def render(self, log_fig_dir=None, is_show=False):
        if self.render_mode == None:
            self.env.render(log_fig_dir=log_fig_dir, is_show=is_show)

    def close(self):
        self.env.close()