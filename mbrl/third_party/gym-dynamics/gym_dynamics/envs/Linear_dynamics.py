import gym
from gym.spaces import Box
import numpy as np
from ..dynamics.linear_dynamics import LinearDynamics

class LinearEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "single_rgb_array"], "render_fps": 4}

    def __init__(self, net_type='BA', T=10, node_num=20, input_node_num=5, output_node_num=10, noise_std=0., seed=1111):

        self.env = LinearDynamics(net_type=net_type, T=T, node_num=node_num, input_node_num=input_node_num, output_node_num=output_node_num, noise_std=noise_std, seed=seed)
        self.T = self.env.T
        self.node_num = self.env.node_num
        self.input_node_num = self.env.input_node_num
        self.output_node_num = self.env.output_node_num
        self.G = self.env.G
        self.edge_num = self.env.edge_num
        self.total_nodes = self.env.total_nodes
        self.input_nodes = self.env.input_nodes
        self.output_nodes = self.env.output_nodes

        self.observation_space = Box(low=-10000.0, high=10000.0, shape=(self.node_num,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.input_node_num,), dtype=np.float32)


    def reset(self):
        # We need the following line to seed self.np_random
        observation = self.env.reset()
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def render(self, MAX=100.0, MIN=-100.0, log_fig_dir=None, is_show=False):
        maxNodeValue, minNodeValue= self.env.render(MAX=MAX, MIN=MIN, log_fig_dir=log_fig_dir, is_show=is_show)
        return maxNodeValue, minNodeValue

    def close(self):
        self.env.close()