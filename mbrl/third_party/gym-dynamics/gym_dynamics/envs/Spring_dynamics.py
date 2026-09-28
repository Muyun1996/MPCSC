import gym
from gym.spaces import Box
import numpy as np
from ..dynamics.spring_dynamics import SpringDynamics

class SpringEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "save"], "render_fps": 4}

    def __init__(self, net_type='random', n_balls=5, driven_balls_num=3, box_size=5., loc_std=.5, vel_norm=.5,
                 interaction_strength=.1, T=50, sample_freq=100, spring_prob=[1./2, 0, 1./2], noise_std=0., seed=111):
        self.env = SpringDynamics(net_type=net_type, n_balls=n_balls, box_size=box_size, loc_std=loc_std, vel_norm=vel_norm,
                 interaction_strength=interaction_strength, T=T, sample_freq=sample_freq, spring_prob=spring_prob, noise_std=noise_std, seed=seed)

        self.T = T
        self.n_balls = n_balls
        self.driven_balls_num = self.env.n_balls
        self.obs_dim = 4 * self.n_balls
        self.action_dim = 2 * driven_balls_num

        self.observation_space = Box(low=-10000.0, high=10000.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)


    def reset(self):
        # We need the following line to seed self.np_random
        observation = self.env.reset()
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def render(self, log_fig_dir=None, is_show=False):
        self.env.render()

    def close(self):
        self.env.close()