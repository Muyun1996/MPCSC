import matplotlib.pyplot as plt
import time
import numpy as np
import random
import torch
torch.set_default_dtype(torch.float32)

class SpringDynamics(object):
    def __init__(self, net_type='random', n_balls=5, driven_balls_num=3, box_size=5., loc_std=.5, vel_norm=.5, interaction_strength=.1, T=50, sample_freq=10, spring_prob=[1./2, 0, 1./2], noise_std=0., seed=111):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.net_type = 'random'
        self.n_balls = n_balls
        self.driven_balls_num = driven_balls_num
        self.driven_balls_list = torch.tensor([0, 2, 4])
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_std = noise_std

        self._spring_types = np.array([0., 0.5, 1.])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

        self.T = T
        self.sample_freq = sample_freq
        self.spring_prob = spring_prob
        self.time_step = 0
        self.time_sec = 0

        self.obs_dim = 4 * self.n_balls
        self.action_dim = 2 * self.driven_balls_num
        self.done = False

        # Sample edges
        if self.net_type == 'random':
            self.edges = np.random.choice(self._spring_types, size=(self.n_balls, self.n_balls), p=self.spring_prob)
            self.edges = np.tril(self.edges) + np.tril(self.edges, -1).T
            np.fill_diagonal(self.edges, 0)

        self.init_loc = torch.randn(2, self.n_balls) * self.loc_std
        self.init_vel = torch.randn(2, self.n_balls)
        self.edges = self.edges.astype(np.float32)
        self.edges = torch.from_numpy(self.edges)
        self.obs = torch.zeros(4 * self.n_balls)

    def reset(self):
        self.time_step = 0
        self.done = False
        # Initialize location and velocity
        self.loc_next = self.init_loc
        self.vel_next = self.init_vel
        # self.loc_next = torch.randn(2, self.n_balls) * self.loc_std
        # self.vel_next = torch.randn(2, self.n_balls)
        v_norm = torch.sqrt((self.vel_next ** 2).sum(dim=0)).reshape(1, -1)
        self.vel_next = self.vel_next * self.vel_norm / v_norm
        self.loc_next, self.vel_next = self._clamp(self.loc_next, self.vel_next)

        self.obs = torch.cat((self.loc_next, self.vel_next)).reshape(-1)
        return self.obs

    def step(self, action):
        """
        :param action: [2 * driven_balls_num,]
        :return:
        """
        # TODO: check it:  with np.errstate(divide='ignore'):
        # disables division by zero warning, since I fix it with fill_diagonal
        # run leapfrog

        action = action.reshape(2, -1)
        for i in range(0, self.sample_freq):
            forces_size = - self.interaction_strength * self.edges
            # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
            force = forces_size.reshape(1, self.n_balls, self.n_balls)
            temp1 = (self.loc_next[0, :].reshape(-1, 1) - self.loc_next[0, :]).reshape(1, self.n_balls, self.n_balls)
            temp2 = (self.loc_next[1, :].reshape(-1, 1) - self.loc_next[1, :]).reshape(1, self.n_balls, self.n_balls)
            F = (force * torch.cat((temp1, temp2), dim=0)).sum(dim=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            # if i == self.sample_freq-1:
            #     print(F[:, self.driven_balls_list], action)
            # control
            F[:, self.driven_balls_list] = F[:, self.driven_balls_list] + action
            # apply force
            self.vel_next = self.vel_next + self._delta_T * F
            self.loc_next = self.loc_next + self._delta_T * self.vel_next
            self.loc_next, self.vel_next = self._clamp(self.loc_next, self.vel_next)
        self.time_step += 1

        # Add noise to observations
        self.loc_next = self.loc_next + torch.randn(2, self.n_balls) * self.noise_std
        self.vel_next = self.vel_next + torch.randn(2, self.n_balls) * self.noise_std

        obs_next = torch.cat((self.loc_next, self.vel_next)).reshape(-1)
        self.obs = obs_next
        rew = self._energy(self.loc_next, self.vel_next, self.edges)
        self.done = False
        if self.time_step >= self.T:
            self.done = True
        info = {}
        return obs_next, rew, self.done, info

    def render(self, log_fig_dir=None, is_show=False):
        pass

    def close(self):
        pass

    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        K = 0.5 * torch.sum(vel * vel)
        U = 0
        for i in range(loc.shape[1]):
            for j in range(loc.shape[1]):
                if i != j:
                    r = loc[:, i] - loc[:, j]
                    dist = torch.sqrt(torch.sum(r * r))
                    U = U + 0.5 * self.interaction_strength * edges[i, j] * (dist ** 2) / 2
        return K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (torch.all(loc < self.box_size * 3))
        assert (torch.all(loc > -self.box_size * 3))
        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (torch.all(loc <= self.box_size))
        # assert(np.all(vel[over]>0))
        vel[over] = - torch.abs(vel[over])
        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (torch.all(loc >= -self.box_size))
        vel[under] = torch.abs(vel[under])

        return loc, vel



if __name__ == '__main__':
    sim = SpringDynamics(n_balls=500, T=50, sample_freq=100)

    t = time.time()
    obs = sim.reset()
    loc = torch.zeros((50, 2, 500))
    vel = torch.zeros((50, 2, 500))
    obs = obs.reshape(4, -1)
    loc[0, :, :] = obs[:2, :]
    vel[0, :, :] = obs[2:, :]
    for i in range(0, 50):
        action = torch.randn(2, 3) * 0
        obs_next, rew, done, info = sim.step(action)
        obs_next = obs_next.reshape(4, -1)
        loc[i, :, :] = obs_next[:2, :]
        vel[i, :, :] = obs_next[2:, :]


    print(sim.edges)
    print("Simulation time: {}".format(time.time() - t))
    vel_norm = np.sqrt((vel ** 2).sum(axis=1))
    plt.figure()
    axes = plt.gca()
    axes.set_xlim([-5., 5.])
    axes.set_ylim([-5., 5.])
    for i in range(loc.shape[-1]):
        plt.plot(loc[:, 0, i], loc[:, 1, i])
        plt.plot(loc[0, 0, i], loc[0, 1, i], 'd')
    plt.figure()
    energies = [sim._energy(loc[i, :, :], vel[i, :, :], sim.edges) for i in
                range(loc.shape[0])]
    plt.plot(energies)
    plt.show()