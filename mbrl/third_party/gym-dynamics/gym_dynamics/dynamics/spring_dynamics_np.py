import numpy as np
import matplotlib.pyplot as plt
import time
import random


class SpringDynamics_old(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=.5, vel_norm=.5,
                 interaction_strength=.1, T=50, sample_freq=100, spring_prob=[1./2, 0, 1./2], noise_var=0., seed=111):
        random.seed(seed)
        np.random.seed(seed)
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._spring_types = np.array([0., 0.5, 1.])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

        self.T = T
        self.sample_freq = sample_freq
        self.spring_prob = spring_prob
        self.time_step = 0
        self.time_sec = 0
        self.done = False

        # Sample edges
        self.edges = np.random.choice(self._spring_types,
                                      size=(self.n_balls, self.n_balls),
                                      p=self.spring_prob)
        self.edges = np.tril(self.edges) + np.tril(self.edges, -1).T
        np.fill_diagonal(self.edges, 0)

    def reset(self):
        self.time_step = 0
        # Initialize location and velocity
        self.loc_next = np.random.randn(2, self.n_balls) * self.loc_std
        self.vel_next = np.random.randn(2, self.n_balls)
        v_norm = np.sqrt((self.vel_next ** 2).sum(axis=0)).reshape(1, -1)
        self.vel_next = self.vel_next * self.vel_norm / v_norm
        self.loc_next, self.vel_next = self._clamp(self.loc_next, self.vel_next)

        obs = np.concatenate((self.loc_next, self.vel_next))
        return obs

    def step(self):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            # run leapfrog
            for i in range(0, self.sample_freq):
                forces_size = - self.interaction_strength * self.edges
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
                F = (forces_size.reshape(1, self.n_balls, self.n_balls) *
                     np.concatenate((
                         np.subtract.outer(self.loc_next[0, :],
                                           self.loc_next[0, :]).reshape(1, self.n_balls, self.n_balls),
                         np.subtract.outer(self.loc_next[1, :],
                                           self.loc_next[1, :]).reshape(1, self.n_balls, self.n_balls)))).sum(
                    axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                self.vel_next += self._delta_T * F
                self.loc_next += self._delta_T * self.vel_next
                self.loc_next, self.vel_next = self._clamp(self.loc_next, self.vel_next)
            self.time_step += 1

            # Add noise to observations
            self.loc_next += np.random.randn(2, self.n_balls) * self.noise_var
            self.vel_next += np.random.randn(2, self.n_balls) * self.noise_var

            obs_next = np.concatenate((self.loc_next, self.vel_next))
            rew = self._energy(self.loc_next, self.vel_next, self.edges)
            if self.time_step == self.T:
                self.done = True
            info = {}
            return obs_next, rew, self.done, info

    def render(self):
        pass

    def close(self):
        pass

    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] * (dist ** 2) / 2
            return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist



if __name__ == '__main__':
    sim = SpringDynamics(T=50, sample_freq=100)
    # sim = ChargedParticlesSim()


    t = time.time()
    obs = sim.reset()
    loc = np.zeros((50, 2, 5))
    vel = np.zeros((50, 2, 5))
    loc[0, :, :] = obs[:2, :]
    vel[0, :, :] = obs[2:, :]
    for i in range(0, 50):
        obs_next, rew, done, info = sim.step()
        loc[i, :, :] = obs_next[:2, :]
        vel[i, :, :] = obs_next[2:, :]


    # print("loc: ", loc)
    # print("vel: ", vel)
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