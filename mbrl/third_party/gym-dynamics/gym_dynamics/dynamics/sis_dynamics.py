import io
import os
import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import imageio
import csv
from PIL import Image
device = 'cpu'
use_cuda = None



class SISDynamics:
    def __init__(self, net_type='ER', T=150, node_num=20, driver_node_num=10, observe_node_num=20, initial_I_node_num=5, beta=0.6, gamma=0.2, noise_std=0., seed=111111):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.net_type = net_type
        self.T = T
        self.node_num = node_num
        self.driver_node_num = driver_node_num
        self.observe_node_num = observe_node_num
        self.initial_I_node_num = initial_I_node_num
        self.driver_node_list = random.sample(range(0, node_num), driver_node_num)
        self.initial_I_node_list = random.sample(range(0, node_num), initial_I_node_num)
        self.default_initial_I_node_list = random.sample(range(0, node_num), initial_I_node_num)
        self.beta = beta
        self.gamma = gamma
        self.dt = 0.1
        self.noise_std = noise_std
        self.seed = seed

        self.adj_matrix = torch.zeros([node_num, node_num])

        # init graph
        if self.net_type == 'ER':
            while True:
                self.G = nx.erdos_renyi_graph(node_num, 3.0 / float(node_num - 1))
                if nx.is_connected(self.G):
                    break
            print("Find a fully connected ER graph.")
            for edge in self.G.edges:
                self.adj_matrix[edge[0]][edge[1]] = 1.0
                self.adj_matrix[edge[1]][edge[0]] = 1.0

        # get node_edge_dict
        self.node_edge_dict = {}
        for u in range(self.node_num):
            self.node_edge_dict[u] = []
            for v in range(self.node_num):
                if self.adj_matrix[u][v] == 1:
                    self.node_edge_dict[u].append(v)

        # init control_matrix
        self.control_matrix = torch.zeros([self.driver_node_num, self.node_num])
        for idx, node in enumerate(self.driver_node_list):
            self.control_matrix[idx][node] = 1

        # get init_state by initial_I_node_list
        self.I0 = torch.zeros(self.node_num)
        for node in self.initial_I_node_list:
            self.I0[node] = 1
        self.I = self.I0.clone()


        self.obs_dim = self.node_num
        self.action_dim = self.driver_node_num
        self.done = False

        self.datas = []
        self.reward_list = []

        self.render_num = 1

    def reset(self, random_init=False):
        self.time_step = 0
        self.done = False
        if random_init:
            self.initial_I_node_list = random.sample(range(0, self.node_num), self.initial_I_node_num)
        else:
            self.initial_I_node_list = self.default_initial_I_node_list
        # get init_state by initial_I_node_list
        self.I0 = torch.zeros(self.node_num)
        for node in self.initial_I_node_list:
            self.I0[node] = 1
        self.I = self.I0.clone()
        self.obs = self.I.numpy()
        self.datas = [self.I.data.numpy()]
        self.reward_list = []
        return self.obs

    # TODO: delete this function
    def step_state(self, action):
        if not torch.is_tensor(action):
            action = torch.from_numpy(action)

        new_action = action @ self.control_matrix
        new_state = torch.zeros_like(self.state)
        for node in range(self.node_num):
            if self.state[node] == 1:  # infected
                if random.random() < self.gamma:
                    new_state[node] = 0
                else:
                    new_state[node] = 1
            elif self.state[node] == 0:  # suspected
                for v in self.node_edge_dict[node]:
                    if self.state[v] == 1 and random.random() < (self.beta - new_action[v]):
                        new_state[node] = 1
                        break
        self.state = new_state.clone()
        rho = self.state.mean()
        rho = rho.cpu() if use_cuda else rho
        self.rho_list.append(rho.data.numpy())

        self.obs = self.state
        reward = self.get_reward()
        self.reward_list.append(reward)
        self.time_step += 1
        if self.time_step >= self.T:
            self.done = True
        info = {}
        return self.obs, reward, self.done, info

    def step(self, action):
        if not torch.is_tensor(action):
            action = torch.from_numpy(action)
        action = action.float()
        new_action = action @ self.control_matrix

        # update I
        Inew = torch.zeros_like(self.I)
        for i in range(self.node_num):
            infected_by_neighbor_ratio = 0
            degree_i = len(self.node_edge_dict[i])
            for j in self.node_edge_dict[i]:
                new_beta = max(self.beta - new_action[i], 0)
                infected_by_neighbor_ratio += (new_beta * self.I[j] * (1-self.I[i])) / degree_i
            Inew[i] = self.I[i] + self.dt * (-self.gamma * self.I[i] + infected_by_neighbor_ratio + self.noise_std * torch.randn_like(Inew[i]))
        Inew = torch.clamp(Inew, 0, 1)
        self.I = Inew.clone()

        data = self.I
        data = data.cpu() if use_cuda else data
        self.datas.append(data.data.numpy())

        self.obs = self.I.numpy()
        reward = self.get_reward()
        self.reward_list.append(reward)

        self.time_step += 1
        if self.time_step >= self.T:
            self.done = True

        info = {}
        return self.obs, reward, self.done, info

    def get_reward(self):
        reward = - self.I.mean()
        return reward

    def render(self, log_fig_dir=None, is_show=False):
        start_time = time.time()
        if (log_fig_dir is not None) and (not os.path.exists(log_fig_dir)):
            os.makedirs(log_fig_dir)
        if self.time_step == self.T:
            # draw dynamic simulation'
            maxNodeValue = np.amax(self.datas)
            minNodeValue = np.amin(self.datas)
            fig, ax = plt.subplots()
            layout = nx.spring_layout(self.G, seed=self.seed)
            cmap = plt.cm.winter
            norm = plt.Normalize(vmin=minNodeValue, vmax=maxNodeValue)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            plt.colorbar(sm)
            edgecolors = ['red' if i in self.driver_node_list else 'blue' for i in range(self.node_num)]
            frames = []
            for i, data in enumerate(self.datas):
                ax.clear()
                #plt.colorbar(sm, cax=ax)
                nx.draw(self.G, layout, ax=plt.gca(), with_labels=False, node_size=200, width=1, edgecolors=edgecolors,
                        node_color=data, cmap=cmap, vmin=minNodeValue, vmax=maxNodeValue)
                # labels = {i: data[i] for i in range(self.node_num)}
                # nx.draw_networkx_labels(self.G, layout, labels)
                if log_fig_dir:
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    image = Image.open(buf)
                    frame = np.asarray(image)
                    frames.append(frame)
                    buf.close()
            # imageio.mimsave(log_fig_dir + '/sis_animation_{}.gif'.format(self.render_num), frames, duration=0.2) if log_fig_dir else None
            # draw simulation curve
            plt.close()
            plt.plot(self.datas)
            plt.savefig(
                log_fig_dir + '/kuramoto_simulation_{}.jpg'.format(self.render_num)) if log_fig_dir else None
            plt.show() if is_show else None

            # draw reward curve
            total_reward_tensor = "{:.3f}".format(torch.tensor(self.reward_list).sum().item())
            plt.close()
            plt.plot(self.reward_list)
            plt.savefig(log_fig_dir + '/sis_reward_{}_rt{}.jpg'.format(self.render_num, total_reward_tensor)) if log_fig_dir else None
            plt.show() if is_show else None
            plt.close()

            # log_fig_dir = "results"
            # save mean I
            if log_fig_dir:
                csv_file = log_fig_dir + "/sis_mean_I_data_{}_rt{}.csv".format(self.render_num, total_reward_tensor)
                reward_list = [[tensor_item.item()] for tensor_item in self.reward_list]
                with open(csv_file, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(reward_list)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print("render_num={}: time={:.6f} s".format(self.render_num, elapsed_time))

            self.render_num += 1


if __name__ == '__main__':
    sis_dynamics = SISDynamics(net_type='ER', T=150, node_num=20, driver_node_num=10, observe_node_num=20, initial_I_node_num=5, beta=0.6, gamma=0.2, noise_std=0.0, seed=111111)
    print("torch.seed:", torch.initial_seed())
    print("numpy seed: ", np.random.get_state()[1][0])
    for t in range(2 * sis_dynamics.T):
        if t % sis_dynamics.T == 0:
            sis_dynamics.reset()
            print(t)
        action = torch.randn(sis_dynamics.driver_node_num).abs()
        indices = torch.randperm(10)[:5]
        action[indices] = 0
        action = torch.zeros(sis_dynamics.driver_node_num)
        sis_dynamics.step(action)
        sis_dynamics.render(log_fig_dir="log", is_show=True)
    total_reward = torch.tensor(sis_dynamics.reward_list)
    print("return: ", total_reward.sum())



