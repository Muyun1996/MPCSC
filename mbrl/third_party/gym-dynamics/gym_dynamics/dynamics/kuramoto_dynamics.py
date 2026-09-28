import io
import os
import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import imageio
import csv
from PIL import Image
device = 'cpu'
use_cuda = None


class KuramotoDynamics:
    def __init__(self, net_type='ring', T=150, node_num=50, driver_node_num=15, observe_node_num=50, avg_degree=6, coupling=0.6, noise_std=0., seed=111111):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.net_type = net_type
        self.T = T
        self.node_num = node_num
        self.driver_node_num = driver_node_num
        self.driver_node_list = random.sample(range(0, node_num), driver_node_num)
        self.observe_node_num = observe_node_num
        self.avg_degree = avg_degree
        self.coupling = coupling
        self.dt = 0.01
        self.noise_std = noise_std
        self.seed = seed

        self.adj_matrix = torch.zeros([node_num, node_num])

        if self.net_type == 'default':
            for i in range(node_num // 2):
                for j in range(node_num // 2):
                    self.adj_matrix[i, j] = 1
                    self.adj_matrix[i + node_num // 2, j + node_num // 2] = 1
            self.G = nx.from_numpy_matrix(self.adj_matrix.numpy())
        if self.net_type == 'ring':
            for i in range(self.node_num-1):
                self.adj_matrix[i, i+1] = 1
                self.adj_matrix[i+1, i] = 1
            self.adj_matrix[0, self.node_num-1] = 1
            self.adj_matrix[self.node_num-1, 0] = 1
            self.G = nx.from_numpy_matrix(self.adj_matrix.numpy())
            self.driver_node_list = [3, 4, 5]
        if self.net_type == 'ER':
            while True:
                self.G = nx.erdos_renyi_graph(node_num, self.avg_degree / float(node_num - 1))
                if nx.is_connected(self.G):
                    break
            for edge in self.G.edges:
                self.adj_matrix[edge[0]][edge[1]] = 1.0
                self.adj_matrix[edge[1]][edge[0]] = 1.0

        #self.thetas = torch.rand(self.node_num) * 2 * np.pi
        #self.omegas = torch.randn(self.node_num)
        self.thetas = torch.arange(1, self.node_num + 1, 1) * (2.0 / float(self.node_num)) * np.pi
        self.init_omegas = torch.randn(self.node_num)

        self.control_matrix = torch.zeros([self.driver_node_num, self.node_num])
        for idx, node in enumerate(self.driver_node_list):
            self.control_matrix[idx][node] = 1

        self.obs = self.thetas
        self.obs_dim = self.node_num
        self.action_dim = self.driver_node_num
        self.done = False

        self.datas = []
        self.reward_list = []

        self.render_num = 1

    def set_state_from_obs(self, obs):
        self.thetas = torch.from_numpy(obs)

    def reset(self, random_init=False):
        self.time_step = 0
        self.done = False
        #self.thetas = 10 * torch.rand(self.node_num) * 0.2 * np.pi
        # self.omegas = torch.randn(self.node_num)
        if random_init:
            thetas_range = torch.arange(1, self.node_num + 1, 1)
            thetas_perm = torch.randperm(self.node_num)
            self.thetas = thetas_range[thetas_perm] * (2.0 / float(self.node_num)) * np.pi
        else:
            self.thetas = torch.arange(1, self.node_num + 1, 1) * (2.0 / float(self.node_num)) * np.pi
        self.omegas = self.init_omegas.clone()
        self.obs = self.thetas.numpy()
        self.datas = [self.thetas.data.numpy()]
        self.reward_list = []
        return self.obs

    def step(self, action):
        if not torch.is_tensor(action):
            action = torch.from_numpy(action)
        action = action.float()
        ii = self.thetas.unsqueeze(0).repeat(self.thetas.size()[0], 1)
        jj = ii.transpose(0, 1)
        dff = jj - ii
        sindiff = torch.sin(dff)
        mult = self.coupling * self.adj_matrix @ sindiff
        dia = torch.diagonal(mult)
        # dxdt = omegas + coupling * interactions.sum(axis=0)  # sum over incoming interactions
        #TODO: check this action
        new_action = 20 * action @ self.control_matrix
        self.thetas = self.dt * (self.omegas + dia + new_action) + self.thetas + self.noise_std * torch.randn_like(self.thetas)

        #data = torch.sin(self.thetas)
        data = self.thetas
        data = data.cpu() if use_cuda else data
        self.datas.append(data.data.numpy())

        self.obs = self.thetas.numpy()
        reward = self.get_reward()
        self.reward_list.append(reward)
        self.time_step += 1
        if self.time_step >= self.T:
            self.done = True
            # current_directory = os.getcwd()
            # current_directory = current_directory + "/results"
            # self.render(log_fig_dir=current_directory)
        info = {}
        return self.obs, reward, self.done, info

    def get_reward(self, reward_obj="order_parameter"):
        if reward_obj == "variance":
            reward = - torch.var(self.thetas)
        elif reward_obj == "fixed_target":
            reward = -torch.mean(torch.abs(self.thetas[:5]-0) + torch.abs(self.thetas[5:]-2*3.1415))
        elif reward_obj == "order_parameter":
            angle_radians = self.thetas.numpy()
            # Convert the orientations to vectors on the complex plane
            vectors = np.array([np.cos(angle_radians), np.sin(angle_radians)]).T
            # Calculate the magnitude of the sum of vectors
            vector_sum_length = np.linalg.norm(np.sum(vectors, axis=0))
            # Calculate the Order Parameter
            order_parameter = vector_sum_length / self.node_num
            reward = torch.tensor(order_parameter, dtype=torch.float32)
        return reward

    def render(self, log_fig_dir=None, is_show=False):
        start_time = time.time()
        if (log_fig_dir is not None) and (not os.path.exists(log_fig_dir)):
            os.makedirs(log_fig_dir)
        if self.time_step == self.T:
            # draw dynamic simulation
            maxNodeValue = np.amax(self.datas)
            minNodeValue = np.amin(self.datas)
            fig, ax = plt.subplots()
            layout = nx.spring_layout(self.G, seed=self.seed)
            cmap = plt.cm.winter
            norm = plt.Normalize(vmin=minNodeValue, vmax=maxNodeValue)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            plt.colorbar(sm)
            edgecolors = ['red' if i in self.driver_node_list else 'gray' for i in range(self.node_num)]
            frames = []
            for i, data in enumerate(self.datas):
                ax.clear()
                plt.colorbar(sm, cax=ax)
                nx.draw(self.G, layout, ax=plt.gca(), with_labels=False, node_size=200, width=1, edgecolors=edgecolors,
                        node_color=data, cmap=cmap, vmin=minNodeValue, vmax=maxNodeValue)
                # labels = {i: data[i] for i in range(self.node_num)}
                # nx.draw_networkx_labels(self.G, layout, labels)
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                image = Image.open(buf)
                frame = np.asarray(image)
                frames.append(frame)
                buf.close()
            # imageio.mimsave(log_fig_dir + '/kuramoto_animation_{}.gif'.format(self.render_num), frames, duration=0.2) if log_fig_dir else None
            """
            data = self.thetas
            maxNodeValue = max(data.tolist())
            minNodeValue = min(data.tolist())
            layout = nx.spring_layout(self.G, seed=self.seed)
            cmap = plt.cm.winter
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([])
            plt.colorbar(sm)
            edgecolors = ['red' if i in self.driver_node_list else 'gray' for i in range(self.node_num)]
            nx.draw(self.G, layout, with_labels=False, node_size=200, width=1, edgecolors=edgecolors,
                    node_color=data, cmap=cmap, vmin=minNodeValue, vmax=maxNodeValue)
            # labels = {i: data[i] for i in range(self.node_num)}
            # nx.draw_networkx_labels(self.G, layout, labels)
            plt.savefig(log_fig_dir + "/kuramoto_network_t" + str(self.time_step) + "_" + str(self.render_num) + ".png") if log_fig_dir else None
            plt.show() if is_show else None
            plt.close()
            """
            total_reward_tensor = "{:.1f}".format(torch.tensor(self.reward_list).sum().item())
            # draw simulation curve
            plt.close()
            plt.plot(self.datas)
            plt.savefig(log_fig_dir + '/kuramoto_simulation_{}_rt{}.jpg'.format(self.render_num, total_reward_tensor)) if log_fig_dir else None

            # colors = ['#012a4a', '#013a63', '#01497c', '#014f86', '#2a6f97', '#2c7da0', '#468faf', '#61a5c2', '#89c2d9', '#a9d6e5']
            # np_data = np.array(self.datas)
            # sns.set(style="white", font_scale=1.5)
            # sns.set_palette("deep")
            # plt.figure(figsize=(5, 4))
            # for i, theta_row in enumerate(np_data.transpose()):
            #     plt.plot(theta_row, linewidth=2, color=colors[i])
            # plt.xlabel('Steps')
            # plt.ylabel('Phases')
            # plt.ylim(-5, 10)
            # plt.tight_layout()
            # plt.savefig(log_fig_dir + '/kuramoto_simulation_{}_rt{}.pdf'.format(self.render_num, total_reward_tensor), format='pdf') if log_fig_dir else None

            plt.show() if is_show else None

            # draw reward curve
            plt.close()
            plt.plot(self.reward_list)
            plt.savefig(log_fig_dir + '/kuramoto_reward_{}_rt{}.jpg'.format(self.render_num, total_reward_tensor)) if log_fig_dir else None
            plt.show() if is_show else None
            plt.close()

            # log_fig_dir = "results"
            # save order parameter data
            if log_fig_dir:
                csv_file = log_fig_dir + "/order_parameter_data_{}_rt{}.csv".format(self.render_num, total_reward_tensor)
                reward_list = [[tensor_item.item()] for tensor_item in self.reward_list]
                with open(csv_file, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(reward_list)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print("render_num={}: time={:.6f} s".format(self.render_num, elapsed_time))

            self.render_num += 1



if __name__ == '__main__':
    kuramoto_dynamics = KuramotoDynamics(net_type='ER', T=150, node_num=10, driver_node_num=3, observe_node_num=10, avg_degree=6, coupling=0.6, noise_std=0., seed=111111)
    for t in range(2 * kuramoto_dynamics.T):
        if t % kuramoto_dynamics.T == 0:
            kuramoto_dynamics.reset()
            print(t)
        #action = torch.randn(kuramoto_dynamics.driver_node_num)
        #action = 2 * torch.rand(kuramoto_dynamics.driver_node_num) - 1
        action = torch.zeros(kuramoto_dynamics.driver_node_num)
        kuramoto_dynamics.step(action)
        kuramoto_dynamics.render(log_fig_dir="log", is_show=True)


"""
device = 'cpu'
use_cuda = None
coupling=2
dt=0.01
#adj_matrix = adj_matrix - adj_matrix * np.eye(sz)

#sz = adj_mat_modular.shape[0]
#adj_matrix = adj_mat_modular

for t in range(10000):
    #ii = np.repeat(np.array([thetas]), thetas.shape[0], 0)
    ii = thetas.unsqueeze(0).repeat(thetas.size()[0], 1)
    jj = ii.transpose(0, 1)
    dff = jj - ii
    sindiff = torch.sin(dff)
    mult = coupling * adj_matrix @ sindiff
    dia =  torch.diagonal(mult)
    #dxdt = omegas + coupling * interactions.sum(axis=0)  # sum over incoming interactions
    thetas = dt * (omegas + dia) + thetas
    data = torch.sin(thetas)
    data = data.cpu() if use_cuda else data
    datas.append(data.data.numpy())
"""