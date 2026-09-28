import random
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
torch.set_default_dtype(torch.float32)
torch.autograd.set_detect_anomaly(True)

class LinearDynamics:
    def __init__(self, net_type='BA', T=10, node_num=20, input_node_num=5, output_node_num=10, noise_std=0.,seed=111):
        """
        :param net_type: network type
        :param T: time horizon
        :param node_num: number of nodes
        :param input_node_num: number of control nodes
        :param output_node_num:
        :param mat_W:                   [node_num * node_num]
        :param mat_A: adj               [node_num * node_num]
        :param mat_B:                   [node_num * input_node_num]
        :param mat_C:                   [output_node_num * node_num]
        :param x: states of all nodes , [node_num]
        :param seed: control the config of the generated BA graph
        """
        self.seed = seed
        self.noise_std = noise_std
        random.seed(self.seed)
        np.random.seed(seed)
        torch.manual_seed(self.seed)
        self.T = T
        self.node_num = node_num
        self.input_node_num = input_node_num
        self.output_node_num = output_node_num
        if net_type == 'BA':
            self.G = nx.barabasi_albert_graph(node_num, 1)
        elif net_type == 'ER':
            self.G = nx.erdos_renyi_graph(node_num, 0.04)
        elif net_type == 'WS':
            self.G = nx.watts_strogatz_graph(node_num, 2, 0.3)
        else:
            raise Exception("Invalid network type!")
        self.edge_num = self.G.number_of_edges()
        self.total_nodes = [i for i in range(self.node_num)]
        input_nodes = random.sample(self.total_nodes, self.input_node_num)
        self.input_nodes = sorted(input_nodes)
        left_nodes = [i for i in self.total_nodes if i not in self.input_nodes]
        output_nodes = random.sample(left_nodes, output_node_num)
        self.output_nodes = sorted(output_nodes)

        self.x = torch.zeros(self.node_num)
        self.obs = self.x
        self.mat_W = torch.randn(node_num, node_num)
        self.mat_A = torch.zeros(node_num, node_num)
        self.mat_B = torch.zeros(node_num, input_node_num).round()
        self.mat_C = torch.zeros(output_node_num, node_num).round()
        # init mat_A
        for edge in self.G.edges:
            self.mat_A[edge[0]][edge[1]] = 1.0
            self.mat_A[edge[1]][edge[0]] = 1.0
        # init mat_B
        for node in self.input_nodes:
            self.mat_B[node][self.input_nodes.index(node)] = 1.0
        # init mat_C
        for node in self.output_nodes:
            self.mat_C[self.output_nodes.index(node)][node] = 1.0
        self.time_step = 0
        self.obs_dim = self.node_num
        self.action_dim = self.input_node_num
        self.done = False

    def reset(self, x=None):
        self.time_step = 0
        self.done = False
        if x is None:
            self.x = torch.zeros(self.node_num)
        self.obs = self.x
        return self.obs

    def step(self, u):
        """
        :param u: [input_node_num,]
        :return:
        """
        if not torch.is_tensor(u):
            u = torch.from_numpy(u)
        if torch.max(u) > 1.0 or torch.min(u) < -1.0:
            raise Exception("Error! action is not in[-1, 1]!")
        x_next = torch.matmul(self.mat_W * self.mat_A, self.x + torch.matmul(self.mat_B, u))
        x_next = x_next + self.noise_std * torch.randn(self.node_num)
        y = torch.matmul(self.mat_C, x_next)
        reward = y.sum(dim=0)
        self.x = x_next.data
        self.obs = self.x
        self.time_step += 1
        if self.time_step >= self.T:
            self.done = True
        info = {}
        return x_next, reward, self.done, info

    def render(self, MAX=100.0, MIN=-100.0, log_fig_dir=None, is_show=False):
        """
        :param MAX: maxNodeValue of all time_step
        :param MIN: minNodeValue of all time_step
        :param log_dir:
        :param is_show:
        :return: return maxNodeValue and minNodeValue of current time_step
        """
        plt.close()
        data = self.x.numpy()
        maxNodeValue = max(data.tolist())
        minNodeValue = min(data.tolist())
        data = (data - MIN) / (MAX - MIN)
        data = np.around(data, 2)

        layout = nx.spring_layout(self.G, seed=self.seed)
        cmap = plt.cm.winter
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm)

        edgecolors = []
        for i in range(self.node_num):
            if i in self.input_nodes:
                edgecolors.append('red')
            elif i in self.output_nodes:
                edgecolors.append('black')
            else:
                edgecolors.append('gray')
        #edgecolors = ['red' if i in self.input_nodes else 'black' for i in range(self.node_num)]
        nx.draw(self.G, layout, with_labels=False, node_size=50, edgecolors=edgecolors, node_color=data, cmap=cmap, vmin=0, vmax=1)
        # labels = {i: data[i] for i in range(self.node_num)}
        # nx.draw_networkx_labels(self.G, layout, labels)
        if is_show:
            plt.show()
        if log_fig_dir is not None:
            plt.savefig(log_fig_dir + "/" + str(self.time_step) + ".png")
        plt.close()
        return maxNodeValue, minNodeValue

    def save_config(self):
        return self.G, self.mat_W, self.mat_A, self.mat_B, self.mat_C

    def close(self):
        pass
