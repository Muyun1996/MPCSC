# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from dataclasses import dataclass, field
from typing import Optional
from typing import List

import gym
import numpy as np
import omegaconf
from omegaconf import OmegaConf
import torch

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math

import datetime
import platform
import wandb
import cProfile

EVAL_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT



import hydra
import numpy as np
import omegaconf
import torch

import os
import sys
print("current dir: ", os.getcwd())
sys.path.insert(1, os.getcwd())

import mbrl.util
import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.pets as pets
import mbrl.algorithms.planet as planet
import mbrl.util.env


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from torch_geometric_temporal.nn.recurrent import TGCN
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from torch_geometric.data import Data
from torch_geometric.data import Dataset as gnnDataset
from torch_geometric.loader import DataLoader as gnnDataLoader
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

torch.set_default_dtype(torch.float32)


# 定义自定义数据集类
class MyCustomDataset_GNN(gnnDataset):
    def __init__(self, graphs, transform=None, pre_transform=None):
        self.graphs = graphs
        super(MyCustomDataset_GNN, self).__init__(None, transform, pre_transform)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

# 定义自定义数据集类
class MyCustomDataset_MLP(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


def train(
    env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    reward_fn: mbrl.types.RewardFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- init wandb-------------------
    sys_platform = platform.platform().lower()
    if "linux" in sys_platform:
        OmegaConf.update(cfg, "system", "linux", force_add=True)
    elif "darwin" in sys_platform:
        OmegaConf.update(cfg, "system", "mac", force_add=True)
    elif "windows" in sys_platform:
        OmegaConf.update(cfg, "system", "windows", force_add=True)
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("-%m-%d %H:%M")
    wandb.init(
        # set the wandb project where this run will be logged
        project="mbrl_pets_test_gnn",
        entity="moumuyun",
        name=cfg.name + formatted_time,
        # track hyperparameters and run metadata
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    OmegaConf.save(config=cfg, f=os.path.join(wandb.run.dir, 'conf.yaml'))

    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    work_dir = work_dir or os.getcwd()
    print(f"Results will be saved at {work_dir}.")

    if silent:
        logger = None
    else:
        logger = mbrl.util.Logger(work_dir)
        logger.register_group(
            mbrl.constants.RESULTS_LOG_NAME, EVAL_LOG_FORMAT, color="green"
        )

    # -------- add driver_node and adj_mat to cfg ----------
    @dataclass
    class Network_data:
        node_num: int = 0
        driver_node_num: int = 0
        driver_node: List[int] = field(default_factory=lambda: [])
        has_adj_mat: bool = False
        adj_mat: List[List[int]] = field(default_factory=lambda: [[]])
        use_node_id: bool = False

    network_config: Network_data = OmegaConf.structured(Network_data)
    driver_node = [int(x) for x in env.unwrapped.driver_node]
    network_config.node_num = env.unwrapped.node_num
    network_config.driver_node_num = env.unwrapped.driver_node_num
    network_config.driver_node = driver_node
    if hasattr(cfg.overrides, "use_node_id"):
        network_config.use_node_id = cfg.overrides.use_node_id
    if hasattr(env.unwrapped, 'adj_mat'):
        network_config.has_adj_mat = True
        network_config.adj_mat = [[int(x) for x in row] for row in env.unwrapped.adj_mat]
    else:
        network_config.has_adj_mat = False
        network_config.adj_mat = [[]]
    cfg.dynamics_model = OmegaConf.merge(cfg.dynamics_model, network_config)

    # -------- Create and populate initial env dataset --------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    experiment_dir = "/Users/moumuyun/mbrl-lib/mbrl/examples/exp/pets/default/gym___gym_dynamics/SIS_dynamics/2023.05.09/115051"
    work_dir = experiment_dir
    # replay_buffer_offline = np.load(work_dir + "/replay_buffer_offline.npz")
    replay_buffer.load(work_dir)

    obs_action = np.concatenate([replay_buffer.obs, replay_buffer.action], axis=1)
    obs_action = torch.from_numpy(obs_action).float()
    obs_action = obs_action[:1500]
    next_obs = torch.from_numpy(replay_buffer.next_obs).float()
    next_obs = next_obs[:1500].unsqueeze(dim=-1)

    print("dynamic_learner")
    obs_dim = 200
    node_num = 200
    node_obs_dim = 1
    node_action_dim = 1
    driver_node_num = cfg.dynamics_model.driver_node_num
    driver_node = cfg.dynamics_model.driver_node
    obs_action = obs_action.contiguous()
    obs = obs_action[..., :obs_dim]
    action = obs_action[..., obs_dim:]
    batch_shape = list(obs_action[..., 0].shape)
    obs = obs.view(*batch_shape, node_num, node_obs_dim)
    extended_action = torch.zeros(*batch_shape, node_num, node_action_dim, dtype=torch.float32)
    action = action.float()
    extended_action[..., driver_node, :] = action.view(*batch_shape, driver_node_num, node_action_dim)
    obs_action = torch.cat([obs, extended_action], dim=-1)
    print("obs_action.shape", obs_action.shape)

    edge_index = [[], []]
    adj_mat = cfg.dynamics_model.adj_mat
    for i in range(node_num):
        for j in range(node_num):
            if i == j or adj_mat[i][j] == 1:
                edge_index[0].append(i)
                edge_index[1].append(j)

    edge_index = torch.tensor(np.array(edge_index), dtype=torch.long)
    obs_action = obs_action.view(-1, node_num, node_obs_dim + node_action_dim)
    x_data = obs_action.clone()


    print("edge_index.shape", edge_index.shape)
    print("x_data.shape", x_data.shape)
    print("next_obs.shape", next_obs.shape)

    def mlp(x_data, next_obs):
        print("training mlp")
        # 将节点特征拼接为一个向量
        x_data = x_data.view(x_data.shape[0], -1)
        next_obs = next_obs.view(x_data.shape[0], -1)
        # 创建数据集
        dataset = MyCustomDataset_MLP(x_data, next_obs)
        # 定义一个简单的MLP模型
        class SimpleMLP(nn.Module):
            def __init__(self):
                super(SimpleMLP, self).__init__()
                self.fc1 = nn.Linear(400, 16)
                self.fc2 = nn.Linear(16, 200)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = SimpleMLP()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        num_epochs = 10
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            print("Epoch {}/{} - Loss: {:.8f}".format(epoch + 1, num_epochs, epoch_loss / len(loader)))
            wandb.log({"epoch": epoch, "train loss": epoch_loss / len(loader)})

    def gnn(x_data, edge_index, next_obs):
        print("training gnn")
        graphs = []
        for t in range(x_data.shape[0]):
            x = x_data[t]
            y = next_obs[t]
            graphs.append(Data(x=x, edge_index=edge_index, y=y))

        #dataset = MyCustomDataset_GNN(graphs)
        # 划分训练集和验证集
        train_size = int(0.8 * len(graphs))
        val_size = len(graphs) - train_size
        train_graphs, val_graphs = random_split(graphs, [train_size, val_size])

        # 创建DataLoader实例
        train_loader = gnnDataLoader(train_graphs, batch_size=1, shuffle=True)
        val_loader = gnnDataLoader(val_graphs, batch_size=1, shuffle=False)

        # 定义一个简单的图神经网络模型
        class SimpleGNN(torch.nn.Module):
            def __init__(self):
                super(SimpleGNN, self).__init__()
                self.conv1 = GCNConv(2, 64)
                self.conv2 = GCNConv(64, 1)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index

                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
                x = self.conv2(x, edge_index)

                return x

        class GAT(torch.nn.Module):
            def __init__(self):
                super(GAT, self).__init__()
                num_heads = 4
                self.conv1 = GATConv(2, 64, heads=num_heads)
                self.conv2 = GATConv(64 * num_heads, 1)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index

                x = self.conv1(x, edge_index)
                x = F.elu(x)  # GAT paper recommends using ELU instead of ReLU
                x = F.dropout(x, training=self.training)
                x = x.view(-1, 64 * 4)  # Concatenate the output of attention heads
                x = self.conv2(x, edge_index)
                x = F.relu(x)
                return x

        class MultiheadAttention(nn.Module):
            def __init__(self, in_channels, out_channels, num_heads):
                super(MultiheadAttention, self).__init__()
                self.attention = nn.MultiheadAttention(out_channels, num_heads)
                self.fc_q = nn.Linear(in_channels, out_channels)
                self.fc_k = nn.Linear(in_channels, out_channels)
                self.fc_v = nn.Linear(in_channels, out_channels)

            def forward(self, x):
                q = self.fc_q(x)
                k = self.fc_k(x)
                v = self.fc_v(x)
                x, _ = self.attention(q, k, v)
                return x

        class SetTransformer(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels, num_heads, num_inds):
                super(SetTransformer, self).__init__()
                self.encoder = MultiheadAttention(in_channels, hidden_channels, num_heads)
                self.decoder = MultiheadAttention(hidden_channels, out_channels, num_heads)
                self.inducing_points = nn.Parameter(torch.Tensor(num_inds, in_channels))
                self.fc_inducing_points = nn.Linear(in_channels, hidden_channels)
                nn.init.xavier_uniform_(self.inducing_points)

            def forward(self, x):
                x = x.unsqueeze(0)  # Reshape input to (1, batch_size, num_features)
                x = self.encoder(x)
                x = F.relu(x)
                inducing_points = self.fc_inducing_points(self.inducing_points)
                inducing_points = inducing_points.repeat(x.size(0), 1, 1)
                x = self.decoder(inducing_points)
                x = F.relu(x)
                return x.squeeze(0)  # Reshape output back to (batch_size, num_features)

        class SetTransformerModel(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels, num_heads, num_inds, dim_output=1):
                super(SetTransformerModel, self).__init__()
                self.set_transformer = SetTransformer(in_channels, hidden_channels, out_channels, num_heads, num_inds)
                self.fc = nn.Linear(out_channels, dim_output)

            def forward(self, batch):
                x = batch.x
                x = self.set_transformer(x)
                x = self.fc(x)
                return x

        model = SimpleGNN()
        # model = SetTransformerModel(in_channels=2, hidden_channels=64, out_channels=32, num_heads=4, num_inds=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        #loader = gnnDataLoader(dataset, batch_size=32, shuffle=True)
        num_epochs = 10
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                # print("batch.x.shape", batch.x.shape)
                # print("batch.edge_index.shape", batch.edge_index.shape)
                out = model(batch)
                print("out.shape", out.shape)
                print("batch.y.shape", batch.y.shape)
                # if epoch == 5:
                #     print("out: ", out)
                #     print("batch.y", batch.y.squeeze(-1))
                loss = criterion(out, batch.y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

            print("Epoch {}/{} - Loss: {:.8f}".format(epoch + 1, num_epochs, epoch_loss / len(train_loader)))
            wandb.log({"epoch": epoch, "train loss": epoch_loss / len(train_loader)})

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    out = model(batch)
                    loss = criterion(out, batch.y)
                    val_loss += loss.item()

    def tgcn(x_data, edge_index, next_obs):
        print("training tgcn")
        # 创建Data对象列表
        graphs = []
        for t in range(x_data.shape[0]):
            x = x_data[t]
            y = next_obs[t]
            graphs.append(Data(x=x, edge_index=edge_index, y=y))

        # 划分训练集和验证集
        train_size = int(0.8 * len(graphs))
        val_size = len(graphs) - train_size
        train_graphs, val_graphs = random_split(graphs, [train_size, val_size])

        # 创建DataLoader实例
        train_loader = gnnDataLoader(train_graphs, batch_size=10, shuffle=True)
        val_loader = gnnDataLoader(val_graphs, batch_size=10, shuffle=False)

        # 定义一个TGCN模型
        class TGCNModel(torch.nn.Module):
            def __init__(self, node_features, output_channels):
                super(TGCNModel, self).__init__()
                self.tgcn = TGCN(node_features, output_channels)

            def forward(self, x, edge_index, edge_weight=None):
                y = self.tgcn(x, edge_index, edge_weight)
                return y


        node_features = 2
        output_channels = 1
        model = TGCNModel(node_features, output_channels)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        num_epochs = 10
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                loss = criterion(out, batch.y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

            print("Epoch {}/{} - Loss: {:.8f}".format(epoch + 1, num_epochs, epoch_loss / len(train_loader)))
            wandb.log({"epoch": epoch, "train loss": epoch_loss / len(train_loader)})

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                out = model(batch.x, batch.edge_index)
                loss = criterion(out, batch.y)
                val_loss += loss.item()

        print("Validation Loss: {:.8f}".format(val_loss / len(val_loader)))

    # mlp(x_data, next_obs)
    gnn(x_data, edge_index, next_obs)
    # tgcn(x_data, edge_index, next_obs)

    wandb.finish()



@hydra.main(config_path="../examples/conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    train(env, term_fn, reward_fn, cfg)

run()















