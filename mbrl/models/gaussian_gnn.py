# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F

from torch_geometric.nn import ChebConv, GCNConv, GATConv, GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import numpy as np
import mbrl.util.math
from sklearn.neighbors import NearestNeighbors

from .model import Ensemble
from .util import EnsembleLinearLayer, truncated_normal_init
from .sgcn import TrajectoryModel

import wandb

class GaussianGNN(Ensemble):
    """Implements an ensemble of multi-layer perceptrons each modeling a Gaussian distribution.

    This model corresponds to a Probabilistic Ensemble in the Chua et al.,
    NeurIPS 2018 paper (PETS) https://arxiv.org/pdf/1805.12114.pdf

    It predicts per output mean and log variance, and its weights are updated using a Gaussian
    negative log likelihood loss. The log variance is bounded between learned ``min_log_var``
    and ``max_log_var`` parameters, trained as explained in Appendix A.1 of the paper.

    This class can also be used to build an ensemble of GaussianMLP models, by setting
    ``ensemble_size > 1`` in the constructor. Then, a single forward pass can be used to evaluate
    multiple independent MLPs at the same time. When this mode is active, the constructor will
    set ``self.num_members = ensemble_size``.

    For the ensemble variant, uncertainty propagation methods are available that can be used
    to aggregate the outputs of the different models in the ensemble.
    Valid propagation options are:

            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

    The default value of ``None`` indicates that no uncertainty propagation, and the forward
    method returns all outputs of all models.

    Args:
        in_size (int): size of model input.
        out_size (int): size of model output.
        device (str or torch.device): the device to use for the model.
        num_layers (int): the number of layers in the model
                          (e.g., if ``num_layers == 3``, then model graph looks like
                          input -h1-> -h2-> -l3-> output).
        ensemble_size (int): the number of members in the ensemble. Defaults to 1.
        hid_size (int): the size of the hidden layers (e.g., size of h1 and h2 in the graph above).
        deterministic (bool): if ``True``, the model will be trained using MSE loss and no
            logvar prediction will be done. Defaults to ``False``.
        propagation_method (str, optional): the uncertainty propagation method to use (see
            above). Defaults to ``None``.
        learn_logvar_bounds (bool): if ``True``, the logvar bounds will be learned, otherwise
            they will be constant. Defaults to ``False``.
        activation_fn_cfg (dict or omegaconf.DictConfig, optional): configuration of the
            desired activation function. Defaults to torch.nn.ReLU when ``None``.
    """

    def __init__(
        self,
        model_type: str,
        node_num: int,
        driver_node_num: int,
        driver_node: omegaconf.ListConfig,
        has_adj_mat: bool,
        adj_mat: omegaconf.ListConfig,
        use_node_id: bool,
        obs_dim: int,
        action_dim: int,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        deterministic: bool = False,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
    ):
        super().__init__(
            ensemble_size, device, propagation_method, deterministic=deterministic
        )
        self.model_type = model_type
        self.use_node_id = use_node_id
        self.node_num = node_num
        self.driver_node_num = driver_node_num
        self.driver_node = torch.tensor(driver_node, dtype=torch.long)
        self.has_adj_mat = has_adj_mat
        self.adj_mat = torch.tensor(adj_mat, dtype=torch.long)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        assert self.obs_dim % self.node_num == 0
        assert self.action_dim % self.driver_node_num == 0
        self.node_obs_dim = int(self.obs_dim / self.node_num)
        self.node_action_dim = int(self.action_dim / self.driver_node_num)
        self.node_id_embedding_dim = 4 if self.use_node_id else 0

        #self.feature_dim = (self.node_obs_dim + self.node_action_dim + self.node_id_embedding_dim)
        self.in_size_for_gnn = (self.node_obs_dim + self.node_id_embedding_dim)
        self.out_size_for_gnn = 8
        self.in_size_for_sgcn = (self.node_obs_dim + self.node_action_dim + self.node_id_embedding_dim)
        self.out_size_for_sgcn = 8
        self.in_size = in_size
        self.out_size = out_size
        self.in_size_for_hidden_layers = self.node_num * (self.node_obs_dim + self.node_action_dim + self.node_id_embedding_dim)

        if self.model_type == "sgcn":
            self.in_size_for_hidden_layers = self.out_size_for_sgcn
        elif self.model_type == "gnn":
            self.in_size_for_hidden_layers = self.out_size_for_gnn

        self.node_one_hot_to_embedding = nn.Sequential(nn.Linear(self.node_num, self.node_id_embedding_dim),
                                                       nn.ReLU())
        self.obs_action_encoder = nn.Sequential(nn.Linear(self.out_size_for_gnn + self.node_action_dim, self.out_size_for_gnn),
                                                       nn.ReLU())
        # self.feature_embedding = nn.Sequential(nn.Linear(self.feature_dim, self.in_size_for_gnn),
        #                                                nn.ReLU())

        # self.gnn_layer1 = GCNConv(self.in_size_for_gnn, 32)
        # self.gnn_layer2 = GCNConv(32, self.out_size_for_gnn)
        num_heads = 4
        self.gnn_layer1 = GATv2Conv(self.in_size_for_gnn, 32 // num_heads, heads=num_heads)
        self.gnn_layer2 = GATv2Conv(32, self.out_size_for_gnn // num_heads, heads=num_heads)
        self.gnn_layer1_cat_last = GATConv(1, 32)
        self.gnn_layer2_cat_last = GATConv(32, 1)

        edge_index = [[], []]
        if self.has_adj_mat:
            for i in range(self.node_num):
                for j in range(self.node_num):
                    if i == j or self.adj_mat[i, j] == 1:
                        edge_index[0].append(i)
                        edge_index[1].append(j)
        else:
            for i in range(self.node_num):
                for j in range(self.node_num):
                    edge_index[0].append(i)
                    edge_index[1].append(j)

        self.edge_index = torch.tensor(np.array(edge_index), dtype=torch.long).to(self.device)

        self.sgcn = TrajectoryModel(number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1, dropout=0,
                                obs_len=1, pred_len=1, n_tcn=5, in_dim=self.in_size_for_sgcn, out_dims=self.out_size_for_sgcn, device=self.device).to(self.device)


        def create_activation():
            if activation_fn_cfg is None:
                activation_func = nn.ReLU()
            else:
                activation_func = activation_fn_cfg
            return activation_func

        def create_linear_layer(l_in, l_out):
            return EnsembleLinearLayer(ensemble_size, l_in, l_out)

        hidden_layers = [
            nn.Sequential(create_linear_layer(self.in_size_for_hidden_layers, hid_size), create_activation())
        ]
        for i in range(num_layers - 1):
            hidden_layers.append(
                nn.Sequential(
                    create_linear_layer(hid_size, hid_size),
                    create_activation(),
                )
            )
        self.hidden_layers = nn.Sequential(*hidden_layers)

        if deterministic:
            if self.model_type == "sgcn" or self.model_type == "gnn":
                self.mean_and_logvar = create_linear_layer(hid_size, self.node_obs_dim)
            else:
                self.mean_and_logvar = create_linear_layer(hid_size, self.out_size)
        else:
            if self.model_type == "sgcn" or self.model_type == "gnn":
                self.mean_and_logvar = create_linear_layer(hid_size, 2 * self.node_obs_dim)
            else:
                self.mean_and_logvar = create_linear_layer(hid_size, 2 * self.out_size)
            self.min_logvar = nn.Parameter(
                -10 * torch.ones(1, out_size), requires_grad=learn_logvar_bounds
            )
            self.max_logvar = nn.Parameter(
                0.5 * torch.ones(1, out_size), requires_grad=learn_logvar_bounds
            )

        self.apply(truncated_normal_init)
        self.to(self.device)

        self.elite_models: List[int] = None

        sgcn_total_params = sum(p.numel() for p in self.sgcn.parameters())
        print("total_params in sgcn: ", sgcn_total_params)
        mlp_total_params = sum(p.numel() for p in self.hidden_layers.parameters())
        print("total_params in mlp: ", mlp_total_params)

    def _maybe_toggle_layers_use_only_elite(self, only_elite: bool):
        if self.elite_models is None:
            return
        if self.num_members > 1 and only_elite:
            for layer in self.hidden_layers:
                # each layer is (linear layer, activation_func)
                layer[0].set_elite(self.elite_models)
                layer[0].toggle_use_only_elite()
            self.mean_and_logvar.set_elite(self.elite_models)
            self.mean_and_logvar.toggle_use_only_elite()

    def gnn_hidden_layers_cat_first(self, obs_action):
        """
        concatenate state and action before GNN
        :param state_action: [..., obs_dim + action_dim]
        :return:
        """
        obs_action = obs_action.contiguous()
        obs = obs_action[..., :self.obs_dim]
        action = obs_action[..., self.obs_dim:]
        batch_shape = list(obs_action[..., 0].shape)

        obs = obs.view(*batch_shape, self.node_num, self.node_obs_dim)
        extended_action = torch.zeros(*batch_shape, self.node_num, self.node_action_dim).to(self.device)
        extended_action[..., self.driver_node, :] = action.view(*batch_shape, self.driver_node_num, self.node_action_dim)

        if self.use_node_id:
            from functools import reduce
            bs = reduce(lambda x, y: x * y, batch_shape)
            node_index = torch.linspace(0, self.node_num-1, self.node_num, dtype=torch.long).to(self.device)

            one_hot = F.one_hot(node_index, num_classes=self.node_num).float()
            node_index_embedding = self.node_one_hot_to_embedding(one_hot)
            node_index_embedding = node_index_embedding.reshape(1, self.node_num, self.node_id_embedding_dim).repeat(bs, 1, 1)
            node_index_embedding = node_index_embedding.reshape(*batch_shape, self.node_num, self.node_id_embedding_dim)
            obs = torch.cat([obs, node_index_embedding], dim=-1)

        obs = obs.view(-1, self.node_num, self.node_obs_dim + self.node_id_embedding_dim)
        batch_size = obs.shape[0]
        edge_index = self.edge_index.clone()
        edge_index = edge_index.view(2, 1, -1).repeat(1, batch_size, 1) + torch.arange(batch_size).view(1, -1, 1).to(self.device) * self.node_num
        edge_index = edge_index.view(2, -1)
        x_gnn = obs.clone()
        x_gnn = x_gnn.view(batch_size * self.node_num, self.node_obs_dim + self.node_id_embedding_dim)

        # x_gnn = self.feature_embedding(x_gnn)
        obs_embedding = self.gnn_layer1(x_gnn, edge_index)
        obs_embedding = torch.tanh(obs_embedding)
        obs_embedding = self.gnn_layer2(obs_embedding, edge_index)

        obs_embedding = obs_embedding.view(*batch_shape, self.node_num, self.out_size_for_gnn)

        obs_embedding_cat_action = torch.cat([obs_embedding, extended_action], dim=-1)
        obs_action_embedding = self.obs_action_encoder(obs_embedding_cat_action)

        obs_action_embedding = obs_action_embedding.view(*batch_shape, self.node_num * self.out_size_for_gnn)

        # TODO: check
        new_obs_action = obs_action_embedding.view(*batch_shape, self.node_num, self.out_size_for_gnn)
        new_shape = new_obs_action.shape[:-3] + (new_obs_action.shape[-2] * new_obs_action.shape[-3],) + new_obs_action.shape[-1:]
        new_obs_action = new_obs_action.view(new_shape)

        # print("new_obs_action.shape", new_obs_action.shape)
        return new_obs_action

    def gnn_hidden_layers_cat_last(self, obs_action):
        """
        concatenate state and action after GNN
        :param state_action: [..., obs_dim + action_dim]
        :return:
        """
        obs = obs_action[..., :self.obs_dim]
        action = obs_action[..., self.obs_dim:]
        batch_shape = list(obs_action[..., 0].shape)
        obs = obs.view(-1, obs.shape[-1])
        action = action.view(-1, action.shape[-1])
        batch_size = obs.shape[0]

        edge_index = self.edge_index.clone()
        edge_index = edge_index.view(2, 1, -1).repeat(1, batch_size, 1) + torch.arange(batch_size).view(1, -1, 1).to(self.device) * self.node_num
        edge_index = edge_index.view(2, -1)
        x_gnn = obs.clone()
        x_gnn = x_gnn.view(batch_size * self.node_num, self.node_obs_dim)

        obs_embedding = self.gnn_layer1_cat_last(x_gnn, edge_index)
        obs_embedding = torch.tanh(obs_embedding)
        obs_embedding = self.gnn_layer2_cat_last(obs_embedding, edge_index)

        obs_embedding = obs_embedding.view(*batch_shape, self.obs_dim)
        action = action.view(*batch_shape, self.action_dim)
        state_action_embedding = torch.cat([obs_embedding, action], dim=-1)
        return state_action_embedding

    def mlp_hidden_layers(self, obs_action):
        """
        :param state_action: [..., obs_dim + action_dim]
        :return:
        """
        obs_action = obs_action.contiguous()
        obs = obs_action[..., :self.obs_dim]
        action = obs_action[..., self.obs_dim:]
        batch_shape = list(obs_action[..., 0].shape)

        obs = obs.view(*batch_shape, self.node_num, self.node_obs_dim)
        extended_action = torch.zeros(*batch_shape, self.node_num, self.node_action_dim).to(self.device)
        extended_action[..., self.driver_node, :] = action.view(*batch_shape, self.driver_node_num, self.node_action_dim)

        if self.use_node_id:
            from functools import reduce
            bs = reduce(lambda x, y: x * y, batch_shape)
            node_index = torch.linspace(0, self.node_num-1, self.node_num, dtype=torch.long).to(self.device)

            one_hot = F.one_hot(node_index, num_classes=self.node_num).float()
            node_index_embedding = self.node_one_hot_to_embedding(one_hot)
            node_index_embedding = node_index_embedding.reshape(1, self.node_num, self.node_id_embedding_dim).repeat(bs, 1, 1)
            node_index_embedding = node_index_embedding.reshape(*batch_shape, self.node_num, self.node_id_embedding_dim)
            obs_action = torch.cat([obs, extended_action, node_index_embedding], dim=-1)
        else:
            obs_action = torch.cat([obs, extended_action], dim=-1)

        obs_action = obs_action.view(*batch_shape, self.in_size_for_hidden_layers)
        return obs_action

    def create_adj_mat_for_boids(self, boids):
        """
        :param boids: [boid_num, 2(x,y)]
        :return: adj_mat of these boids
        """
        bSize = 17
        max_dis = bSize * 12
        mid_dis = bSize * 6
        min_dis = bSize
        # [boid_num, boid_num]
        pairwise_distances = torch.cdist(boids, boids)
        adj_mat = torch.zeros_like(pairwise_distances).to(self.device)
        adj_mat[pairwise_distances < max_dis] = 1
        return adj_mat

    def build_dynamic_graph_for_boids(self, obs):
        """
        :param obs: [batch_shape, node_num, node_obs_dim]
        :return: [batch_shape, node_num, node_num]
        """
        # TODO: check
        obs = torch.sign(obs) * torch.exp(torch.abs(obs)) - 1
        # print("obs: ", obs)

        obs = obs.view(-1, self.node_num, self.node_obs_dim)
        batch_size = obs.shape[0]
        adj_mat_list = []
        for i in range(batch_size):
            adj_mat = self.create_adj_mat_for_boids(obs[i, :, :4])
            adj_mat_list.append(adj_mat)
        real_adj_mat = torch.stack(adj_mat_list, dim=0)
        return real_adj_mat.to(self.device)

    def get_two_adj_mat(self, model_in):
        """
        only work for obs_len=1 currently!
        :param state_action: [..., obs_dim + action_dim]
        :return:
        """
        assert model_in.ndim == 2
        with torch.no_grad():
            obs_action = model_in
            self._maybe_toggle_layers_use_only_elite(only_elite=False)

            obs_len = 1
            identity_spatial = torch.ones((obs_len, self.node_num, self.node_num), device=self.device) * \
                               torch.eye(self.node_num, device=self.device)  # [obs_len N N]
            identity_temporal = torch.ones((self.node_num, obs_len, obs_len), device=self.device) * \
                                torch.eye(obs_len, device=self.device)  # [N obs_len obs_len]
            identity = [identity_spatial, identity_temporal]

            obs_action = obs_action.contiguous()
            obs = obs_action[..., :self.obs_dim]
            action = obs_action[..., self.obs_dim:]
            batch_shape = list(obs_action[..., 0].shape)

            obs = obs.view(*batch_shape, self.node_num, self.node_obs_dim)
            extended_action = torch.zeros(*batch_shape, self.node_num, self.node_action_dim).to(self.device)
            extended_action[..., self.driver_node, :] = action.view(*batch_shape, self.driver_node_num, self.node_action_dim)

            if self.use_node_id:
                from functools import reduce
                bs = reduce(lambda x, y: x * y, batch_shape)
                node_index = torch.linspace(0, self.node_num-1, self.node_num, dtype=torch.long).to(self.device)

                one_hot = F.one_hot(node_index, num_classes=self.node_num).float()
                node_index_embedding = self.node_one_hot_to_embedding(one_hot)
                node_index_embedding = node_index_embedding.reshape(1, self.node_num, self.node_id_embedding_dim).repeat(bs, 1, 1)
                node_index_embedding = node_index_embedding.reshape(*batch_shape, self.node_num, self.node_id_embedding_dim)
                obs_action = torch.cat([obs, extended_action, node_index_embedding], dim=-1)
            else:
                obs_action = torch.cat([obs, extended_action], dim=-1)

            obs_action = obs_action.view(-1, self.node_num, self.in_size_for_sgcn)
            obs_action = obs_action.unsqueeze(dim=1)

            # obs_action input graph of observed trajectory represented by velocity  [T 1 N 4]
            new_obs_action, sgcn_adj_mat = self.sgcn(obs_action, identity)

            # sgcn_adj_mat[sgcn_adj_mat > 0.1] = 1
            if self.has_adj_mat:
                real_adj_mat = self.adj_mat
                real_adj_mat = real_adj_mat.unsqueeze(dim=0).repeat(sgcn_adj_mat.shape[0], 1, 1).to(self.device)
            else:
                real_adj_mat = self.build_dynamic_graph_for_boids(obs)
            adj_mat_loss = torch.abs(sgcn_adj_mat - real_adj_mat).mean()
            wandb.log(
                {
                    "adj_mat_loss": adj_mat_loss,
                }
            )
            return sgcn_adj_mat, real_adj_mat

    def sgcn_hidden_layers(self, obs_action):
        """
        only work for obs_len=1 currently!
        :param state_action: [..., obs_dim + action_dim]
        :return:
        """
        obs_len = 1
        identity_spatial = torch.ones((obs_len, self.node_num, self.node_num), device=self.device) * \
                           torch.eye(self.node_num, device=self.device)  # [obs_len N N]
        identity_temporal = torch.ones((self.node_num, obs_len, obs_len), device=self.device) * \
                            torch.eye(obs_len, device=self.device)  # [N obs_len obs_len]
        identity = [identity_spatial, identity_temporal]

        obs_action = obs_action.contiguous()
        obs = obs_action[..., :self.obs_dim]
        action = obs_action[..., self.obs_dim:]
        batch_shape = list(obs_action[..., 0].shape)

        obs = obs.view(*batch_shape, self.node_num, self.node_obs_dim)
        extended_action = torch.zeros(*batch_shape, self.node_num, self.node_action_dim).to(self.device)
        extended_action[..., self.driver_node, :] = action.view(*batch_shape, self.driver_node_num, self.node_action_dim)

        if self.use_node_id:
            from functools import reduce
            bs = reduce(lambda x, y: x * y, batch_shape)
            node_index = torch.linspace(0, self.node_num-1, self.node_num, dtype=torch.long).to(self.device)

            one_hot = F.one_hot(node_index, num_classes=self.node_num).float()
            node_index_embedding = self.node_one_hot_to_embedding(one_hot)
            node_index_embedding = node_index_embedding.reshape(1, self.node_num, self.node_id_embedding_dim).repeat(bs, 1, 1)
            node_index_embedding = node_index_embedding.reshape(*batch_shape, self.node_num, self.node_id_embedding_dim)
            obs_action = torch.cat([obs, extended_action, node_index_embedding], dim=-1)
        else:
            obs_action = torch.cat([obs, extended_action], dim=-1)

        obs_action = obs_action.view(-1, self.node_num, self.in_size_for_sgcn)
        obs_action = obs_action.unsqueeze(dim=1)

        # obs_action input graph of observed trajectory represented by velocity  [T 1 N 4]
        new_obs_action, sgcn_adj_mat = self.sgcn(obs_action, identity)
        # print("new_obs_action: ", new_obs_action.shape)

        new_obs_action = new_obs_action.view(*batch_shape, self.node_num, obs_len * self.out_size_for_sgcn)
        new_shape = new_obs_action.shape[:-3] + (new_obs_action.shape[-2] * new_obs_action.shape[-3],) + new_obs_action.shape[-1:]
        new_obs_action = new_obs_action.view(new_shape)

        return new_obs_action

    def _default_forward(
        self, x: torch.Tensor, only_elite: bool = False, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._maybe_toggle_layers_use_only_elite(only_elite)
        # print("before sgcn: x.shape", x.shape)
        if self.model_type == "gnn":
            x = self.gnn_hidden_layers_cat_first(x)
        elif self.model_type == "sgcn":
            x = self.sgcn_hidden_layers(x)
        elif self.model_type == "mlp":
            x = self.mlp_hidden_layers(x)
        else:
            raise NotImplementedError("This model has not been implemented yet.")
        # print("before hidden_layers: x.shape", x.shape)
        x = self.hidden_layers(x)
        # print("after hidden_layers: x.shape", x.shape)
        mean_and_logvar = self.mean_and_logvar(x)
        # print("mean_and_logvar.shape", mean_and_logvar.shape)
        self._maybe_toggle_layers_use_only_elite(only_elite)
        if self.deterministic:
            if self.model_type == "sgcn" or self.model_type == "gnn":
                mean_and_logvar = mean_and_logvar.view(mean_and_logvar.shape[0], -1, self.out_size, 2)
                mean_and_logvar = mean_and_logvar.transpose(-2, -1).contiguous()
                mean_and_logvar = mean_and_logvar.view(mean_and_logvar.shape[0], -1, self.out_size * 2)
            return mean_and_logvar, None
        else:
            if self.model_type == "sgcn" or self.model_type == "gnn":
                mean_and_logvar = mean_and_logvar.view(mean_and_logvar.shape[0], -1, self.out_size, 2)
                mean_and_logvar = mean_and_logvar.transpose(-2, -1).contiguous()
                mean_and_logvar = mean_and_logvar.view(mean_and_logvar.shape[0], -1, self.out_size * 2)
            mean = mean_and_logvar[..., : self.out_size]
            logvar = mean_and_logvar[..., self.out_size :]
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
            return mean, logvar

    def _forward_from_indices(
        self, x: torch.Tensor, model_shuffle_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, batch_size, _ = x.shape

        num_models = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        shuffled_x = x[:, model_shuffle_indices, ...].view(
            num_models, batch_size // num_models, -1
        )

        mean, logvar = self._default_forward(shuffled_x, only_elite=True)
        # note that mean and logvar are shuffled
        mean = mean.view(batch_size, -1)
        mean[model_shuffle_indices] = mean.clone()  # invert the shuffle

        if logvar is not None:
            logvar = logvar.view(batch_size, -1)
            logvar[model_shuffle_indices] = logvar.clone()  # invert the shuffle

        return mean, logvar

    def _forward_ensemble(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.propagation_method is None:
            mean, logvar = self._default_forward(x, only_elite=False)
            if self.num_members == 1:
                mean = mean[0]
                logvar = logvar[0] if logvar is not None else None
            return mean, logvar
        assert x.ndim == 2
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if x.shape[0] % model_len != 0:
            raise ValueError(
                f"GaussianMLP ensemble requires batch size to be a multiple of the "
                f"number of models. Current batch size is {x.shape[0]} for "
                f"{model_len} models."
            )
        x = x.unsqueeze(0)
        if self.propagation_method == "random_model":
            # passing generator causes segmentation fault
            # see https://github.com/pytorch/pytorch/issues/44714
            model_indices = torch.randperm(x.shape[1], device=self.device)
            return self._forward_from_indices(x, model_indices)
        if self.propagation_method == "fixed_model":
            if propagation_indices is None:
                raise ValueError(
                    "When using propagation='fixed_model', `propagation_indices` must be provided."
                )
            return self._forward_from_indices(x, propagation_indices)
        if self.propagation_method == "expectation":
            mean, logvar = self._default_forward(x, only_elite=True)
            return mean.mean(dim=0), logvar.mean(dim=0)
        raise ValueError(f"Invalid propagation method {self.propagation_method}.")

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        use_propagation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes mean and logvar predictions for the given input.

        When ``self.num_members > 1``, the model supports uncertainty propagation options
        that can be used to aggregate the outputs of the different models in the ensemble.
        Valid propagation options are:

            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

        If a set of elite models has been indicated (via :meth:`set_elite()`), then all
        propagation methods will operate with only on the elite set. This has no effect when
        ``propagation is None``, in which case the forward pass will return one output for
        each model.

        Args:
            x (tensor): the input to the model. When ``self.propagation is None``,
                the shape must be ``E x B x Id`` or ``B x Id``, where ``E``, ``B``
                and ``Id`` represent ensemble size, batch size, and input dimension,
                respectively. In this case, each model in the ensemble will get one slice
                from the first dimension (e.g., the i-th ensemble member gets ``x[i]``).

                For other values of ``self.propagation`` (and ``use_propagation=True``),
                the shape must be ``B x Id``.
            rng (torch.Generator, optional): random number generator to use for "random_model"
                propagation.
            propagation_indices (tensor, optional): propagation indices to use,
                as generated by :meth:`sample_propagation_indices`. Ignore if
                `use_propagation == False` or `self.propagation_method != "fixed_model".
            use_propagation (bool): if ``False``, the propagation method will be ignored
                and the method will return outputs for all models. Defaults to ``True``.

        Returns:
            (tuple of two tensors): the predicted mean and log variance of the output. If
            ``propagation is not None``, the output will be 2-D (batch size, and output dimension).
            Otherwise, the outputs will have shape ``E x B x Od``, where ``Od`` represents
            output dimension.

        Note:
            For efficiency considerations, the propagation method used by this class is an
            approximate version of that described by Chua et al. In particular, instead of
            sampling models independently for each input in the batch, we ensure that each
            model gets exactly the same number of samples (which are assigned randomly
            with equal probability), resulting in a smaller batch size which we use for the forward
            pass. If this is a concern, consider using ``propagation=None``, and passing
            the output to :func:`mbrl.util.math.propagate`.

        """
        if use_propagation:
            return self._forward_ensemble(
                x, rng=rng, propagation_indices=propagation_indices
            )
        return self._default_forward(x)

    def _mse_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add model dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, _ = self.forward(model_in, use_propagation=False)
        return F.mse_loss(pred_mean, target, reduction="none").sum((1, 2)).sum()

    def _nll_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, pred_logvar = self.forward(model_in, use_propagation=False)
        if target.shape[0] != self.num_members:
            target = target.repeat(self.num_members, 1, 1)
        nll = (
            mbrl.util.math.gaussian_nll(pred_mean, pred_logvar, target, reduce=False)
            .mean((1, 2))  # average over batch and target dimension
            .sum()
        )  # sum over ensemble dimension
        nll += 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        return nll

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes Gaussian NLL loss.

        It also includes terms for ``max_logvar`` and ``min_logvar`` with small weights,
        with positive and negative signs, respectively.

        This function returns no metadata, so the second output is set to an empty dict.

        Args:
            model_in (tensor): input tensor. The shape must be ``E x B x Id``, or ``B x Id``
                where ``E``, ``B`` and ``Id`` represent ensemble size, batch size, and input
                dimension, respectively.
            target (tensor): target tensor. The shape must be ``E x B x Id``, or ``B x Od``
                where ``E``, ``B`` and ``Od`` represent ensemble size, batch size, and output
                dimension, respectively.

        Returns:
            (tensor): a loss tensor representing the Gaussian negative log-likelihood of
            the model over the given input/target. If the model is an ensemble, returns
            the average over all models.
        """
        if self.deterministic:
            return self._mse_loss(model_in, target), {}
        else:
            return self._nll_loss(model_in, target), {}

    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the squared error for the model over the given input/target.

        When model is not an ensemble, this is equivalent to
        `F.mse_loss(model(model_in, target), reduction="none")`. If the model is ensemble,
        then return is batched over the model dimension.

        This function returns no metadata, so the second output is set to an empty dict.

        Args:
            model_in (tensor): input tensor. The shape must be ``B x Id``, where `B`` and ``Id``
                batch size, and input dimension, respectively.
            target (tensor): target tensor. The shape must be ``B x Od``, where ``B`` and ``Od``
                represent batch size, and output dimension, respectively.

        Returns:
            (tensor): a tensor with the squared error per output dimension, batched over model.
        """
        assert model_in.ndim == 2 and target.ndim == 2
        with torch.no_grad():
            pred_mean, _ = self.forward(model_in, use_propagation=False)
            target = target.repeat((self.num_members, 1, 1))
            return F.mse_loss(pred_mean, target, reduction="none"), {}

    def sample_propagation_indices(
        self, batch_size: int, _rng: torch.Generator
    ) -> torch.Tensor:
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if batch_size % model_len != 0:
            raise ValueError(
                "To use GaussianMLP's ensemble propagation, the batch size must "
                "be a multiple of the number of models in the ensemble."
            )
        # rng causes segmentation fault, see https://github.com/pytorch/pytorch/issues/44714
        return torch.randperm(batch_size, device=self.device)

    def set_elite(self, elite_indices: Sequence[int]):
        if len(elite_indices) != self.num_members:
            self.elite_models = list(elite_indices)

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        model_dict = {
            "state_dict": self.state_dict(),
            "elite_models": self.elite_models,
        }
        torch.save(model_dict, pathlib.Path(save_dir) / self._MODEL_FNAME)

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / self._MODEL_FNAME)
        self.load_state_dict(model_dict["state_dict"])
        self.elite_models = model_dict["elite_models"]
