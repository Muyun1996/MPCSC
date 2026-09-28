import torch
import torch.nn as nn
from torch.nn import functional as F


class AsymmetricConvolution(nn.Module):

    def __init__(self, in_cha, out_cha):
        super(AsymmetricConvolution, self).__init__()

        self.conv1 = nn.Conv2d(in_cha, out_cha, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv2 = nn.Conv2d(in_cha, out_cha, kernel_size=(1, 3), padding=(0, 1))

        self.shortcut = lambda x: x

        if in_cha != out_cha:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_cha, out_cha, 1, bias=False)
            )

        self.activation = nn.PReLU()

    def forward(self, x):

        shortcut = self.shortcut(x)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.activation(x2 + x1)

        return x2 + shortcut


class InteractionMask(nn.Module):

    def __init__(self, number_asymmetric_conv_layer=7, spatial_channels=4, device='cpu'):
        super(InteractionMask, self).__init__()

        self.number_asymmetric_conv_layer = number_asymmetric_conv_layer

        self.spatial_asymmetric_convolutions = nn.ModuleList()

        for i in range(self.number_asymmetric_conv_layer):
            self.spatial_asymmetric_convolutions.append(
                AsymmetricConvolution(spatial_channels, spatial_channels)
            )

        self.spatial_output = nn.Sigmoid()
        self.temporal_output = nn.Sigmoid()

        self.device = device

    def forward(self, dense_spatial_interaction, threshold=0.5):

        assert len(dense_spatial_interaction.shape) == 4

        for j in range(self.number_asymmetric_conv_layer):
            dense_spatial_interaction = self.spatial_asymmetric_convolutions[j](dense_spatial_interaction)

        spatial_interaction_mask = self.spatial_output(dense_spatial_interaction)

        spatial_zero = torch.zeros_like(spatial_interaction_mask, device=self.device)

        spatial_interaction_mask = torch.where(spatial_interaction_mask > threshold, spatial_interaction_mask,
                                               spatial_zero)

        return spatial_interaction_mask


class ZeroSoftmax(nn.Module):

    def __init__(self):
        super(ZeroSoftmax, self).__init__()

    def forward(self, x, dim=0, eps=1e-5):
        x_exp = torch.pow(torch.exp(x) - 1, exponent=2)
        x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
        x = x_exp / (x_exp_sum + eps)
        return x


class SelfAttention(nn.Module):

    def __init__(self, in_dims=2, d_model=64, num_heads=4, device='cpu'):
        super(SelfAttention, self).__init__()

        self.embedding = nn.Linear(in_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)

        self.scaled_factor = torch.sqrt(torch.Tensor([d_model])).to(device)
        self.softmax = nn.Softmax(dim=-1)

        self.num_heads = num_heads

        self.device = device

    def split_heads(self, x):

        # x [batch_size seq_len d_model]

        x = x.reshape(x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads).contiguous()

        return x.permute(0, 2, 1, 3)  # [batch_size nun_heads seq_len depth]

    def forward(self, x, mask=False, multi_head=False):

        # batch_size seq_len 2

        assert len(x.shape) == 3

        embeddings = self.embedding(x)  # batch_size seq_len d_model
        query = self.query(embeddings)  # batch_size seq_len d_model
        key = self.key(embeddings)      # batch_size seq_len d_model

        if multi_head:
            query = self.split_heads(query)  # B num_heads seq_len d_model
            key = self.split_heads(key)  # B num_heads seq_len d_model
            attention = torch.matmul(query, key.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)
        else:
            attention = torch.matmul(query, key.permute(0, 2, 1))  # (batch_size, seq_len, seq_len)

        attention = self.softmax(attention / self.scaled_factor)

        if mask is True:

            mask = torch.ones_like(attention)
            attention = attention * torch.tril(mask)

        return attention, embeddings


class SparseWeightedAdjacency(nn.Module):

    def __init__(self, spa_in_dims=2, tem_in_dims=3, embedding_dims=64, obs_len=8, dropout=0,
                 number_asymmetric_conv_layer=7, device='cpu'):
        super(SparseWeightedAdjacency, self).__init__()

        # dense interaction
        self.spatial_attention = SelfAttention(spa_in_dims, embedding_dims, device=device)

        # interaction mask
        self.interaction_mask = InteractionMask(
            number_asymmetric_conv_layer=number_asymmetric_conv_layer, device=device
        )

        self.dropout = dropout
        self.zero_softmax = ZeroSoftmax()

    def forward(self, graph, identity):

        assert len(graph.shape) == 3

        spatial_graph = graph[:, :, :]  # (T N 2)

        # (T num_heads N N)   (T N d_model)
        dense_spatial_interaction, spatial_embeddings = self.spatial_attention(spatial_graph, multi_head=True)

        # [150, 4, 10, 10]
        spatial_mask = self.interaction_mask(dense_spatial_interaction)

        # self-connected
        spatial_mask = spatial_mask + identity[0].unsqueeze(1)
        # [150, 4, 10, 10]
        normalized_spatial_adjacency_matrix = self.zero_softmax(dense_spatial_interaction * spatial_mask, dim=-1)

        return normalized_spatial_adjacency_matrix, spatial_embeddings


class GraphConvolution(nn.Module):

    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()

        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.PReLU()

        self.dropout = dropout

    def forward(self, graph, adjacency):

        # graph [batch_size 1 seq_len 2]
        # adjacency [batch_size num_heads seq_len seq_len]
        gcn_features = self.embedding(torch.matmul(adjacency, graph))
        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout)

        return gcn_features  # [batch_size num_heads seq_len hidden_size]


class SparseGraphConvolution(nn.Module):

    def __init__(self, in_dims=16, embedding_dims=16, dropout=0):
        super(SparseGraphConvolution, self).__init__()

        self.dropout = dropout

        self.spatial_temporal_sparse_gcn = nn.ModuleList()

        self.spatial_temporal_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))
        self.spatial_temporal_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims))

    def forward(self, graph, normalized_spatial_adjacency_matrix):

        # graph [T seq_len num_pedestrians  3]
        # _matrix [batch num_heads seq_len seq_len]
        graph = graph[:, :, :, :]
        spa_graph = graph# (T, seq_len, boids_num, 2)

        # [T=150, num_heads, boids_num, embedding_dims=16]
        gcn_spatial_features = self.spatial_temporal_sparse_gcn[0](spa_graph, normalized_spatial_adjacency_matrix)
        gcn_spatial_features = self.spatial_temporal_sparse_gcn[1](gcn_spatial_features, normalized_spatial_adjacency_matrix)

        # [boids_num, num_heads, T=150, embedding_dims=16] [10, 4, 150, 16]
        gcn_spatial_features = gcn_spatial_features.permute(2, 1, 0, 3)

        return gcn_spatial_features


class TrajectoryModel(nn.Module):

    def __init__(self,
                 number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1, dropout=0,
                 obs_len=8, pred_len=12, n_tcn=5, in_dim=8,
                 out_dims=5, num_heads=4, device='cpu'):
        super(TrajectoryModel, self).__init__()

        self.number_gcn_layers = number_gcn_layers
        self.n_tcn = n_tcn
        self.dropout = dropout

        # sparse graph learning
        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency(spa_in_dims=in_dim, obs_len=obs_len,
            number_asymmetric_conv_layer=number_asymmetric_conv_layer, device=device
        )

        # graph convolution
        self.stsgcn = SparseGraphConvolution(
             in_dims=in_dim, embedding_dims=embedding_dims // num_heads, dropout=dropout
        )

        self.output = nn.Linear(embedding_dims // num_heads, out_dims)

    def forward(self, graph, identity):

        # graph : [1 obs_len N 3]
        # [T, heads, N, N] [150, N, 64]
        normalized_spatial_adjacency_matrix, spatial_embeddings = \
            self.sparse_weighted_adjacency_matrices(graph.squeeze(), identity)
        # [boids_num, heads, T=150, embedding_dim=16]
        gcn_representation = self.stsgcn(
            graph, normalized_spatial_adjacency_matrix
        )
        # [boids_num, obs_len, heads, embedding_dim]
        gcn_representation = gcn_representation.permute(0, 2, 1, 3)

        prediction = torch.mean(self.output(gcn_representation), dim=-2)
        mean_spatial_adjacency_matrix = normalized_spatial_adjacency_matrix.mean(dim=1)
        return prediction.permute(1, 0, 2).contiguous(), mean_spatial_adjacency_matrix
