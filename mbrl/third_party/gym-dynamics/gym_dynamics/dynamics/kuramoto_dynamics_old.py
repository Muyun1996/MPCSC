import torch
import numpy as np
import matplotlib.pyplot as plt
device = 'cpu'
use_cuda = None

sz = 10
coupling = 2#10/adj_mat_modular.shape[0]
dt = 0.01
obj_matrix = torch.zeros([sz,sz])
for i in range(sz//2):
    for j in range(sz//2):
        obj_matrix[i,j] = 1
        obj_matrix[i+sz//2,j+sz//2]=1
#obj_matrix = obj_matrix - obj_matrix * np.eye(sz)

#sz = adj_mat_modular.shape[0]
#obj_matrix = adj_mat_modular
thetas = torch.rand(sz, device = device) * 2 * np.pi
omegas = torch.randn(sz, device = device)
datas = []
for t in range(10000):
    #ii = np.repeat(np.array([thetas]), thetas.shape[0], 0)
    ii = thetas.unsqueeze(0).repeat(thetas.size()[0], 1)
    jj = ii.transpose(0, 1)
    dff = jj - ii
    sindiff = torch.sin(dff)
    mult = coupling * obj_matrix @ sindiff
    dia =  torch.diagonal(mult)
    #dxdt = omegas + coupling * interactions.sum(axis=0)  # sum over incoming interactions
    thetas = dt * (omegas + dia) + thetas
    data = torch.sin(thetas)
    data = data.cpu() if use_cuda else data
    datas.append(data.data.numpy())
plt.plot(datas)
plt.show()