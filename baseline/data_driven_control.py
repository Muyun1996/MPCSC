import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.linalg import null_space, cholesky, pinv, svd
from scipy.sparse.linalg import svds
import torch

def get_order_parameter(thetas, node_num=10):
    angle_radians = thetas
    vectors = np.array([np.cos(angle_radians), np.sin(angle_radians)]).T
    vector_sum_length = np.linalg.norm(np.sum(vectors, axis=0))
    order_parameter = vector_sum_length / node_num
    order_parameter = torch.tensor(order_parameter, dtype=torch.float32)
    return order_parameter

np.random.seed(1)

# network size
n = 10
# control node size
m = 3
# control nodes
m_set = [3, 4, 5]
# m_set = [i for i in range(10)]
# number of neighbors per node
k = 2

# define patterns
theta1 = np.zeros(n)
theta2 = np.zeros(n)
for i in range(n):
    theta1[i] = (i+1)/n * 2 * np.pi
    #theta1[i] = np.remainder(2 * np.pi * (i+1) / n, 2 * np.pi)
    #theta2[i] = np.remainder(2 * np.pi * (i+1) / n, 2 * np.pi)
    #
    # theta2[i] = np.pi
    if i < 5:
        theta2[i] = 0
    else:
        theta2[i] = 2 * np.pi


# natural frequencies
omega = np.zeros(n)

# Construct a regular ring lattice: a graph with n nodes, each connected to k neighbors, k/2 on each side
A = np.zeros((n, n))
kHalf = k // 2
for i in range(n):
    for j in range(1, kHalf + 1):
        A[i, (i + j) % n] = 0.6
        A[i, (i - j) % n] = 0.6

# number of data
N = 2000

X0 = []#
X_f = []
X_bar = []
U = []

tspan = [0, 1.5]  # control times
h = 0.01  # discretization step
T = int(tspan[1] / h)

for l in range(N):
    theta = np.reshape(theta1 + 0.1 * np.random.randn(n), (n, 1))
    u = np.zeros((m, T-1))
    for t in range(T - 1):
        u[:, t] = 20 * np.random.randn(m)
        p = 0
        next_theta = np.zeros(n)
        for node in range(n):
            if node in m_set:
                next_theta[node] = theta[node, t] + h * omega[node] + h * u[p, t]
                p += 1
            else:
                next_theta[node] = theta[node, t] + h * omega[node]

            for neighbor in range(n):
                next_theta[node] += h * A[node, neighbor] * np.sin(theta[neighbor, t] - theta[node, t])
        theta = np.hstack((theta, next_theta.reshape(n, 1)))

    X0.append(theta[:, 0])
    U.append(np.flip(u, axis=1).flatten())
    X_bar.append(theta[:, 1:T-1].flatten())
    X_f.append(theta[:, -1])

def matrix_rank(M, tol=None):
    S = svd(M, compute_uv=False)
    if tol is None:
        tol = S.max() * max(M.shape) * np.finfo(S.dtype).eps
    return np.sum(S > tol)

X0 = np.array(X0).T
U = np.array(U).T
X_bar = np.array(X_bar).T
X_f = np.array(X_f).T

K_X0 = null_space(X0, rcond=1e-10)
K_U = null_space(U, rcond=1e-10)
xf_c = (theta2 - (X_f @ K_U @ pinv(X0 @ K_U, rcond=1e-10)) @ theta1)

U = U @ K_X0
X_bar = X_bar @ K_X0
X_f = X_f @ K_X0

# compute data-driven input
K_f = null_space(X_f, rcond=1e-10)
Q = 150 * np.eye(n*(T - 2))
R = np.eye(m*(T - 1))
tmp = X_bar.T @ Q @ X_bar + U.T @ R @ U
tmp += np.eye(tmp.shape[0]) * 1e-6
L = cholesky(tmp)

# W, S, V = svds(L @ K_f, k=min(m*(T-1) - n, matrix_rank(L @ K_f)))
# V = V.T
# idx = S.argsort()[::-1]
# S = S[idx]
# W = W[:, idx]
# V = V[:, idx]

W, S, Vh = svd(L @ K_f)
Vh = Vh.T
num_svs = min(m*(T-1) - n, matrix_rank(L @ K_f))
W = W[:, :num_svs]
S = S[:num_svs]
V = Vh[:, :num_svs]

u_opt = U @ pinv(X_f, rcond=1e-10) @ xf_c - U @ K_f @ pinv(W @ np.diag(S) @ V.T, rcond=1e-10) @ L @ pinv(X_f, rcond=1e-10) @ xf_c


# simulate controlled system
u_opt_seq = np.flip(u_opt.reshape(m, T - 1), axis=1)
theta = theta1.reshape(n, 1)
tspan = [0, 1.5]
T = int(tspan[1] / h)

for t in range(T - 1):
    N = theta.shape[1]
    u = np.zeros((m, T-1))
    if t < 150:
        u[:, t] = u_opt_seq[:, t]
    else:
        u[:, t] = np.zeros(m)

    p = 0
    next_theta = np.zeros(n)
    for node in range(n):
        if node in m_set:
            next_theta[node] = theta[node, t] + h * omega[node] + h * u[p, t]
            p += 1
        else:
            next_theta[node] = theta[node, t] + h * omega[node]

        for neighbor in range(n):
            next_theta[node] += h * A[node, neighbor] * np.sin(theta[neighbor, t] - theta[node, t])

    theta = np.hstack((theta, next_theta.reshape(n, 1)))

#theta_tot = np.hstack((theta1.reshape(n, 1) * np.ones((n, 100)), theta))
theta_tot = theta
final_order_parameter = get_order_parameter(theta[:,-1])
print("final_order_parameter = ", final_order_parameter)

#colors = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']
# colors = ['#582f0e', '#7f4f24', '#936639', '#a68a64', '#b6ad90', '#c2c5aa', '#a4ac86', '#656d4a', '#414833', '#333d29']
colors = ['#012a4a', '#013a63', '#01497c', '#014f86', '#2a6f97', '#2c7da0', '#468faf', '#61a5c2', '#89c2d9', '#a9d6e5']
sns.set(style="white", font_scale=1.5)
sns.set_palette("deep")
plt.figure(figsize=(5, 4))
#plt.plot(np.arange(T), theta_tot.T, linewidth=2)
for i, theta_row in enumerate(theta_tot):
    plt.plot(np.arange(T), theta_row, linewidth=2, color=colors[i])
plt.xlabel('Steps')
plt.ylabel('Phases')
plt.ylim(-5, 10)
plt.tight_layout()
plt.savefig("plot/data_driven_control.pdf", format='pdf')
plt.show()