# Adapted from MPCSC; see LICENSE.MPCSC and THIRD_PARTY_NOTICES.md.
import numpy as np
import torch
import networkx as nx

from .random_streams import RandomStreams


class KuramotoDynamics(RandomStreams):
    def __init__(self, net_type='ring', T=150, node_num=50, driver_node_num=15, observe_node_num=50, avg_degree=6, coupling=0.6, noise_std=0., seed=111111, frozen_adjacency=None, frozen_neighbors=None):
        self.reseed(seed)
        self.net_type = net_type
        self.T = T
        self.node_num = node_num
        self.driver_node_num = driver_node_num
        self.driver_node_list = self.py_random.sample(range(0, node_num), driver_node_num)
        if driver_node_num == node_num:
            # Action i controls node i; preserve the RNG draws for topology.
            self.driver_node_list.sort()
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
            self.G = None if frozen_neighbors is not None else nx.from_numpy_array(self.adj_matrix.numpy())
        if self.net_type == 'ring':
            for i in range(self.node_num-1):
                self.adj_matrix[i, i+1] = 1
                self.adj_matrix[i+1, i] = 1
            self.adj_matrix[0, self.node_num-1] = 1
            self.adj_matrix[self.node_num-1, 0] = 1
            self.G = None if frozen_neighbors is not None else nx.from_numpy_array(self.adj_matrix.numpy())
            self.driver_node_list = [3, 4, 5]
        if frozen_adjacency is not None:
            self.adj_matrix = torch.tensor(frozen_adjacency, dtype=torch.float32)
            self.G = None if frozen_neighbors is not None else nx.from_numpy_array(self.adj_matrix.numpy())
        elif self.net_type == 'ER':
            while True:
                self.G = nx.erdos_renyi_graph(node_num, self.avg_degree / float(node_num - 1), seed=self.py_random)
                if nx.is_connected(self.G):
                    break
            for edge in self.G.edges:
                self.adj_matrix[edge[0]][edge[1]] = 1.0
                self.adj_matrix[edge[1]][edge[0]] = 1.0

        self.thetas = torch.arange(1, self.node_num + 1, 1) * (2.0 / float(self.node_num)) * np.pi
        self.init_omegas = torch.randn(self.node_num, generator=self.torch_generator)

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
        if random_init:
            thetas_range = torch.arange(1, self.node_num + 1, 1)
            thetas_perm = torch.randperm(self.node_num, generator=self.torch_generator)
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
        new_action = 20 * action @ self.control_matrix
        self.thetas = self.dt * (self.omegas + dia + new_action) + self.thetas + self.noise_std * self.randn_like(self.thetas)

        #data = torch.sin(self.thetas)
        data = self.thetas
        self.datas.append(data.data.numpy())

        self.obs = self.thetas.numpy()
        reward = self.get_reward()
        self.reward_list.append(reward)
        self.time_step += 1
        if self.time_step >= self.T:
            self.done = True
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
