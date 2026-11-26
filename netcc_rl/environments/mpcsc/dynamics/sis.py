# Adapted from MPCSC; see LICENSE.MPCSC and THIRD_PARTY_NOTICES.md.
import numpy as np
import torch
import networkx as nx

from .random_streams import RandomStreams


class SISDynamics(RandomStreams):
    def __init__(self, net_type='ER', T=150, node_num=20, driver_node_num=10, observe_node_num=20, initial_I_node_num=5, beta=0.6, gamma=0.2, noise_std=0., seed=111111, frozen_adjacency=None, frozen_neighbors=None):
        self.reseed(seed)
        self.net_type = net_type
        self.T = T
        self.node_num = node_num
        self.driver_node_num = driver_node_num
        self.observe_node_num = observe_node_num
        self.initial_I_node_num = initial_I_node_num
        self.driver_node_list = self.py_random.sample(range(0, node_num), driver_node_num)
        if driver_node_num == node_num:
            # Action i controls node i; preserve the RNG draws for topology.
            self.driver_node_list.sort()
        self.initial_I_node_list = self.py_random.sample(range(0, node_num), initial_I_node_num)
        self.default_initial_I_node_list = self.py_random.sample(range(0, node_num), initial_I_node_num)
        self.beta = beta
        self.gamma = gamma
        self.dt = 0.1
        self.noise_std = noise_std
        self.seed = seed

        self.adj_matrix = torch.zeros([node_num, node_num])

        # init graph
        if frozen_adjacency is not None:
            self.adj_matrix = torch.tensor(frozen_adjacency, dtype=torch.float32)
            self.G = None if frozen_neighbors is not None else nx.from_numpy_array(self.adj_matrix.numpy())
        elif self.net_type == 'ER':
            while True:
                self.G = nx.erdos_renyi_graph(node_num, 3.0 / float(node_num - 1), seed=self.py_random)
                if nx.is_connected(self.G):
                    break
            for edge in self.G.edges:
                self.adj_matrix[edge[0]][edge[1]] = 1.0
                self.adj_matrix[edge[1]][edge[0]] = 1.0

        # get node_edge_dict
        if frozen_neighbors is not None:
            self.node_edge_dict={i:list(row) for i,row in enumerate(frozen_neighbors)}
        else:
            self.node_edge_dict={u:[v for v in range(self.node_num) if self.adj_matrix[u][v]==1]
                                 for u in range(self.node_num)}

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
            self.initial_I_node_list = self.py_random.sample(range(0, self.node_num), self.initial_I_node_num)
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
            Inew[i] = self.I[i] + self.dt * (-self.gamma * self.I[i] + infected_by_neighbor_ratio + self.noise_std * self.randn_like(Inew[i]))
        Inew = torch.clamp(Inew, 0, 1)
        self.I = Inew.clone()

        data = self.I
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
