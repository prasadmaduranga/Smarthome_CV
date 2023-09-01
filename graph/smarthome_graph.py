import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data

num_node = 15
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(14, 1), (15, 14), (9, 15), (8, 15), (10, 8), (12, 10), (11, 9), (13, 11), (3, 14), (5, 3), (7, 5), (2, 14),
                    (4, 2), (6, 4)]

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = torch.tensor(self.get_spatial_graph(num_node, self_link, inward, outward), dtype=torch.float)
            return A
        else:
            raise ValueError()

    @staticmethod
    def get_spatial_graph(num_node, self_link, inward, outward):
        edge_index = torch.tensor(inward + outward, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=1)
        return edge_index


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    graph = Graph('spatial')
    edge_index = graph.get_adjacency_matrix('spatial')

    print(edge_index)

    data = Data(edge_index=edge_index)
    print(data)

    for i in range(num_node):
        plt.imshow(data.adjacency().to_dense(), cmap='gray')
        plt.title(f"Node {i} adjacency matrix")
        plt.show()
