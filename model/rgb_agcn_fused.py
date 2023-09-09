import math

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv
from torch.autograd import Variable
import torch.nn.functional as F



edge_index = torch.tensor(
    [[0,2,4,1,3,5,6,8,10,7,9,11,13,12,2,4,14,3,5,14,8,10,13,9,11,13,12,13],
     [2,4,14,3,5,14,8,10,13,9,11,13,14,13,0,2,4,1,3,5,6,8,10,7,9,11,13,14]],
    dtype=torch.long)



def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)



class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,out_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError("Graph object must be provided.")
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # ChebConv layer using the provided adjacency matrix
        self.cheb_conv = ChebConv(in_channels, 3, K=2)  # You can adjust K (number of hops) as needed

        self.lstm = nn.LSTM(input_size=45, hidden_size=128, num_layers=2, batch_first=True)

        self.fc = nn.Linear(128, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x1,x2): # x1 : skeleton data , x2: rgb data
        N, C, T, V, M = x1.size()

        # Flatten the input data to align with ChebConv requirements
        # x = x.view(N * M, C, T, V).permute(0, 3, 2, 1).contiguous().view(N * M * V, T, C)
        x1 = x1.permute(0,4,2,3,1).contiguous().view(N * M * T , V , C)
        # ChebConv layer
        x1 = self.cheb_conv(x1, edge_index)

        # LSTM layer
        # x = x.view(N * M * T, V, C)
        x1 = x1.view(N * M ,T, V* C)

        x1, _ = self.lstm(x1)  # Apply LSTM


        # Reshape the LSTM output back to the original shape
        x1 = x1[:, -1, :]
        x1 = self.fc(x1)
        x1 = F.softmax(x1, dim=1)

        return x

# python main_rgb_joint.py --config ./config/smarthome/cross_subject/train_rgb_joint.yaml

# python main.py --config ./config/nturgbd-cross-view/test_rgb_joint.yaml
# not tested yet