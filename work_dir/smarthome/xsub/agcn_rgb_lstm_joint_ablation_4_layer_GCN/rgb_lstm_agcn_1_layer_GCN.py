import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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



class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        if torch.cuda.is_available():
            A = self.A.cuda(x.get_device())
        else:
            A = self.A

        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=False):
        super(GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        # self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        self.dropout = nn.Dropout(p=0.5)
        # else:
        #     self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        # x = self.tcn1(self.gcn1(x)) + self.residual(x)
        x = self.gcn1(x)
        return self.relu(x)
        # residual = x
        # x = self.gcn1(x)
        #
        # # x += residual
        # return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn_x1 = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.data_bn_x2 = nn.BatchNorm1d(1000)

        self.l1 = GCN_unit(3, 128, A, residual=False)
        # self.l2 = GCN_unit(64, 64, A)
        # self.l3 = GCN_unit(64, 64, A)
        # self.l4 = GCN_unit(64, 64, A)
        # self.l5 = GCN_unit(64, 128, A, stride=2)
        # self.l6 = GCN_unit(128, 128, A)
        # self.l7 = GCN_unit(128, 128, A)
        # self.l8 = GCN_unit(128, 256, A, stride=2)
        # self.l9 = GCN_unit(256, 256, A)
        # # self.l10 = GCN_unit(256, 256, A)
        # self.l10 = GCN_unit(256, 128, A)

        self.cnn1 = nn.Conv1d(1000, 512, 1)
        self.cnn2 = nn.Conv1d(512, 256, 1)
        self.cnn3 = nn.Conv1d(256, 128, 1)



        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        self.lstm = nn.LSTM(128*num_point+128, 128, batch_first=True)

        self.fc = nn.Linear(128, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn_x1, 1)
        bn_init(self.data_bn_x2, 1)

    def forward(self, x1,x2):
        N, C, T, V, M = x1.size()
        # N=32,M=1,T=4000,V=15,C=3
        x1 = x1.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x1 = self.data_bn_x1(x1)
        x1 = x1.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x1 = self.l1(x1)
        # x1 = self.l2(x1)
        # x1 = self.l3(x1)
        # x1 = self.l4(x1)
        # x1 = self.l5(x1)
        # x1 = self.l6(x1)
        # x1 = self.l7(x1)
        # x1 = self.l8(x1)
        # x1 = self.l9(x1)
        # x1 = self.l10(x1)

        # N*M,C,T,V
        x1 = x1.permute(0, 1, 3, 2)
        x1 = x1.contiguous().view(N, M,-1, T).mean(1)
        x1 = x1.permute(0, 2,1).contiguous() # 32,4000,1920
        # try max pooling or average pooling to reduce the dimension


        # Process x2

        x2 = x2.permute(0, 2, 1).contiguous()  # Shape: [N, T, 1000]
        x2 = self.data_bn_x2(x2)
        x2 = F.relu(self.cnn1(x2))
        x2 = self.dropout1(x2)
        x2 = F.relu(self.cnn2(x2))
        x2 = self.dropout2(x2)
        x2 = F.relu(self.cnn3(x2))
        x2 = x2.permute(0, 2, 1).contiguous() # 32,4000,128

        x = torch.cat((x1, x2), dim=2)

        x, (h_n, c_n) = self.lstm(x)
        x = x[:, -1, :]




        return self.fc(x)

