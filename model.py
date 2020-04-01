import torch
import torch.nn.functional as F
from torch_geometric.nn import SplineConv


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1, 16, dim=1, kernel_size=5)
        self.conv2 = SplineConv(16, 32, dim=1, kernel_size=5)
        self.conv3 = SplineConv(32, 64, dim=1, kernel_size=7)
        self.conv4 = SplineConv(64, 128, dim=1, kernel_size=7)
        self.conv5 = SplineConv(128, 128, dim=1, kernel_size=11)
        self.conv6 = SplineConv(128, 2, dim=1, kernel_size=11)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(self.conv3(x, edge_index, edge_attr))
        x = self.conv4(x, edge_index, edge_attr)
        x = F.elu(self.conv5(x, edge_index, edge_attr))
        x = self.conv6(x, edge_index, edge_attr)
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)
