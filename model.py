import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNetwork(nn.Module):

    def __init__(self, states_qty, actions_qty, h, w):
        super(DQNetwork, self).__init__()

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        self.states_qty = states_qty
        self.actions_qty = actions_qty

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 2, 1)
        self.conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 2, 1)

        self.fc1 = nn.Linear(64 * self.conv_h * self.conv_w, 256)
        self.fc2 = nn.Linear(256, self.actions_qty)

    def forward(self, x):
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.elu(self.fc1(x))
        return self.fc2(x)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class DDQNetwork(DQNetwork):

    def __init__(self, states_qty, actions_qty, h, w):
        super(DDQNetwork, self).__init__(states_qty, actions_qty, h, w)
        self.advantage_fc1 = nn.Linear(64 * self.conv_h * self.conv_w, 256)
        self.advantage_fc2 = nn.Linear(256, self.actions_qty)

    def forward(self, x):
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        value = F.elu(self.fc1(x))
        value = self.fc2(value)
        advantage = F.elu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)
        return value + advantage - torch.mean(advantage, dim=1, keepdim=True)
