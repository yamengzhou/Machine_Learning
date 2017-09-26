import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class SimpleTrackingCNN(nn.Module):

    def __init__(self):
        super(SimpleTrackingCNN, self).__init__()

        # 1 input image channel, 4 output channels, 5x5 square kernel
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)

        # MLP layer
        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)
        # self.fc4 = nn.Linear(256, 64)
        # self.fc5 = nn.Linear(64, 2)

    def forward(self, x):
        # Max pooling over a 2x2 window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = self.fc5(x)
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
