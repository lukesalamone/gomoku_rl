import torch
import torch.nn as nn
import torch.nn.functional as F

class GomokuNet(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.batch_norm = nn.BatchNorm2d(3)


        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = nn.Identity()
        self.activation = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.in_channels != self.out_channels:
            residual = self.shortcut

        x = self.blocks(x)
        x += residual
        return self.activation(x)