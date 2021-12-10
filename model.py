import torch
import torch.nn as nn
import torch.nn.functional as F


class GomokuNet(nn.Module):
    """
    Use CNN for value network
    https://github.com/junxiaosong/AlphaZero_Gomoku/
    """

    def __init__(self, size):
        super(GomokuNet, self).__init__()
        self.board_size = size

        self.value_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2, kernel_size=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(2*size**2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.value_net(x)
        x = x.view(-1, 2*self.board_size**2)
        return self.fc(x)