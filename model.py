import torch
import torch.nn as nn
import torch.nn.functional as F


class GomokuNet(nn.Module):
    """
    Use AlexNet for network.
    https://github.com/dansuh17/alexnet-pytorch/blob/master/model.py
    """

    def __init__(self, size):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=size**2),
        )
        self.init_bias()

    def init_bias(self):
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.constant_(layer.bias, 0)
        
        nn.init.constant_(self.conv_layers[4].bias, 1)
        nn.init.constant_(self.conv_layers[10].bias, 1)
        nn.init.constant_(self.conv_layers[12].bias, 1)


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 256*6*6)
        return self.fc_layers(x)