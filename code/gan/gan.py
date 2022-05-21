from numpy import outer
import torch
import torch.nn as nn
from typing import List
from torch.autograd import Variable
from torch.autograd import Function


def Gdense(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU6()
    )

def Ddense(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.1)
    )


class Generator(nn.Module):
    def __init__(self, dims: List):
        super().__init__()
        self.dims = dims
        net = nn.ModuleList(
            [Gdense(dims[i], dims[i+1]) for i in range(len(dims)-2)]
        )
        self.layer = nn.Sequential(*net)
        self.layer1 = nn.Linear(dims[-2], dims[-1])
        self.tanh = nn.Tanh()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer(x)
        out = self.layer1(out)
        out = self.tanh(out)
        
        return out

class Discriminator(nn.Module):
    def __init__(self, dims: List):
        super().__init__()
        self.dims = dims
        net = nn.ModuleList(
            [Ddense(dims[i], dims[i+1]) for i in range(len(dims)-2)]
        )
        self.layer = nn.Sequential(*net)
        self.layer1 = nn.Linear(dims[-2], dims[-1])
        self.sigmoid = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer(x)
        out = self.layer1(out)
        out = self.sigmoid(out)
        
        return out


