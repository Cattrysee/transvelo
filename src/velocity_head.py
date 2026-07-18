""" import torch
import torch.nn as nn

class VelocityHead(nn.Module):

    def __init__(self, d_model):

        super().__init__()

        self.linear = nn.Linear(d_model, 1)

    def forward(self, h):

        v = self.linear(h).squeeze(-1)

        return v """



# new version (updated 3-8)
import torch
import torch.nn as nn
import torch.nn.functional as F

class VelocityHead(nn.Module):

    def __init__(self, d_model):

        super().__init__()

        self.linear = nn.Linear(d_model, 1)
        #self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h):

        v = self.linear(h).squeeze(-1)

        # normalize per cell
        #norm = torch.norm(v, dim=1, keepdim=True) + 1e-6
        #v = v / norm

        # learnable scale
        #v = v * self.scale

        return v
