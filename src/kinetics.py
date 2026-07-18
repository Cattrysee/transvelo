import torch
import torch.nn as nn

class KineticLayer(nn.Module):

    def __init__(self, n_genes):

        super().__init__()

        """ self.beta = nn.Parameter(torch.ones(n_genes))
        self.gamma = nn.Parameter(torch.ones(n_genes)) """
        """ self.beta = nn.Parameter(torch.rand(n_genes) * 0.5)
        self.gamma = nn.Parameter(torch.rand(n_genes) * 0.5) """
        self.beta = nn.Parameter(torch.randn(n_genes) * 0.1 + 1)
        self.gamma = nn.Parameter(torch.randn(n_genes) * 0.1 + 1)

    def forward(self, U, S):

        v = self.beta * U - self.gamma * S

        return v