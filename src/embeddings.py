import torch
import torch.nn as nn

class GeneEmbedding(nn.Module):

    def __init__(self, d_model):

        super().__init__()

        self.linear = nn.Linear(2, d_model)

    def forward(self, S, U):

        """ x = torch.stack([S, U], dim=-1) """
        S = torch.log1p(S)
        U = torch.log1p(U)

        x = torch.stack([S, U], dim=-1)

        z = self.linear(x)

        return z


class ConditionEmbedding(nn.Module):

    def __init__(self, n_conditions, d_model):

        super().__init__()

        self.embedding = nn.Embedding(n_conditions, d_model)

    def forward(self, cond):

        return self.embedding(cond)