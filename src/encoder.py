""" import torch
import torch.nn as nn

class GeneTransformer(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        dropout=0.1
    ):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

    def forward(self, z):

        h = self.encoder(z)

        return h """
# new version (updated 3-8)
import torch
import torch.nn as nn


class GeneTransformer(nn.Module):

    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        dropout=0.1
    ):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # added layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z):

        h = self.encoder(z)

        h = self.norm(h)

        return h
