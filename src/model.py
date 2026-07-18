import torch
import torch.nn as nn

class TransVelo(nn.Module):

    def __init__(
        self,
        gene_embedding,
        condition_embedding,
        encoder,
        velocity_head,
        kinetic_layer=None,
        cell_graph=None,   # added
    ):
        super().__init__()

        self.gene_embedding = gene_embedding
        self.condition_embedding = condition_embedding
        self.encoder = encoder
        self.velocity_head = velocity_head
        self.kinetic_layer = kinetic_layer
        self.cell_graph = cell_graph   # added
    def forward(self, S, U, cond, adj=None):
        z = self.gene_embedding(S, U)
        """ cond_embed = self.condition_embedding(cond)
        z = z + cond_embed.unsqueeze(1) """

        # ⭐⭐⭐ key change: condition is now optional
        if self.condition_embedding is not None and cond is not None:
            cond_embed = self.condition_embedding(cond)
            z = z + cond_embed.unsqueeze(1)

        """ if self.cell_graph is not None:
            if adj is None:
                raise ValueError("CellGraphAttention requires adjacency matrix")
            z = self.cell_graph(z, adj)
        if adj is not None and hasattr(self, "cell_graph"):
            z = self.cell_graph(z, adj)  # call the new CellGraphAttention """
        if self.cell_graph is not None:
            if adj is None:
                raise ValueError("CellGraphAttention requires adjacency matrix")
            z = self.cell_graph(z, adj)

        h = self.encoder(z)
        v_pred = self.velocity_head(h)

        if self.kinetic_layer is not None:
            v_kin = self.kinetic_layer(S, U)
            return v_pred, v_kin

        return v_pred
