import torch
import torch.nn as nn
import math


class GRNTransformerLayer(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1):

        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, grn_mask):

        B, G, D = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, G, self.n_heads, self.head_dim).transpose(1,2)
        K = K.view(B, G, self.n_heads, self.head_dim).transpose(1,2)
        V = V.view(B, G, self.n_heads, self.head_dim).transpose(1,2)

        attn = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.head_dim)

        # GRN mask
        attn = attn + grn_mask

        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)

        out = out.transpose(1,2).contiguous().view(B,G,D)

        out = self.out_proj(out)

        x = x + self.dropout(out)
        x = self.norm1(x)

        ff = self.ffn(x)

        x = x + self.dropout(ff)
        x = self.norm2(x)

        return x
    
class GRNTransformer(nn.Module):

    def __init__(self, d_model, n_heads, n_layers):

        super().__init__()

        self.layers = nn.ModuleList([
            GRNTransformerLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])

    def forward(self, x, grn_mask):

        for layer in self.layers:

            x = layer(x, grn_mask)

        return x