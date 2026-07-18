import torch
import scanpy as sc
import scvelo as scv
import numpy as np
import scipy.sparse as sp

from torch.utils.data import DataLoader

from dataset import VelocityDataset
from embeddings import GeneEmbedding, ConditionEmbedding
from encoder import GeneTransformer
from velocity_head import VelocityHead
from kinetics import KineticLayer
from model import TransVelo

from train import train_model
from infer_velocity import infer_velocity
from utils_graph import velocity_graph_smoothing


# =========================
# 1 device
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# =========================
# 2 load data
# =========================

adata = sc.read("data/danymic_rice.h5ad")

S = adata.layers["spliced"]
U = adata.layers["unspliced"]

if sp.issparse(S):
    S = S.toarray()

if sp.issparse(U):
    U = U.toarray()

S = torch.tensor(S, dtype=torch.float32)
U = torch.tensor(U, dtype=torch.float32)

conditions = adata.obs["condition"].astype("category").cat.codes.values
cond = torch.tensor(conditions, dtype=torch.long)

num_cells, num_genes = S.shape
n_conditions = len(np.unique(conditions))

print("Cells:", num_cells)
print("Genes:", num_genes)


# =========================
# 3 dataset
# =========================

dataset = VelocityDataset(S, U, cond)

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)


# =========================
# 4 model
# =========================

d_model = 128
n_heads = 4
n_layers = 2

gene_embedding = GeneEmbedding(d_model)

condition_embedding = ConditionEmbedding(
    n_conditions,
    d_model
)

encoder = GeneTransformer(
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers
)

velocity_head = VelocityHead(d_model)

kinetic_layer = KineticLayer(num_genes)


model = TransVelo(
    gene_embedding,
    condition_embedding,
    encoder,
    velocity_head,
    kinetic_layer
)

model = model.to(device)

print(model)


# =========================
# 5 optimizer
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)


# =========================
# 6 training
# =========================

train_model(
    model,
    loader,
    optimizer,
    epochs=50,
    device=device
)


# =========================
# 7 velocity inference
# =========================

velocity = infer_velocity(
    model,
    S,
    U,
    cond,
    device
)

adata.layers["transvelo_velocity"] = velocity


# =========================
# 8 graph smoothing
# =========================

sc.pp.neighbors(adata, n_neighbors=30)

adj = adata.obsp["connectivities"]

velocity_smoothed = velocity_graph_smoothing(
    velocity,
    adj
)

adata.layers["velocity"] = velocity_smoothed


# =========================
# 9 velocity graph
# =========================

scv.tl.velocity_graph(adata)

scv.pl.velocity_embedding_stream(
    adata,
    basis="umap"
)


# =========================
# 10 save
# =========================

adata.write("transvelo_output.h5ad")

torch.save(
    model.state_dict(),
    "transvelo_model.pt"
)

print("Training finished")