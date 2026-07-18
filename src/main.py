import torch
import scanpy as sc
import scvelo as scv
import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader

# ===== Import your modules =====
from dataset import VelocityDataset
from embeddings import GeneEmbedding, ConditionEmbedding
from encoder import GeneTransformer
from velocity_head import VelocityHead
from kinetics import KineticLayer
from model import TransVelo
from loss import TransVeloLoss

import gc
import torch


# =========================
# 1 Device
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

SEED = 123

import random
import numpy as np
import torch
import scanpy as sc

# python
random.seed(SEED)

# numpy
np.random.seed(SEED)

# pytorch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# cuda deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# scanpy / scvelo
sc.settings.seed = SEED

# =========================
# 2 Load data
# =========================

adata = sc.read("data/processed_ctrl_data.h5ad")

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
# 3 Dataset
# =========================

dataset = VelocityDataset(S, U, cond)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)


# =========================
# 4 Initialize model modules
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


# =========================
# 5 Build TransVelo
# =========================

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
# 6 Optimizer
# =========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-3
)


# =========================
# 7 Train
# =========================

epochs = 100

loss_fn = TransVeloLoss(
    w_velocity=1.0,
    w_kinetic=0.5,
    w_smooth=0.2,
    w_direction=0.1
)

for epoch in range(epochs):

    model.train()

    total_loss = 0
    total_vel = 0
    total_kin = 0
    n_batches = 0

    for batch in loader:

        S_batch = batch["S"].to(device)
        U_batch = batch["U"].to(device)
        cond_batch = batch["cond"].to(device)

        v_pred, v_kin = model(
            S_batch,
            U_batch,
            cond_batch
        )

        loss, loss_dict = loss_fn(
            v_pred=v_pred,
            v_kinetic=v_kin,
            S=S_batch,
            X=S_batch,
            #adj=adj_matrix
        )

        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )
        optimizer.step()

        total_loss += loss.item()
        total_vel += loss_dict.get("velocity",0)
        total_kin += loss_dict.get("kinetic",0)
        n_batches += 1

    print(
        f"Epoch {epoch+1} | "
        f"Loss {total_loss/n_batches:.3f} | "
        f"vel {total_vel/n_batches:.3f} | "
        f"kin {total_kin/n_batches:.3f}"
    )


# =========================
# 8 Infer velocity
# =========================

""" model.eval()

S = S.to(device)
U = U.to(device)
cond = cond.to(device)

with torch.no_grad():

    v_pred, _ = model(S, U, cond)

velocity = v_pred.cpu().numpy()

adata.layers["transvelo_velocity"] = velocity """

model.eval()

velocity_list = []

with torch.no_grad():

    for batch in loader:   # reuse the DataLoader defined above

        S_batch = batch["S"].to(device)
        U_batch = batch["U"].to(device)
        cond_batch = batch["cond"].to(device)

        v_pred, _ = model(
            S_batch,
            U_batch,
            cond_batch
        )

        velocity_list.append(v_pred.cpu())

velocity = torch.cat(velocity_list, dim=0).numpy()

adata.layers["transvelo_velocity"] = velocity


# =========================
# 9 Visualize velocity
# =========================

""" adata.layers["velocity"] = velocity

scv.pp.neighbors(adata)

scv.tl.velocity_graph(adata)

scv.pl.velocity_embedding_stream(
    adata,
    basis="umap"
) """

""" adata.layers["velocity"] = velocity
# neighbors
scv.pp.neighbors(adata)

# UMAP
scv.tl.umap(adata)

# velocity graph
scv.tl.velocity_graph(adata, vkey="transvelo_velocity")

# plot
scv.pl.velocity_embedding_stream(
    adata,
    basis="umap",
    vkey="transvelo_velocity"
) """


# =========================
# 10 Save results
# =========================

adata.write("data/transvelo_output_ctrl.h5ad")

torch.save(
    model.state_dict(),
    "transvelo_model_ctrl.pt"
)

print("Training finished")


# =========================
# 11 Release GPU / memory
# =========================

print("Releasing GPU memory...")

del S
del U
del cond
del velocity
del velocity_list
del dataset
del loader
del model
del optimizer

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

print("Memory released.")
