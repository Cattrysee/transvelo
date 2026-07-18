#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TransVelo downstream analysis
"""

import os
import scanpy as sc
import scvelo as scv

# ==============================
# Parameters
# ==============================

adata_path = "transvelo_output.h5ad"

figdir = "figures"

os.makedirs(figdir, exist_ok=True)

scv.settings.figdir = figdir
scv.settings.set_figure_params(
    dpi=300,
    frameon=False
)

print("Loading data...")

# ==============================
# 1 Load data
# ==============================

adata = sc.read_h5ad(adata_path)

print(adata)
print("Layers:", adata.layers.keys())

# ==============================
# 2 Use TransVelo velocity
# ==============================

if "transvelo_velocity" not in adata.layers:
    raise ValueError("transvelo_velocity not found in h5ad")

adata.layers["velocity"] = adata.layers["transvelo_velocity"]

print("Velocity loaded")

# ==============================
# 3 neighbors
# ==============================

print("Computing neighbors...")

sc.pp.neighbors(adata)
sc.tl.umap(adata)

# ==============================
# 4 velocity graph
# ==============================

print("Computing velocity graph...")

scv.tl.velocity_graph(adata)

# ==============================
# 5 latent time
# ==============================

print("Computing latent time...")

scv.tl.latent_time(adata)

# ==============================
# 6 UMAP cell type
# ==============================

print("Plotting UMAP...")

scv.pl.umap(
    adata,
    color="celltype.anno",
    save="_celltype_umap.png",
    show=False
)

# ==============================
# 7 velocity stream plot
# ==============================

print("Plotting velocity stream...")

scv.pl.velocity_embedding_stream(
    adata,
    basis="umap",
    color="celltype.anno",
    legend_loc="right",
    save="_transvelo_stream.png",
    show=False
)

# ==============================
# 8 velocity arrow plot
# ==============================

print("Plotting velocity arrows...")

scv.pl.velocity_embedding(
    adata,
    basis="umap",
    arrow_length=3,
    arrow_size=2,
    color="celltype.anno",
    save="_transvelo_arrow.png",
    show=False
)

# ==============================
# 9 latent time trajectory
# ==============================

print("Plotting latent time...")

scv.pl.scatter(
    adata,
    color="latent_time",
    cmap="gnuplot",
    save="_latent_time.png",
    show=False
)

# ==============================
# 10 velocity grid
# ==============================

print("Plotting velocity grid...")

scv.pl.velocity_embedding_grid(
    adata,
    basis="umap",
    color="celltype.anno",
    save="_velocity_grid.png",
    show=False
)

# ==============================
# 11 Save results
# ==============================

print("Saving AnnData...")

adata.write("transvelo_result.h5ad")

print("All analysis finished.")
print("Figures saved in:", figdir)
