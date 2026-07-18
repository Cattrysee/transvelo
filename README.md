# TransVelo

**TransVelo** is a Transformer-based method for single-cell RNA velocity inference.
It models each cell's spliced / unspliced expression as a gene sequence,
uses a Transformer encoder to capture gene-level dependencies, and predicts
per-gene velocity vectors with a kinetics-consistency regularizer.

> TransVelo is a Transformer-based RNA velocity method. It models each cell's
> spliced / unspliced expression as a gene sequence, uses a Transformer encoder
> to capture gene-level dependencies, and predicts per-gene velocity vectors with a
> kinetics-consistency regularizer.

---

## 📁 Repository Structure

```
TransVelo/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/                 # model code (formerly mode/)
│   ├── model.py          # TransVelo: composes all modules
│   ├── embeddings.py     # GeneEmbedding / ConditionEmbedding
│   ├── encoder.py       # GeneTransformer (PyTorch TransformerEncoder, GELU, pre-norm)
│   ├── encoder_grn.py   # GRNTransformer (optional: GRN-mask constrained self-attention)
│   ├── velocity_head.py  # VelocityHead: maps hidden states to velocity scalars
│   ├── kinetics.py       # KineticLayer: β·U − γ·S kinetic baseline
│   ├── loss.py           # TransVeloLoss: velocity/kinetic/smooth/direction multi-objective loss
│   ├── dataset.py       # VelocityDataset
│   ├── train.py          # train_model(): training loop
│   ├── infer_velocity.py # infer_velocity(): velocity inference
│   ├── utils_graph.py    # velocity_graph_smoothing: graph smoothing
│   ├── main.py          # training entry point (single-condition example)
│   └── downstream*.py   # downstream analysis
├── notebooks/           # data processing / method / baselines / plotting (outputs stripped)
├── data/                # data description + preprocessing scripts (large files, see note below)
└── figures/            # key paper figures (compressed)
```

---

## 🧠 Model Architecture

```
S, U  ──log1p──►  [S, U]  ──GeneEmbedding(Linear 2→d)──► z
                                                          │
                                              + ConditionEmbedding(cond)
                                                          │
                                                  GeneTransformer
                                                  (TransformerEncoder × n_layers,
                                                   GELU, norm_first=True)
                                                          │
                                                  VelocityHead (Linear d→1)
                                                          │
                                                        v_pred  ──► TransVeloLoss
                                                          │                         │
KineticLayer: β·U − γ·S ─────────────────────────► v_kin ─┘ (kinetic consistency)
```

- **GeneEmbedding**: stacks `log1p(S)` and `log1p(U)`, then applies `Linear(2→d_model)` to obtain the initial per-gene representation.
- **ConditionEmbedding**: embeds the condition (e.g. treated / control) and adds it to the gene representation (optional; skipped when `cond=None`).
- **GeneTransformer**: standard PyTorch `TransformerEncoder` with `batch_first=True`, `GELU`, `norm_first=True`.
  `encoder_grn.py` provides an optional variant `GRNTransformer` that constrains attention using a gene regulatory network (GRN) mask.
- **VelocityHead**: `Linear(d_model→1)` outputs the per-gene velocity scalar.
- **KineticLayer**: learnable kinetic baseline `v = β·U − γ·S`, used as a consistency supervisor.
- **TransVeloLoss**: velocity reconstruction loss + kinetic consistency loss + (optional) graph-smoothness / direction-consistency loss.

---

## 🚀 Quick Start

### 1. Environment

```bash
pip install -r requirements.txt
```

Dependencies (see `requirements.txt`): `torch`, `scanpy`, `scvelo`, `numpy`, `scipy`.
Running on a GPU is recommended (`device` is selected automatically as `cuda` / `cpu`).

### 2. Prepare Data

Data should be provided as an AnnData object (`.h5ad`) containing
`layers["spliced"]` and `layers["unspliced"]`, with `obs["condition"]` holding
the condition labels.

- Single-condition example: `data/processed_ctrl_data.h5ad`
- Multi-condition example: `data/danymic_rice.h5ad`

The preprocessing pipeline is described in `notebooks/dataprocess.ipynb` and the
notes under `data/`. **Because of their large size, the raw `.h5ad` files are not
included in the repository** — please prepare them following the instructions in
`data/README.md` or request access from the authors.

### 3. Train

```bash
cd src
python main.py            # single condition
# or
python downstream_main.py # includes downstream analysis and multi-condition
```

Training uses `train_model()` (see `src/train.py`). Default hyperparameters:
`d_model=128, n_heads=4, n_layers=2`, optimizer `Adam(W) lr=1e-4`, gradient
clipping `1.0`, loss weights `w_velocity=1.0, w_kinetic=0.5, w_smooth=0.2,
w_direction=0.1`.

### 4. Infer Velocity

```python
from infer_velocity import infer_velocity
velocity = infer_velocity(model, S, U, cond, device)
```

The inferred velocity is written to `adata.layers["transvelo_velocity"]`, after
which `scvelo` can be used for graph smoothing and stream-field visualization
(see the plotting notebooks under `notebooks/`).

---

## 📓 Notebooks

`notebooks/` contains data processing, the main method pipeline, baseline
comparisons (scVelo / DeepVelo), and the paper's plotting scripts.
To reduce file size, **all notebook outputs have been removed**; figures can be
regenerated after running the notebooks.

---

## ⚠️ Changes Relative to the Original Working Directory

This repository was reorganized from the original working directory. The following
**necessary** changes were made to ensure it runs and can be published:

1. **Added `src/train.py`**: the original `main.py` / `downstream_main.py` both
   `from train import train_model`, but `train.py` was missing. This file
   reconstructs `train_model()` from the inline training loop in `main.py` and is
   compatible with the call signature in `downstream_main.py`
   `(model, loader, optimizer, epochs, device)`.
2. **Fixed `src/dataset.py`**: `VelocityDataset.__getitem__` originally commented
   out the `"cond"` field, which would cause a `KeyError` when `main.py` /
   `downstream_main.py` access `batch["cond"]`. The `"cond"` return has been
   restored (i.e. "mixed-data" mode). If you need the "separate-data" mode, set
   `cond` to `None` at the call site.
3. **Directory rename**: `mode/` → `src/`, following common publishing conventions.
4. **Removed large files**: the original `Results/*.h5ad` (~2.1 GB total) and
   scratch directories were moved out of the repository (see "Archive" below).
5. **Stripped notebook outputs** and **compressed large figures**.

Please review items 1 and 2 above to confirm they match your experimental setup.

---

## 🗄️ Archive (Not in the Repository)

The following large / intermediate artifacts were moved to `transvelo2_archive/`
at the same level as the repository and are **not** committed to Git:

- `Results/*.h5ad` (umap-aligned results for DeepVelo / TransVelo / scVelo, ~2.1 GB)
- Exploratory scratch directories (`3-1`, `3-17`, `6-16`, `6-19`, `blank`, `Chinese`, `Chinese_old`, `Figure2`, `final_photo`, `other_photo`, `saved`, etc.)
- Uncompressed original large images

To restore them, retrieve them from `transvelo2_archive/`.

---

## 📜 License

This project is released under the MIT License (see `LICENSE`). For a different
license or special terms for thesis reproduction, please contact the authors.

## ✉️ Citation

(Please fill in the paper information here, e.g.)

```bibtex
@article{yourname2026transvelo,
  title={TransVelo: Transformer-based RNA velocity ...},
  author={Your Name},
  journal={...},
  year={2026}
}
```
