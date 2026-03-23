import scanpy as sc
import scvi
import numpy as np
import sys
sys.path.append('../')

from scripts.subset_hvg import subset_to_hvg

adata = sc.read_h5ad(
    "../../data/obesity_challenge_2.h5ad"
)

print(adata)

adata_subset, hvg_genes,sig_genes = subset_to_hvg(
    adata,
    hvg_path="../../data/preprocessed/HVG/hvg10000_genes.txt",
    include_signature_genes=True
)

print(adata_subset.var_names)

import sys
sys.path.append('../')
from scripts.pairing import assign_state_label
adata_subset.obs["cell_state"] = assign_state_label(
    adata_subset.obs
)
print(adata_subset.obs["cell_state"].value_counts())

import scvi

# setup anndata
scvi.model.SCVI.setup_anndata(
    adata_subset,
    layer="counts"
)

# 先訓練 unsupervised VAE
vae = scvi.model.SCVI(
    adata_subset,
    n_latent=50
)

vae.train(
    max_epochs=200,
    accelerator="mps",
    devices=1,
    batch_size=512,
    early_stopping=True
)

# 再轉成 semi-supervised model
scanvi = scvi.model.SCANVI.from_scvi_model(
    vae,
    labels_key="cell_state",
    unlabeled_category="unknown"   # 必須是字串
)

scanvi.train(
    max_epochs=200,
    accelerator="gpu",
    devices=1,
    batch_size=512,
    early_stopping=True
)

# 取得 latent embedding
adata_subset.obsm["X_scanvi"] = scanvi.get_latent_representation(adata_subset)

print(adata_subset.obsm["X_scanvi"].shape)

save_dir = "model/scanvi_model"

scanvi.save(
    save_dir,
    overwrite=True
)

print("model saved to:", save_dir)