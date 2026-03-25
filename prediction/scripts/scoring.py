import numpy as np
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import pandas as pd
import scanpy as sc
def compute_metric_l1_distance(
    true_state_proportion_df: pd.DataFrame,
    pred_state_proprotion_df: pd.DataFrame,
) -> float:
    # Going over all the genes that were perturbed in this set
    unique_perturb_genes = list(true_state_proportion_df["gene"].unique())

    all_l1_loss_list = []
    for gene in unique_perturb_genes:
        # Slicing the column with this gene
        true_gene_df = true_state_proportion_df[true_state_proportion_df["gene"] == gene]
        pred_gene_df = pred_state_proprotion_df[pred_state_proprotion_df["gene"] == gene]

        # print(gene, pred_gene_df.shape[0])
        assert true_gene_df.shape[0] == 1 and pred_gene_df.shape[0] == 1, f"Invalid prediction count for state gene={gene} count={pred_gene_df.shape[0]}!=1"

        # Getting the L1 loss for main  pre, adipo and other
        l1_three = (
            np.abs(true_gene_df.iloc[0]["pre_adipo"] - pred_gene_df.iloc[0]["pre_adipo"]) +
            np.abs(true_gene_df.iloc[0]["adipo"] - pred_gene_df.iloc[0]["adipo"]) +
            np.abs(true_gene_df.iloc[0]["other"] - pred_gene_df.iloc[0]["other"])
        )

        # Getting the L1 loss for lipo by adipo
        numerical_stab_term = 1e-20
        pred_lipo_adipo = pred_gene_df.iloc[0]["lipo"] / (pred_gene_df.iloc[0]["adipo"] + numerical_stab_term)
        true_lipo_adipo = true_gene_df.iloc[0]["lipo"] / (true_gene_df.iloc[0]["adipo"] + numerical_stab_term)
        l1_lipo_adipo = np.abs(true_lipo_adipo - pred_lipo_adipo)

        # Getting the average error
        average_l1 = 0.75 * l1_three + 0.25 * l1_lipo_adipo
        all_l1_loss_list.append(average_l1)

    # Getting the overall average over all the gene perturbation
    l1_loss = np.mean(all_l1_loss_list)
    return float(l1_loss)


def pearson_score(
    gtruth_X,
    pred_X,
    perturbed_centroid
):
    gtruth_mean = gtruth_X.mean(axis=0)
    pred_mean = pred_X.mean(axis=0)

    return scipy.stats.pearsonr(
        gtruth_mean - perturbed_centroid,
        pred_mean - perturbed_centroid
    )[0]


import numpy as np


def gaussian_kernel(
    source,
    target,
    kernel_mul=2.0,
    kernel_num=5,
    fix_sigma=2326
):

    total = np.concatenate(
        [source, target],
        axis=0
    )

    total0 = np.expand_dims(total, 0)
    total1 = np.expand_dims(total, 1)

    L2 = np.sum(
        (total0 - total1) ** 2,
        axis=2
    )

    n_samples = total.shape[0]

    # base bandwidth
    bandwidth = fix_sigma

    # create multi-scale bandwidths
    bandwidth /= kernel_mul ** (kernel_num // 2)

    bandwidth_list = [
        bandwidth * (kernel_mul ** i)
        for i in range(kernel_num)
    ]

    kernel_val = [

        np.exp(-L2 / bw)

        for bw in bandwidth_list

    ]

    return sum(kernel_val)



def compute_mmd_official(

    X,
    Y

):

    n = X.shape[0]

    kernels = gaussian_kernel(
        X,
        Y
    )

    XX = kernels[:n, :n]

    YY = kernels[n:, n:]

    XY = kernels[:n, n:]

    YX = kernels[n:, :n]

    mmd = np.sum(

        XX + YY - XY - YX

    ) / (n ** 2)

    return mmd



def mmd_metric(

    gt,
    pred

):

    n = min(

        len(gt),
        len(pred)

    )

    gt = gt[:n]

    pred = pred[:n]

    return compute_mmd_official(

        gt,
        pred

    )
    

def load_control_cells(
    control_source,
    gene_order,
    base_path="../../data/preprocessed/control_cells/"
):

    if control_source == "NC":

        path = base_path + "NC_control_cells.h5ad"

    elif control_source == "NC+NC":

        path = base_path + "NC_NC_control_cells.h5ad"

    else:

        raise ValueError(
            "control_source must be 'NC' or 'NC+NC'"
        )

    adata_ctrl = sc.read_h5ad(path)

    X_ctrl = adata_ctrl.X

    if not isinstance(X_ctrl, np.ndarray):
        X_ctrl = X_ctrl.toarray()

    ##################################
    # align gene order
    ##################################

    idx = np.array([
        np.where(
            adata_ctrl.var_names == g
        )[0][0]
        for g in gene_order
    ])

    X_ctrl = X_ctrl[:, idx]

    print(
        f"{control_source} control loaded:",
        X_ctrl.shape
    )

    return X_ctrl

def reconstruct_full_expression_with_control(

    adata_full,
    adata_hvg,

    X_hvg_rec,

    pert_list,

    X_ctrl_NC,
    X_ctrl_NCNC

):

    hvg_mask = np.isin(

        adata_full.var_names,

        adata_hvg.var_names

    )


    non_hvg_mask = ~hvg_mask


    ctrl_blocks = []


    ##################################
    # build control matrix per perturbation
    ##################################

    for pert in pert_list:

        n_genes = len(pert.split("+"))

        if n_genes == 1:

            ctrl_block = X_ctrl_NC

        else:

            ctrl_block = X_ctrl_NCNC


        ctrl_blocks.append(ctrl_block)


    X_ctrl_full = np.vstack(ctrl_blocks)


    ##################################
    # combine HVG + non-HVG
    ##################################

    X_pred = np.zeros(

        (

            X_hvg_rec.shape[0],

            adata_full.n_vars

        )

    )


    X_pred[:, hvg_mask] = X_hvg_rec


    X_pred[:, non_hvg_mask] = X_ctrl_full[:, non_hvg_mask]


    return X_pred

############################################
# helper: subsample across batches
############################################

def sample_100_across_batches(
    adata,
    mask,
    batch_key="batch",
    n_cells=100
):

    idx_all = np.where(mask)[0]

    batches = adata.obs.iloc[idx_all][batch_key]

    sampled_idx = []

    ##################################
    # sample evenly from batches
    ##################################

    unique_batches = batches.unique()

    per_batch = max(
        1,
        n_cells // len(unique_batches)
    )

    for b in unique_batches:

        batch_idx = idx_all[
            batches == b
        ]

        k = min(
            per_batch,
            len(batch_idx)
        )

        sampled_idx.extend(

            np.random.choice(
                batch_idx,
                k,
                replace=False
            )

        )

    ##################################
    # fill remaining slots randomly
    ##################################

    if len(sampled_idx) < n_cells:

        remaining = np.setdiff1d(
            idx_all,
            sampled_idx
        )

        extra = np.random.choice(

            remaining,

            min(
                n_cells-len(sampled_idx),
                len(remaining)
            ),

            replace=False
        )

        sampled_idx.extend(extra)

    ##################################

    sampled_idx = sampled_idx[:n_cells]

    return np.array(sampled_idx)


import numpy as np
from sklearn.neighbors import NearestNeighbors


def latent_to_expression_nn(

    latent_query,

    latent_train,

    adata_train,

    k=30,

    method="softmax",

    eps=1e-8
):

    nbrs = NearestNeighbors(

        n_neighbors=k,

        metric="euclidean"

    ).fit(latent_train)


    distances, indices = nbrs.kneighbors(latent_query)


    # convert sparse to dense if needed
    gene_train = adata_train.X

    if not isinstance(gene_train, np.ndarray):

        gene_train = gene_train.toarray()


    preds = []


    for d, idx in zip(distances, indices):

        neigh_expr = gene_train[idx]


        # 保證 2D
        neigh_expr = np.atleast_2d(neigh_expr)

        d = np.atleast_1d(d)


        if method == "mean":

            pred = neigh_expr.mean(axis=0)


        elif method == "distance":

            w = 1 / (d + eps)

            w = w / w.sum()

            pred = (neigh_expr * w[:, None]).sum(axis=0)


        elif method == "softmax":

            w = np.exp(-d)

            w = w / w.sum()

            pred = (neigh_expr * w[:, None]).sum(axis=0)


        else:

            raise ValueError(

                "method must be:\n"

                "mean\n"

                "distance\n"

                "softmax"

            )


        preds.append(pred)


    return np.vstack(preds)
