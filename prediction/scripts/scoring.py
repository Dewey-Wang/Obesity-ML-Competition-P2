import numpy as np
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import scanpy as sc
import numpy as np


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

    X_ctrl_sample

):

    hvg_mask = np.isin(
        adata_full.var_names,
        adata_hvg.var_names
    )

    non_hvg_mask = ~hvg_mask


    X_pred = np.zeros(
        (X_hvg_rec.shape[0],
         adata_full.n_vars)
    )

    ##################################
    # HVG genes
    ##################################

    X_pred[:, hvg_mask] = X_hvg_rec


    ##################################
    # non-HVG genes
    ##################################

    X_pred[:, non_hvg_mask] = X_ctrl_sample[:, non_hvg_mask]


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
