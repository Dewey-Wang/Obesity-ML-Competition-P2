import pandas as pd
import numpy as np
import scanpy as sc
import gc

from scripts.scoring import (
    pearson_score,
    mmd_metric,
    latent_to_expression_nn
)


def evaluate_generated_expression_fast(
    pred_gex_matric,
    generated_obs,
    conditions,

    adata_gt_path,
    centroid_path,

    hvg_path="../../data/preprocessed/HVG/hvg3000_genes.txt",

    n_random_genes=1000,

    random_seed=0,

    verbose=True
):

    rng = np.random.default_rng(random_seed)

    # -------------------------
    # centroid
    # -------------------------

    centroid_df = pd.read_csv(

        centroid_path,
        index_col=0

    )

    single_centroid = centroid_df.loc["single_centroid"].values
    double_centroid = centroid_df.loc["double_centroid"].values


    # ==================================================
    # load GT matrix
    # ==================================================

    print("\nLoading GT matrix...")

    adata_gt = sc.read_h5ad(

        adata_gt_path
    )

    X_gt_sparse = adata_gt.X
    gt_gene_array = adata_gt.obs["gene"].values

    gene_names = adata_gt.var_names.to_numpy()


    # ==================================================
    # gene sets
    # ==================================================

    print("\nPreparing gene sets...")


    # ---------- random genes ----------

    random_genes = rng.choice(

        gene_names,

        size=n_random_genes,

        replace=False

    )


    # ---------- HVG ----------

    hvg_genes = pd.read_csv(

        hvg_path,
        header=None

    )[0].values


    hvg_genes = np.intersect1d(

        hvg_genes,
        gene_names

    )


    # ==================================================
    # gene index mapping
    # ==================================================

    gene_to_idx = {

        g:i for i,g in enumerate(gene_names)

    }


    gene_sets = {

        "random1000": np.array([

            gene_to_idx[g]

            for g in random_genes

        ]),

        "hvg3000": np.array([

            gene_to_idx[g]

            for g in hvg_genes

        ])

    }


    # ---------- union for fast slicing ----------

    gene_union_idx = np.unique(

        np.concatenate(

            list(gene_sets.values())

        )

    )


    print({

        k: len(v)

        for k,v in gene_sets.items()

    })


    print(

        "union genes:",
        len(gene_union_idx)
    )


    # ==================================================
    # preload GT subset (huge speedup)
    # ==================================================

    print("\nPreloading GT gene subset...")

    X_gt_subset = X_gt_sparse[:, gene_union_idx].toarray().astype(np.float32)

    del X_gt_sparse
    gc.collect()


    # centroid subset once

    single_centroid = single_centroid[gene_union_idx]
    double_centroid = double_centroid[gene_union_idx]


    # map subset index

    subset_lookup = {

        g:i for i,g in enumerate(gene_union_idx)

    }


    gene_sets_subset = {

        k: np.array([

            subset_lookup[i]

            for i in v

        ])

        for k,v in gene_sets.items()

    }


    # ==================================================
    # index lookup
    # ==================================================

    gene_array = generated_obs["gene"].values


    gen_index = {

        pert: np.where(

            gene_array == pert

        )[0]

        for pert in conditions

    }


    gt_index = {

        pert: np.where(

            gt_gene_array == pert

        )[0]

        for pert in conditions

    }


    is_single = {

        pert: ("+" not in pert)

        for pert in conditions

    }


    # ==================================================
    # decode once
    # ==================================================

    gene_pred_all = pred_gex_matric


    # subset prediction once

    gene_pred_all = gene_pred_all[:, gene_union_idx]


    # ==================================================
    # evaluation
    # ==================================================

    results = []


    for pert in conditions:


        if verbose:

            print(f"\nEvaluating {pert}")


        idx_gen = gen_index[pert]
        idx_gt = gt_index[pert]


        if len(idx_gen)==0 or len(idx_gt)==0:

            continue


        pred_subset = gene_pred_all[idx_gen]

        gt_subset = X_gt_subset[idx_gt]


        centroid = (

            single_centroid

            if is_single[pert]

            else double_centroid

        )


        row = {

            "perturbation": pert,

            "n_gt_cells": len(idx_gt)

        }


        # ---------- metrics ----------

        for set_name, idx in gene_sets_subset.items():


            row[f"pearson_{set_name}"] = pearson_score(

                gt_subset[:, idx],

                pred_subset[:, idx],

                centroid[idx]

            )


            row[f"mmd_{set_name}"] = mmd_metric(

                gt_subset[:, idx],

                pred_subset[:, idx]

            )


        results.append(row)


    # ==================================================
    # summary
    # ==================================================

    df = pd.DataFrame(results)


    mean_row = {

        "perturbation": "MEAN",

        "n_gt_cells": df["n_gt_cells"].sum()

    }


    for col in df.columns:

        if col.startswith("pearson") or col.startswith("mmd"):

            mean_row[col] = df[col].mean()


    df = pd.concat(

        [df, pd.DataFrame([mean_row])],

        ignore_index=True

    )


    print("\nSummary")

    print(df)


    print("\nAverages")


    for col in df.columns:

        if col.startswith("pearson") or col.startswith("mmd"):
    
            print(f"{col}: {mean_row[col]:.4f}")

    return df