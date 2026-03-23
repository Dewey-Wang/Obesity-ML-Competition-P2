import numpy as np
import pandas as pd
import anndata as ad
import os


def load_gene_list(gene_path):
    """
    Load gene list txt file (one gene per line).
    Keeps exact order.
    """

    with open(gene_path) as f:
        genes = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return genes


def save_prediction_h5ad(

    X_pred_full,
    adata_full,

    predict_perturbations,

    prediction_h5ad_file_path,

    latent_all=None,   # <-- 新增

    genes_to_predict_path="data/predict_genes_2.txt",

    cells_per_perturbation=100,

    dtype=np.float32
):

    """
    Save reconstructed gene expression matrix
    into official competition format.

    genes_to_predict loaded automatically from txt file.

    Optional:
    latent_all -> saved into .obsm["X_pca"]
    """

    ############################################################
    # load gene list
    ############################################################

    genes_to_predict = load_gene_list(
        genes_to_predict_path
    )

    print("genes_to_predict:", len(genes_to_predict))


    ############################################################
    # sanity check
    ############################################################

    n_perturb = len(predict_perturbations)

    expected_n_cells = (
        n_perturb * cells_per_perturbation
    )

    assert X_pred_full.shape[0] == expected_n_cells, \
        f"Cell number mismatch {X_pred_full.shape[0]} != {expected_n_cells}"


    if latent_all is not None:

        assert latent_all.shape[0] == expected_n_cells, \
            f"latent cell number mismatch {latent_all.shape[0]} != {expected_n_cells}"


    ############################################################
    # reorder genes to match txt file order
    ############################################################

    gene_index_map = {

        g: i

        for i, g in enumerate(

            adata_full.var_names

        )

    }


    missing_genes = [

        g for g in genes_to_predict

        if g not in gene_index_map

    ]


    if len(missing_genes) > 0:

        raise ValueError(

            f"{len(missing_genes)} genes missing "

            f"(example: {missing_genes[:5]})"

        )


    col_idx = [

        gene_index_map[g]

        for g in genes_to_predict

    ]


    X_final = X_pred_full[:, col_idx].astype(dtype)


    ############################################################
    # obs
    ############################################################

    obs = pd.DataFrame({

        "gene": np.repeat(

            predict_perturbations,

            cells_per_perturbation

        )

    })


    ############################################################
    # var
    ############################################################

    var = pd.DataFrame(

        index=genes_to_predict

    )


    ############################################################
    # AnnData
    ############################################################

    adata_pred = ad.AnnData(

        X=X_final,

        obs=obs,

        var=var

    )


    ############################################################
    # save latent representation
    ############################################################

    if latent_all is not None:

        adata_pred.obsm["X_pca"] = latent_all.astype(dtype)

        print(
            "latent stored in .obsm['X_pca']:",
            latent_all.shape
        )


    ############################################################
    # save
    ############################################################

    if os.path.exists(
        prediction_h5ad_file_path
    ):

        os.remove(
            prediction_h5ad_file_path
        )


    adata_pred.write_h5ad(
        prediction_h5ad_file_path
    )


    print(
        "Saved prediction:",
        prediction_h5ad_file_path
    )


    return adata_pred