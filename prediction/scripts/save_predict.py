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

    latent_all=None,

    genes_to_predict_path="data/predict_genes_2.txt",

    cells_per_perturbation=100,

    dtype=np.float32

):

    """
    Save prediction in official competition format.

    保證：
    gene order 完全符合 predict_genes_2.txt

    步驟：
    1. 以 adata_full.var_names 為原始順序
    2. 按 txt 檔順序重排 columns
    """


    ############################################################
    # load gene list (target order)
    ############################################################

    genes_to_predict = load_gene_list(
        genes_to_predict_path
    )

    print("genes_to_predict:", len(genes_to_predict))


    ############################################################
    # sanity check cell count
    ############################################################

    n_perturb = len(predict_perturbations)

    expected_n_cells = (
        n_perturb * cells_per_perturbation
    )


    assert X_pred_full.shape[0] == expected_n_cells, \
        f"Cell number mismatch {X_pred_full.shape[0]} != {expected_n_cells}"


    if latent_all is not None:

        assert latent_all.shape[0] == expected_n_cells, \
            f"latent cell mismatch {latent_all.shape[0]} != {expected_n_cells}"


    ############################################################
    # build gene index mapping from ORIGINAL adata order
    ############################################################

    original_gene_order = list(
        adata_full.var_names
    )


    gene_index_map = {

        gene: idx

        for idx, gene in enumerate(
            original_gene_order
        )

    }


    ############################################################
    # check missing genes
    ############################################################

    missing_genes = [

        g for g in genes_to_predict

        if g not in gene_index_map

    ]


    if len(missing_genes) > 0:

        raise ValueError(

            f"{len(missing_genes)} genes missing\n"

            f"example missing: {missing_genes[:10]}"

        )


    ############################################################
    # reorder columns to match txt order
    ############################################################

    col_idx = [

        gene_index_map[g]

        for g in genes_to_predict

    ]


    X_final = X_pred_full[:, col_idx]


    ############################################################
    # convert dtype
    ############################################################

    X_final = X_final.astype(dtype)


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
    # create AnnData
    ############################################################

    adata_pred = ad.AnnData(

        X=X_final,

        obs=obs,

        var=var

    )


    ############################################################
    # store latent
    ############################################################

    if latent_all is not None:

        adata_pred.obsm["X_latent"] = latent_all.astype(dtype)

        print(

            "latent stored:",

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

        "\nSaved prediction:",

        prediction_h5ad_file_path

    )


    ############################################################
    # final verification
    ############################################################

    assert list(adata_pred.var_names) == genes_to_predict,"gene order mismatch"


    print(

        "gene order verified"

    )


    return adata_pred