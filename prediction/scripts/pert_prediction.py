import scanpy as sc
import torch
import joblib
import numpy as np

from pathlib import Path

from scripts.flow_model import FlowMLP
from scripts.flow_pred import sample_all_perts
from scripts.scoring import reconstruct_full_expression_with_control
from scripts.save_predict import save_prediction_h5ad


def pert_prediction(

    # paths
    adata_full_path="../../data/obesity_challenge_2.h5ad",
    control_nc_path="../../data/preprocessed/control_cells/NC_control_cells.h5ad",
    control_ncnc_path="../../data/preprocessed/control_cells/NC_NC_control_cells.h5ad",

    gene_order_path="hvg_gene_order_no_whiten.joblib",
    pca_path="pca_512_no_whiten.joblib",

    model_ckpt_path="model/best_flow_no_whiten.pt",

    perturbation_list_path="../../data/predict_perturbations_2.txt",

    predict_genes_path="../../data/predict_genes_2.txt",

    output_path="prediction.h5ad",

    # model params
    dim=512,
    cond_dim=19,
    hidden=1024,

    n_steps=1000,

    device="cpu",

    cells_per_control=100,

    verbose=True
):

    ############################################################
    # load gene order
    ############################################################

    gene_order = joblib.load(gene_order_path)

    if verbose:
        print("gene order loaded:", len(gene_order))


    ############################################################
    # load PCA
    ############################################################

    pca = joblib.load(pca_path)

    if verbose:
        print("PCA loaded")


    ############################################################
    # load controls
    ############################################################

    adata_ctrl_NC = sc.read_h5ad(control_nc_path)
    adata_ctrl_NCNC = sc.read_h5ad(control_ncnc_path)

    adata_ctrl_NC_hvg = adata_ctrl_NC[:, gene_order].copy()
    adata_ctrl_NCNC_hvg = adata_ctrl_NCNC[:, gene_order].copy()


    ############################################################
    # compute latent for controls
    ############################################################

    def to_numpy(X):

        if not isinstance(X, np.ndarray):

            X = X.toarray()

        return X


    X_ctrl_NC = to_numpy(adata_ctrl_NC_hvg.X)
    X_ctrl_NCNC = to_numpy(adata_ctrl_NCNC_hvg.X)

    z_NC = pca.transform(X_ctrl_NC)
    z_NCNC = pca.transform(X_ctrl_NCNC)

    if verbose:

        print("z_NC shape:", z_NC.shape)
        print("z_NCNC shape:", z_NCNC.shape)


    ############################################################
    # load perturbation list
    ############################################################

    pert_list = Path(
        perturbation_list_path
    ).read_text().splitlines()

    if verbose:

        print("n perturbations:", len(pert_list))


    ############################################################
    # build flow model
    ############################################################

    model = FlowMLP(

        dim=dim,

        cond_dim=cond_dim,

        hidden=hidden

    ).to(device)


    checkpoint = torch.load(

        model_ckpt_path,

        map_location=device

    )


    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    model.eval()

    if verbose:

        print("loaded epoch:", checkpoint["epoch"])
        print("best loss:", checkpoint["loss"])


    ############################################################
    # sample latent per perturbation
    ############################################################

    latent_all, pert_labels = sample_all_perts(

        model,

        pert_list,

        z_NC=z_NC,

        z_NCNC=z_NCNC,

        n_steps=n_steps
    )

    if verbose:

        print("latent_all:", latent_all.shape)


    ############################################################
    # decode latent → HVG expression
    ############################################################

    gene_pred = pca.inverse_transform(

        latent_all

    )

    gene_pred = np.clip(

        gene_pred,

        0,

        None

    )


    ############################################################
    # reconstruct full gene expression
    ############################################################

    adata_full = sc.read_h5ad(

        adata_full_path

    )

    X_ctrl_NC_full = to_numpy(
        adata_ctrl_NC.X
    )[:cells_per_control]

    X_ctrl_NCNC_full = to_numpy(
        adata_ctrl_NCNC.X
    )[:cells_per_control]


    X_pred_full = reconstruct_full_expression_with_control(

        adata_full=adata_full,

        adata_hvg=adata_ctrl_NC_hvg,

        X_hvg_rec=gene_pred,

        pert_list=pert_list,

        X_ctrl_NC=X_ctrl_NC_full,

        X_ctrl_NCNC=X_ctrl_NCNC_full

    )


    ############################################################
    # save prediction.h5ad
    ############################################################

    adata_pred = save_prediction_h5ad(

        X_pred_full=X_pred_full,

        latent_all=latent_all,

        adata_full=adata_full,

        predict_perturbations=pert_list,

        genes_to_predict_path=predict_genes_path,

        prediction_h5ad_file_path=output_path,

    )


    if verbose:

        print("prediction saved to:", output_path)


    return adata_pred, latent_all, pert_labels