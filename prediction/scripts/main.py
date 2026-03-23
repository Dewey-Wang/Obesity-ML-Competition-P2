import argparse

from scripts.pert_prediction import pert_prediction
from scripts.prop_prediction import prop_prediction


############################################################
# TRAIN
############################################################

def train():
    print("Train Done")



############################################################
# INFER
############################################################

def infer(

    output_h5ad="prediction.h5ad",

    output_prop="predict_program_proportion.csv"

):

    """
    Generate competition outputs.

    Outputs:

    prediction.h5ad
    predict_program_proportion.csv
    """

    print("INFER MODE")

    ########################################################
    # 1️⃣ generate predicted expression
    ########################################################

    _, _, _ = pert_prediction(

            adata_full_path="../../data/obesity_challenge_2.h5ad",
            control_nc_path="../../data/preprocessed/control_cells/NC_control_cells.h5ad",
            control_ncnc_path="../../data/preprocessed/control_cells/NC_NC_control_cells.h5ad",

            gene_order_path="hvg_gene_order_no_whiten.joblib",
            pca_path="pca_512_no_whiten.joblib",

            model_ckpt_path="model/best_flow_no_whiten.pt",

            perturbation_list_path="../../data/predict_perturbations_2.txt",

            predict_genes_path="../../data/predict_genes_2.txt",

            output_path=output_h5ad

        )


    ########################################################
    # 2️⃣ predict cell state proportion
    ########################################################

    _, _ = prop_prediction(

        prediction_h5ad_path=output_h5ad,

        model_path="model/lgbm_cell_state_no_whiten.pkl",
        
        label_col="pred_cell_state",
        
        output_csv=output_prop

    )


    print("Inference finished")

    print("Saved files:")
    print(output_h5ad)
    print(output_prop)