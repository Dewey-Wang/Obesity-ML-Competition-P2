import joblib
import scanpy as sc

from scripts.cal_propotion import compute_proportion_df


def prop_prediction(

    prediction_h5ad_path="prediction.h5ad",

    model_path="model/lgbm_cell_state_no_whiten.pkl",

    output_csv="predict_program_proportion.csv",

    label_col="pred_cell_state",

    verbose=True
):

    ############################################################
    # load prediction AnnData
    ############################################################

    adata = sc.read_h5ad(

        prediction_h5ad_path

    )

    if verbose:

        print("adata loaded")

        print("X_pca shape:", adata.obsm["X_pca"].shape)


    ############################################################
    # load classifier
    ############################################################

    clf = joblib.load(

        model_path

    )

    if verbose:

        print("model loaded")


    ############################################################
    # predict cell state
    ############################################################

    pred = clf.predict(

        adata.obsm["X_pca"]

    )

    adata.obs[label_col] = pred


    if verbose:

        print("prediction counts:")

        print(

            adata.obs[label_col]

            .value_counts()

        )


    ############################################################
    # compute program proportion
    ############################################################

    df_prop = compute_proportion_df(

        adata,

        label_col,

        save_path=output_csv,

        verbose=verbose

    )


    ############################################################
    # done
    ############################################################

    if verbose:

        print("proportion file saved:", output_csv)


    return df_prop, adata