import numpy as np
import pandas as pd
from pathlib import Path


def compute_proportion_df(
    adata,
    label_col,
    save_path=None,
    verbose=True
):

    rows = []

    for g in adata.obs["gene"].unique():

        subset = adata.obs[
            adata.obs["gene"] == g
        ]

        ########################################################
        # counts
        ########################################################

        n = len(subset)

        pre_adipo = (subset[label_col] == "pre_adipo").sum()

        adipo_only = (subset[label_col] == "adipo").sum()

        lipo_only = (subset[label_col] == "lipo").sum()

        lipo_adipo = (subset[label_col] == "lipo_adipo").sum()

        other = (subset[label_col] == "other").sum()

        ########################################################
        # combine programs
        ########################################################

        adipo_total = adipo_only + lipo_adipo

        lipo_total = lipo_only + lipo_adipo

        ########################################################
        # proportions
        ########################################################

        result = {

            "gene": g,

            "pre_adipo":
                pre_adipo / n,

            "adipo":
                adipo_total / n,

            "lipo":
                lipo_total / n,

            "other":
                other / n,
        }

        ########################################################
        # lipo/adipo ratio
        ########################################################

        result["lipo_adipo"] = (

            result["lipo"] /

            (result["adipo"] + 1e-20)

        )

        rows.append(result)

    df = pd.DataFrame(rows)

    ############################################################
    # competition column order
    ############################################################

    df = df[[
        "gene",
        "pre_adipo",
        "adipo",
        "other",
        "lipo",
        "lipo_adipo"
    ]]

    ############################################################
    # save
    ############################################################

    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            save_path,
            index=False
        )

        if verbose:

            print(f"Saved to: {save_path}")

            print("shape:", df.shape)

    return df