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

        result = {

            "gene": g,

            "pre_adipo":
                (subset[label_col] == "pre_adipo").mean(),

            "adipo":
                (subset[label_col] == "adipo").mean(),

            "other":
                (subset[label_col] == "other").mean(),

            "lipo":
                (subset[label_col] == "lipo").mean(),
        }

        result["lipo_adipo"] = (

            result["lipo"] /

            (result["adipo"] + 1e-20)

        )

        rows.append(result)

    df = pd.DataFrame(rows)

    # competition column order
    df = df[[
        "gene",
        "pre_adipo",
        "adipo",
        "other",
        "lipo",
        "lipo_adipo"
    ]]

    # optional save
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