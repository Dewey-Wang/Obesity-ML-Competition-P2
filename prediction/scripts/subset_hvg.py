import scanpy as sc
import pandas as pd


def subset_to_hvg(
    adata,
    hvg_path="../../data/preprocessed/HVG/hvg5000_genes.txt",
    signature_path="../../data/signature_genes.csv",
    include_signature_genes=False,
    verbose=True
):
    """
    Subset AnnData to HVG genes, optionally including signature genes.

    Returns
    -------
    adata_subset : AnnData

    hvg_genes : list
        original HVG genes (no filtering)

    signature_genes : list or None
        signature genes present in dataset
        returned only when include_signature_genes=True
    """

    # ---- load HVG ----
    hvg_genes = pd.read_csv(
        hvg_path,
        header=None
    )[0].tolist()

    if verbose:
        print("HVG requested:", len(hvg_genes))

    final_genes = set(hvg_genes)

    signature_genes = None

    # ---- optionally load signature genes ----
    if include_signature_genes:

        sig_df = pd.read_csv(signature_path)

        sig_all = (
            sig_df["sig_gene"]
            .dropna()
            .unique()
            .tolist()
        )

        # keep only genes present in dataset
        signature_genes = [
            g for g in sig_all
            if g in adata.var_names
        ]

        final_genes = final_genes.union(signature_genes)

        if verbose:

            print("Signature genes requested:", len(sig_all))
            print("Signature genes found in dataset:", len(signature_genes))
    # ---- subset genes that exist in dataset ----
    overlap = [
        g for g in final_genes
        if g in adata.var_names
    ]

    missing = set(final_genes) - set(overlap)

    if verbose:

        print("Total genes used:", len(overlap))
        print("Missing genes:", len(missing))

    adata_subset = adata[:, overlap].copy()

    return adata_subset, hvg_genes, signature_genes