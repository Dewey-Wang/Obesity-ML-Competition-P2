import pandas as pd

def subset_to_hvg(
    adata,
    hvg_path="../../data/preprocessed/HVG/hvg5000_genes.txt",
    signature_path="../../data/signature_genes.csv",
    include_signature_genes=False,
    verbose=True
):

    ############################################################
    # load HVG in fixed order
    ############################################################

    hvg_genes = pd.read_csv(
        hvg_path,
        header=None
    )[0].tolist()

    if verbose:
        print("HVG requested:", len(hvg_genes))


    ############################################################
    # keep order and avoid duplicates
    ############################################################

    ordered_genes = []
    seen = set()

    for g in hvg_genes:

        if g in adata.var_names and g not in seen:

            ordered_genes.append(g)
            seen.add(g)


    ############################################################
    # optionally add signature genes (append to end)
    ############################################################

    signature_genes = None

    if include_signature_genes:

        sig_df = pd.read_csv(signature_path)

        sig_all = (
            sig_df["sig_gene"]
            .dropna()
            .unique()
            .tolist()
        )

        signature_genes = []

        for g in sig_all:

            if g in adata.var_names and g not in seen:

                ordered_genes.append(g)
                seen.add(g)

                signature_genes.append(g)

        if verbose:

            print("Signature genes requested:", len(sig_all))
            print("Signature genes used:", len(signature_genes))


    ############################################################
    # subset
    ############################################################

    adata_subset = adata[:, ordered_genes].copy()

    if verbose:

        print("Total genes used:", len(ordered_genes))


    return adata_subset, hvg_genes, signature_genes