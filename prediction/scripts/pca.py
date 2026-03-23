from sklearn.decomposition import PCA
import numpy as np
import joblib


def compute_pca_latent(
    adata,
    save_path,
    d=512,
    seed=42,
    whiten = False
    
):
    
    # HVG matrix
    X_hvg = adata.X
    
    if not isinstance(X_hvg, np.ndarray):
        X_hvg = X_hvg.toarray()

    pca = PCA(
        n_components=d,
        whiten=whiten,
        random_state=seed
    )

    X_latent = pca.fit_transform(X_hvg)

    # store in adata
    adata.obsm["X_pca"] = X_latent

    # save PCA for inverse transform
    joblib.dump(
        pca,
        save_path
    )
    print(f"save in {save_path}")
    return adata, pca