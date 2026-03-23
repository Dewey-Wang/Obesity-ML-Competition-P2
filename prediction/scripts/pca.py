from sklearn.decomposition import PCA
import numpy as np
import joblib


def compute_pca_latent(
    adata,
    d=512,
    seed=42,
):
    
    # HVG matrix
    X_hvg = adata.X
    
    if not isinstance(X_hvg, np.ndarray):
        X_hvg = X_hvg.toarray()

    pca = PCA(
        n_components=d,
        random_state=seed
    )

    X_latent = pca.fit_transform(X_hvg)

    # store in adata
    adata.obsm["X_pca"] = X_latent

    # save PCA for inverse transform
    joblib.dump(
        pca,
        f"pca_{d}.joblib"
    )
    print(f"save in pca_{d}.joblib")
    return adata, pca