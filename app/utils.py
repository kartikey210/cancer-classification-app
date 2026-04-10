import pandas as pd
from sklearn.decomposition import PCA

def get_top_features(model, feature_names, top_n=10):
    importance = model.feature_importances_

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    df = df.sort_values(by="importance", ascending=False)
    return df.head(top_n)


def apply_pca(X):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    df_pca = pd.DataFrame(components, columns=["PC1", "PC2"])
    return df_pca