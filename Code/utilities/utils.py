import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    confusion_matrix

)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.optimize import linear_sum_assignment


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def purity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)


def clustering_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / cm.sum()


def pca_by_variance(X, target_var=0.9):
    pca = PCA(n_components=target_var, svd_solver="full", random_state=42)
    Xr = pca.fit_transform(X)
    return Xr, pca


def get_metrics(X, y, y_pred):
    nmi = normalized_mutual_info_score(y, y_pred)
    ari = adjusted_rand_score(y, y_pred)
    purity = purity_score(y, y_pred)
    accuracy = clustering_accuracy(y, y_pred)
    sil = silhouette_score(X, y_pred)
    return {
        "NMI": round(nmi, 4),
        "ARI": round(ari, 4),
        "Purity": round(purity, 4),
        "Accuracy": round(accuracy, 4),
        "Silhouette": round(sil, 4)
    }


def plot_and_save_clusters(dataset, encoder_name, clusterer, X, y_pred, n_clusters, number_of_components):
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, s=5)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(
        f"{dataset}-{encoder_name} - {clusterer} {n_clusters} Clusters (PCA)")
    save_path = f"../outputs/plots/{dataset}_{encoder_name}_{clusterer.lower()}_clusters_k{n_clusters}_components{number_of_components}.png"
    plt.savefig(save_path, dpi=500)
    plt.show()


def find_best_k_silhouette(
    X,
    k_min=2,
    k_max=30,
    sample_size=10000,
    use_minibatch=True,
    random_state=42
):

    N = len(X)

    if N > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(N, size=sample_size, replace=False)
        X_eval = X[idx]
    else:
        X_eval = X

    scores = {}

    for k in range(k_min, k_max + 1):

        if use_minibatch and len(X_eval) > 20000:
            model = MiniBatchKMeans(
                n_clusters=k,
                batch_size=2048,
                n_init=10,
                random_state=random_state
            )
        else:
            model = KMeans(
                n_clusters=k,
                n_init=10,
                random_state=random_state
            )

        labels = model.fit_predict(X_eval)

        # avoid invalid silhouette cases
        if len(set(labels)) < 2:
            scores[k] = -1
            continue

        score = silhouette_score(X_eval, labels)
        scores[k] = score

    best_k = max(scores, key=scores.get)

    return best_k, scores


def log_experiment(
    csv_path,
    model_name,
    dataset_name,
    encoder_name,
    n_rows,
    pca_components,
    nmi,
    ari,
    acc,
    purity
):

    row = {
        "model": model_name,
        "dataset": dataset_name,
        "encoder": encoder_name,
        "rows": n_rows,
        "pca_components": pca_components,
        "NMI": nmi,
        "ARI": ari,
        "ACC": acc,
        "Purity": purity
    }

    csv_path = Path(csv_path)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(csv_path, index=False)
