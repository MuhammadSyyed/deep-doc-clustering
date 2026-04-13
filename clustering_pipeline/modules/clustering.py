import numpy as np
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseClusterer(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def fit_predict(self, embeddings: np.ndarray, k: int) -> np.ndarray:
        pass


class KMeansClusterer(BaseClusterer):
    def fit_predict(self, embeddings: np.ndarray, k: int) -> np.ndarray:
        c = self.cfg.get("kmeans", {})
        model = KMeans(
            n_clusters=k,
            n_init=c.get("n_init", 20),
            max_iter=c.get("max_iter", 300),
            random_state=42,
        )
        return model.fit_predict(embeddings)


class AutoEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


class AEClusterer(BaseClusterer):
    def fit_predict(self, embeddings: np.ndarray, k: int) -> np.ndarray:
        c = self.cfg.get("ae", {})

        device = torch.device(c.get("device", "cpu"))
        X = torch.tensor(embeddings, dtype=torch.float32).to(device)

        model = AutoEncoder(
            X.shape[1],
            c.get("hidden_dim", 256),
            c.get("latent_dim", 64),
        ).to(device)

        opt = torch.optim.Adam(
            model.parameters(),
            lr=c.get("lr", 1e-3),
            weight_decay=1e-5
        )

        min_delta = float(c.get("min_delta", 1e-4))
        patience = int(c.get("patience", 20))

        best_loss = float("inf")
        counter = 0
        best_state = None

        for _ in range(c.get("epochs", 500)):
            z, x_hat = model(X)

            loss = F.mse_loss(x_hat, X)

            opt.zero_grad()
            loss.backward()
            opt.step()

            curr_loss = loss.item()

            if curr_loss < best_loss - min_delta:
                best_loss = curr_loss
                counter = 0
                best_state = model.state_dict()
            else:
                counter += 1

            if counter >= patience:
                print("AE early stopping triggered")
                break

        if best_state:
            model.load_state_dict(best_state)

        with torch.no_grad():
            z, _ = model(X)
            Z = z.cpu().numpy()

        Z = normalize(Z)

        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(Z)

        self.probs_ = None
        return labels

    def predict_proba(self):
        return getattr(self, "probs_", None)


class IDEC(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, n_clusters, alpha=1.0):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        dist = torch.cdist(z, self.centers) ** 2
        q = (1 + dist / self.alpha) ** (-(self.alpha + 1) / 2)
        q = q / q.sum(dim=1, keepdim=True)

        return z, x_hat, q

    def target(self, q):
        w = q ** 2 / q.sum(0)
        return w / w.sum(1, keepdim=True)


class IDECClusterer(BaseClusterer):
    def fit_predict(self, embeddings: np.ndarray, k: int) -> np.ndarray:
        c = self.cfg.get("idec", {})

        device = torch.device(c.get("device", "cpu"))
        X = torch.tensor(embeddings, dtype=torch.float32).to(device)

        model = IDEC(
            in_dim=X.shape[1],
            hidden_dim=c.get("hidden_dim", 256),
            latent_dim=c.get("latent_dim", 64),
            n_clusters=k,
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=c.get("lr", 1e-3))

        min_delta = float(c.get("min_delta", 1e-4))
        patience = int(c.get("patience", 20))

        best_loss = float("inf")
        counter = 0
        best_state = None

        for _ in range(c.get("pretrain_epochs", 500)):
            z, x_hat, _ = model(X)
            loss = F.mse_loss(x_hat, X)

            opt.zero_grad()
            loss.backward()
            opt.step()

            curr_loss = loss.item()

            if curr_loss < best_loss - min_delta:
                best_loss = curr_loss
                counter = 0
                best_state = model.state_dict()
            else:
                counter += 1

            if counter >= patience:
                print("Pretraining early stopping")
                break

        if best_state:
            model.load_state_dict(best_state)

        with torch.no_grad():
            z, _, _ = model(X)
            z_np = z.cpu().numpy()

        km = KMeans(n_clusters=k, n_init=20)
        model.centers.data = torch.tensor(
            km.fit(z_np).cluster_centers_,
            dtype=torch.float32
        ).to(device)

        tol = float(c.get("tol", 1e-3))
        update_interval = c.get("update_interval", 5)

        best_loss = float("inf")
        counter = 0
        best_state = None

        prev_labels = None

        for epoch in range(c.get("epochs", 500)):
            z, x_hat, q = model(X)

            if epoch % update_interval == 0:
                curr_labels = q.argmax(dim=1)

                if prev_labels is not None:
                    delta = (curr_labels != prev_labels).float().mean().item()

                    if delta < tol:
                        print(
                            f"Converged at epoch {epoch} (delta={delta:.6f})")
                        break

                prev_labels = curr_labels.clone()

            p = model.target(q).detach()

            loss_kl = F.kl_div(q.log(), p, reduction="batchmean")
            loss_rec = F.mse_loss(x_hat, X)

            loss = loss_kl + c.get("lambda_rec", 1.0) * loss_rec

            opt.zero_grad()
            loss.backward()
            opt.step()

            curr_loss = loss.item()

            if curr_loss < best_loss - min_delta:
                best_loss = curr_loss
                counter = 0
                best_state = model.state_dict()
            else:
                counter += 1

            if counter >= patience:
                print("Loss-based early stopping triggered")
                break

        if best_state:
            model.load_state_dict(best_state)

        with torch.no_grad():
            _, _, q = model(X)
            q = q.cpu().numpy()

        self.probs_ = q
        return q.argmax(axis=1)

    def predict_proba(self):
        return getattr(self, "probs_", None)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, X, A):
        return F.relu(self.linear(torch.sparse.mm(A, X)))


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.l1 = GCNLayer(in_dim, hidden)
        self.l2 = GCNLayer(hidden, out_dim)

    def forward(self, X, A):
        h = self.l1(X, A)
        return self.l2(h, A)


def build_graph(X, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nbrs.kneighbors(X)

    rows, cols = [], []
    n = len(X)

    for i in range(n):
        for j in idx[i, 1:]:
            rows += [i, j]
            cols += [j, i]

    rows += list(range(n))
    cols += list(range(n))

    idx = torch.tensor([rows, cols])
    data = torch.ones(len(rows))

    assert idx.shape[0] == 2, "idx must be [2, nnz]"
    assert idx.shape[1] == data.shape[0], "Mismatch idx/data"

    assert idx.min() >= 0, "Negative indices found"
    assert idx.max() < n, f"Index out of bounds: max={idx.max()}, n={n}"

    A = torch.sparse_coo_tensor(idx, data, (n, n)).coalesce()

    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
    d = deg.pow(-0.5)

    A = A.to_dense()
    A = d.unsqueeze(1) * A * d.unsqueeze(0)

    return A.to_sparse()


class GCNClusterer(BaseClusterer):
    def fit_predict(self, embeddings: np.ndarray, k: int) -> np.ndarray:
        c = self.cfg.get("gcn", {})

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X = torch.tensor(embeddings, dtype=torch.float32).to(device)
        A = build_graph(embeddings, c.get("knn_k", 10)).to(device)

        model = GCN(
            X.shape[1],
            c.get("hidden_dim", 128),
            c.get("out_dim", 64),
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=c.get("lr", 1e-3))

        with torch.no_grad():
            target = torch.sparse.mm(A, X)

        best_val_loss = float('inf')
        patience = 20
        counter = 0

        train_idx, val_idx = train_test_split(
            torch.arange(X.shape[0]), test_size=0.2, random_state=42
        )

        for epoch in range(500):
            model.train()

            Z = model(X, A)
            loss = F.mse_loss(Z[train_idx], target[train_idx][:, :Z.shape[1]])

            opt.zero_grad()
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                Z = model(X, A)
                val_loss = F.mse_loss(
                    Z[val_idx], target[val_idx][:, :Z.shape[1]])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                best_state = model.state_dict()
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping triggered")
                break

        model.load_state_dict(best_state)

        with torch.no_grad():
            Z = model(X, A).cpu().numpy()

        Z = normalize(Z)

        return KMeansClusterer(self.cfg).fit_predict(Z, k)


_REGISTRY = {
    "kmeans": KMeansClusterer,
    "ae": AEClusterer,
    "dec": IDECClusterer,
    "gcn": GCNClusterer,
}

def get_clusterer(cfg: dict) -> BaseClusterer:
    name = cfg["name"].lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown clusterer: {name}")
    return _REGISTRY[name](cfg)
