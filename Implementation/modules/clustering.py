import numpy as np
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import faiss
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def target_distribution(q):
    weight = q ** 2 / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()

class BaseClusterer(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def fit_predict(self, embeddings: np.ndarray, k: int, true_labels=None) -> np.ndarray:
        pass

class KMeansClusterer(BaseClusterer):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.training_history = []

    def fit_predict(self, embeddings: np.ndarray, k: int, true_labels=None) -> np.ndarray:
        c = self.cfg.get("kmeans", {})
        model = KMeans(
            n_clusters=k,
            n_init=c.get("n_init", 20),
            max_iter=c.get("max_iter", 300),
            random_state=42,
        )
        labels = model.fit_predict(embeddings)
        self.training_history = [{
            "epoch": 0,
            "inertia": float(model.inertia_),
            "n_iter": int(model.n_iter_),
        }]
        return labels

class DECAE(nn.Module):
    def __init__(self, n_input, n_enc_1=256, n_enc_2=128, n_enc_3=64, n_z=32):
        super().__init__()

        # Encoder
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)
        self.z_layer = nn.Linear(n_enc_3, n_z)

        # Decoder
        self.dec_1 = nn.Linear(n_z, n_enc_3)
        self.dec_2 = nn.Linear(n_enc_3, n_enc_2)
        self.dec_3 = nn.Linear(n_enc_2, n_enc_1)
        self.x_bar = nn.Linear(n_enc_1, n_input)

    def encode(self, x):
        h1 = F.relu(self.enc_1(x))
        h2 = F.relu(self.enc_2(h1))
        h3 = F.relu(self.enc_3(h2))
        z = self.z_layer(h3)
        return z

    def decode(self, z):
        d1 = F.relu(self.dec_1(z))
        d2 = F.relu(self.dec_2(d1))
        d3 = F.relu(self.dec_3(d2))
        x_hat = self.x_bar(d3)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

class IDECClusterer(BaseClusterer):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.training_history = []

    @staticmethod
    def target_distribution(q):
        weight = (q ** 2) / torch.sum(q, dim=0)
        return (weight.t() / torch.sum(weight, dim=1)).t()

    def soft_assign(self, z, centers):
        dist = torch.cdist(z, centers) ** 2
        q = 1.0 / (1.0 + dist)
        return q / q.sum(dim=1, keepdim=True)

    def fit_predict(self, embeddings: np.ndarray, k: int, true_labels=None) -> np.ndarray:
        torch.manual_seed(42)
        np.random.seed(42)
        self.training_history = []

        c = self.cfg.get("idec", {})
        device = torch.device(c.get("device", "cpu"))

        X = torch.tensor(embeddings, dtype=torch.float32).to(device)
        p_all = None

        dataset = TensorDataset(X, torch.arange(len(X)))
        loader = DataLoader(dataset, batch_size=c.get(
            "batch_size", 256), shuffle=True)

        model = DECAE(n_input=X.shape[1]).to(device)

        model.cluster_centers = Parameter(
            torch.zeros(k, c.get("latent_dim", 64), device=device)
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(c.get("lr", 1e-3)))

        for epoch in range(c.get("pretrain_epochs", 50)):
            epoch_pretrain_loss = 0.0
            batch_count = 0
            for batch, _ in loader:
                batch = batch.to(device)
                x_hat, z = model(batch)
                loss = F.mse_loss(x_hat, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_pretrain_loss += loss.item()
                batch_count += 1

            if batch_count > 0:
                self.training_history.append({
                    "stage": "pretrain",
                    "epoch": int(epoch),
                    "reconstruction_loss": float(epoch_pretrain_loss / batch_count),
                })

        with torch.no_grad():
            z = model.encode(X).cpu().numpy()

        kmeans = KMeans(n_clusters=k, n_init=20)
        y_pred = kmeans.fit_predict(z)

        model.cluster_centers.data = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32
        ).to(device)

        model.train()
        for epoch in range(c.get("epochs", 100)):
            epoch_total_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            batch_count = 0
            delta_val = None

            if epoch % c.get("update_interval", 10) == 0:
                with torch.no_grad():
                    x_hat, z = model(X)
                    q_all = self.soft_assign(z, model.cluster_centers)
                    q_all = torch.clamp(q_all, min=1e-10)
                    p_all = self.target_distribution(q_all).to(device)

                    y_pred_new = q_all.argmax(1).cpu().numpy()
                    delta = np.mean(y_pred != y_pred_new)
                    delta_val = float(delta)
                    y_pred = y_pred_new

                    if delta < float(c.get("tol", 1e-3)):
                        self.training_history.append({
                            "stage": "cluster",
                            "epoch": int(epoch),
                            "total_loss": None,
                            "reconstruction_loss": None,
                            "kl_loss": None,
                            "delta": delta_val,
                            "converged": True,
                        })
                        break

            for batch, idx in loader:
                if p_all is None:
                    continue
                batch = batch.to(device)
                idx = idx.to(device)

                x_hat, z = model(batch)

                dist = torch.cdist(z, model.cluster_centers) ** 2
                q = 1.0 / (1.0 + dist)
                q = q / q.sum(dim=1, keepdim=True)
                q = torch.clamp(q, min=1e-10)

                p = p_all[idx]

                recon_loss = F.mse_loss(x_hat, batch)
                kl_loss = F.kl_div(q.log(), p, reduction="batchmean")
                total_loss = recon_loss + c.get("lambda_kl", 1.0) * kl_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                batch_count += 1

            if batch_count > 0:
                self.training_history.append({
                    "stage": "cluster",
                    "epoch": int(epoch),
                    "total_loss": float(epoch_total_loss / batch_count),
                    "reconstruction_loss": float(epoch_recon_loss / batch_count),
                    "kl_loss": float(epoch_kl_loss / batch_count),
                    "delta": delta_val,
                })

        with torch.no_grad():
            x_hat, z = model(X)
            q = self.soft_assign(z, model.cluster_centers)
            return q.argmax(1).cpu().numpy()

class AE(nn.Module):
    def __init__(self, n_input, n_enc_1=256, n_enc_2=128, n_enc_3=64, n_z=32):
        super().__init__()

        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)
        self.z_layer = nn.Linear(n_enc_3, n_z)

        self.dec_1 = nn.Linear(n_z, n_enc_3)
        self.dec_2 = nn.Linear(n_enc_3, n_enc_2)
        self.dec_3 = nn.Linear(n_enc_2, n_enc_1)
        self.x_bar = nn.Linear(n_enc_1, n_input)

    def forward(self, x):
        h1 = F.relu(self.enc_1(x))
        h2 = F.relu(self.enc_2(h1))
        h3 = F.relu(self.enc_3(h2))
        z = self.z_layer(h3)

        d1 = F.relu(self.dec_1(z))
        d2 = F.relu(self.dec_2(d1))
        d3 = F.relu(self.dec_3(d2))
        x_hat = self.x_bar(d3)

        return x_hat, h1, h2, h3, z

def cluster_delta(y_prev, y_curr):
    cm = confusion_matrix(y_prev, y_curr)
    row_ind, col_ind = linear_sum_assignment(-cm)
    matched = cm[row_ind, col_ind].sum()
    return 1 - matched / len(y_prev)

def build_sbert_graph(X, topk=10):
    
    X = X.astype(np.float32).copy()
    
    N, d = X.shape

    faiss.normalize_L2(X)

    index = faiss.IndexFlatIP(d)
    index.add(X)

    sims, indices = index.search(X, topk + 1)

    rows, cols, vals = [], [], []

    for i in range(N):
        for j, sim in zip(indices[i][1:], sims[i][1:]):
            rows.append(i)
            cols.append(j)
            vals.append(sim)

    # build sparse matrix
    adj = sp.coo_matrix((vals, (rows, cols)), shape=(N, N))

    # symmetrize
    adj = adj.maximum(adj.T)

    adj = adj + sp.eye(adj.shape[0])

    # normalize
    deg = np.array(adj.sum(1)).flatten()
    deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-8)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)

    adj = D_inv_sqrt @ adj @ D_inv_sqrt

    adj = adj.tocsr()
    adj.eliminate_zeros()

    return adj

def sparse_to_torch(adj):
    adj = adj.tocoo()
    indices = torch.from_numpy(
        np.vstack((adj.row, adj.col))
    ).long()
    values = torch.from_numpy(adj.data).float()
    shape = torch.Size(adj.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x, adj, active=True):
        x = self.linear(x)
    
        if adj.is_sparse:
            x = torch.sparse.mm(adj, x)
        else:
            x = torch.mm(adj, x)
    
        if active:
            x = F.relu(x)
        return x

class SDCN(nn.Module):
    def __init__(self, n_input, n_z, n_clusters):
        super().__init__()

        # AE
        self.ae = AE(n_input=n_input, n_z=n_z)

        # GCN
        self.gnn_1 = GCNLayer(n_input, 256)
        self.gnn_2 = GCNLayer(256, 128)
        self.gnn_3 = GCNLayer(128, 64)
        self.gnn_4 = GCNLayer(64, n_z)
        self.gnn_5 = GCNLayer(n_z, n_clusters)

        # clustering
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z))
        nn.init.xavier_normal_(self.cluster_layer.data)

        self.v = 1.0

    def soft_assign(self, z):
        dist = torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.v)
        q = q ** ((self.v + 1.0) / 2.0)
        return q / q.sum(dim=1, keepdim=True)

    def forward(self, x, adj):
        # AE forward
        x_bar, h1, h2, h3, z = self.ae(x)

        sigma = 0.5

        # GCN with fusion
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * h1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * h2, adj)
        h = self.gnn_4((1 - sigma) * h + sigma * h3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)

        predict = F.log_softmax(h, dim=1)

        # AE clustering
        q = self.soft_assign(z)

        return x_bar, q, predict, z

class SDCNClusterer(BaseClusterer):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.training_history = []

    def fit_predict(self, embeddings: np.ndarray, k: int, true_labels=None) -> np.ndarray:

        min_epochs = 20

        torch.manual_seed(42)

        np.random.seed(42)
        self.training_history = []

        cfg = self.cfg.get("sdcn", {})

        device = torch.device(cfg.get("device"))

        if cfg.get("use_pca", False):
            X_np = PCA(n_components=0.95).fit_transform(embeddings)
        else:
            X_np = embeddings
        
        adj = build_sbert_graph(X_np, cfg.get("knn_k", 5))
        
        X = torch.tensor(X_np, dtype=torch.float32, device=device)

        adj = sparse_to_torch(adj).coalesce().to(device)

        if adj._values().numel() > 0 and torch.isnan(adj._values()).any():

            raise ValueError("Adjacency matrix contains NaNs")

        model = SDCN(
            n_input=X.shape[1],
            n_z=cfg.get("latent_dim", 64),
            n_clusters=k
        ).to(device)

        optimizer_ae = torch.optim.Adam(
            model.parameters(), lr=float(cfg.get("lr", 1e-3))
        )

        for epoch in range(cfg.get("pretrain_epochs", 50)):
            x_hat, *_ = model.ae(X)
            loss = F.mse_loss(x_hat, X)

            optimizer_ae.zero_grad()
            loss.backward()
            optimizer_ae.step()
            self.training_history.append({
                "stage": "pretrain",
                "epoch": int(epoch),
                "reconstruction_loss": float(loss.item()),
            })

        with torch.no_grad():
            _, _, _, _, z = model.ae(X)

        kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)

        z_np = z.detach().cpu().numpy()
        y_pred = kmeans.fit_predict(z_np)
        y_pred_last = y_pred

        model.cluster_layer.data = torch.tensor( kmeans.cluster_centers_, dtype=torch.float32, device=device )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(cfg.get("lr", 1e-3))
        )

        for epoch in range(cfg.get("epochs", 200)):

            # Forward
            x_bar, q, pred, _ = model(X, adj)

            # Stabilize
            q = torch.clamp(q, min=1e-10)

            # Target distribution
            p = target_distribution(q).detach()

            # Losses
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
            ce_loss = F.kl_div(pred, p, reduction='batchmean', log_target=False)
            re_loss = F.mse_loss(x_bar, X)

            loss = (
                cfg.get("alpha") * kl_loss +
                cfg.get("beta") * ce_loss +
                re_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            

            converged_this_epoch = False
            if epoch > min_epochs and epoch % cfg.get("update_interval", 10) == 0:
                with torch.no_grad():
                    y_pred = q.argmax(1).cpu().numpy()
                    delta = cluster_delta(y_pred_last, y_pred)
                    y_pred_last = y_pred
            
                if delta < float(cfg.get("tol", 1e-3)):
                    print(f"[SDCN] Converged at epoch {epoch}")
                    self.training_history.append({
                        "stage": "cluster",
                        "epoch": int(epoch),
                        "loss": float(loss.item()),
                        "kl_loss": float(kl_loss.item()),
                        "ce_loss": float(ce_loss.item()),
                        "reconstruction_loss": float(re_loss.item()),
                        "delta": float(delta),
                        "converged": True,
                    })
                    converged_this_epoch = True
                    break

            # Logging
            if epoch % 10 == 0:
                print(
                    f"[SDCN] Epoch {epoch} | "
                    f"Loss={loss.item():.4f} | "
                    f"KL={kl_loss.item():.4f} | "
                    f"CE={ce_loss.item():.4f} | "
                    f"RE={re_loss.item():.4f}"
                )
            if not converged_this_epoch:
                self.training_history.append({
                    "stage": "cluster",
                    "epoch": int(epoch),
                    "loss": float(loss.item()),
                    "kl_loss": float(kl_loss.item()),
                    "ce_loss": float(ce_loss.item()),
                    "reconstruction_loss": float(re_loss.item()),
                })

        with torch.no_grad():
            _, q, _, _ = model(X, adj)
            labels = q.argmax(1).cpu().numpy()

        return labels

registry = {
    "kmeans": KMeansClusterer,
    "idec": IDECClusterer,
    "sdcn": SDCNClusterer
}


def get_clusterer(cfg: dict) -> BaseClusterer:
    name = cfg["name"].lower()
    if name not in registry:
        raise ValueError(f"Unknown clusterer: {name}")
    return registry[name](cfg)
