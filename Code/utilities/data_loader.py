import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def load_embeddings(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    texts = data["texts"]
    assert len(embeddings) == len(labels) == len(
        texts), "Mismatch in data lengths"
    return embeddings, labels, texts


class EmbeddingDataset(Dataset):
    def __init__(self, npz_path):
        self.embeddings, self.labels, self.texts = load_embeddings(npz_path)
        self.embeddings = torch.from_numpy(self.embeddings)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "label": self.labels[idx],
            "text": self.texts[idx]
        }


def get_dataloader(npz_path, batch_size=64, shuffle=True, num_workers=2, device="mps"):
    dataset = EmbeddingDataset(npz_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )


if __name__ == "__main__":
    dataloader = get_dataloader("../embeddings/emb_bbcnews_sbert.npz")
    for batch in dataloader:
        print(batch["embedding"].shape,
              batch["label"].shape, len(batch["text"]))
        break
