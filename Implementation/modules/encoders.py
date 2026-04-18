import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class BaseEncoder(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def fit_transform(
        self,device,
        cleaned_texts: Optional[List[str]] = None,
        tokenized_texts: Optional[List[List[str]]] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def transform(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support transform()")


class TFIDFEncoder(BaseEncoder):

    def fit_transform(self,device, cleaned_texts=None, **kwargs):
        c = self.cfg.get("tfidf", {})

        vectorizer = TfidfVectorizer(
            max_features=c.get("max_features"),
            ngram_range=tuple(c.get("ngram_range")),
            sublinear_tf=c.get("sublinear_tf"),
        )

        X = vectorizer.fit_transform(cleaned_texts)

        n_comp = min(300, X.shape[1] - 1, X.shape[0] - 1)
        if n_comp > 0:
            X = TruncatedSVD(n_components=n_comp,
                             random_state=42).fit_transform(X)

        return normalize(X.astype(np.float32))


class SBERTEncoder(BaseEncoder):

    def fit_transform(self, device, cleaned_texts=None, **kwargs):
       

        c = self.cfg.get("sbert", {})
        model = SentenceTransformer(
            c.get("model_name"), device=device)

        return model.encode(
            cleaned_texts,
            batch_size=c.get("batch_size"),
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)


registry = {
    "tfidf": TFIDFEncoder, "sbert": SBERTEncoder
}


def get_encoder(cfg: dict) -> BaseEncoder:
    name = cfg["name"].lower()
    if name not in registry:
        raise ValueError(f"Unknown encoder: {name}")
    return registry[name](cfg)
