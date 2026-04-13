import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from gensim.models import Word2Vec, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from nltk.corpus import wordnet as wn


class BaseEncoder(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def fit_transform(
        self,
        cleaned_texts: Optional[List[str]] = None,
        tokenized_texts: Optional[List[List[str]]] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def transform(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support transform()")


class TFIDFEncoder(BaseEncoder):

    def fit_transform(self, cleaned_texts=None, **kwargs):
        c = self.cfg.get("tfidf", {})

        vectorizer = TfidfVectorizer(
            max_features=c.get("max_features", 10000),
            ngram_range=tuple(c.get("ngram_range", [1, 2])),
            sublinear_tf=c.get("sublinear_tf", True),
        )

        X = vectorizer.fit_transform(cleaned_texts)

        n_comp = min(300, X.shape[1] - 1, X.shape[0] - 1)
        if n_comp > 0:
            X = TruncatedSVD(n_components=n_comp,
                             random_state=42).fit_transform(X)

        return normalize(X.astype(np.float32))


class SBERTEncoder(BaseEncoder):

    def fit_transform(self, cleaned_texts=None, **kwargs):
        from sentence_transformers import SentenceTransformer

        c = self.cfg.get("sbert", {})
        model = SentenceTransformer(c.get("model_name", "all-MiniLM-L6-v2"),device='mps')

        return model.encode(
            cleaned_texts,
            batch_size=c.get("batch_size", 64),
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)


class Doc2VecEncoder(BaseEncoder):

    def fit_transform(self, cleaned_texts=None, tokenized_texts=None):
        c = self.cfg.get("doc2vec", {})

        tagged = [
            TaggedDocument(words=toks, tags=[str(i)])
            for i, toks in enumerate(tokenized_texts)
        ]

        model = Doc2Vec(
            vector_size=c.get("vector_size", 300),
            window=c.get("window", 5),
            min_count=c.get("min_count", 2),
            dm=c.get("dm", 1),
            workers=1,
            seed=42,
        )

        model.build_vocab(tagged)

        model.train(
            tagged,
            total_examples=model.corpus_count,
            epochs=c.get("epochs", 40),
        )

        X = np.array([model.dv[str(i)] for i in range(len(tagged))])
        return normalize(X.astype(np.float32))


_REGISTRY = {
    "tfidf": TFIDFEncoder,
    "doc2vec": Doc2VecEncoder,
    "sbert": SBERTEncoder
}


def get_encoder(cfg: dict) -> BaseEncoder:
    name = cfg["name"].lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown encoder: {name}")
    return _REGISTRY[name](cfg)
