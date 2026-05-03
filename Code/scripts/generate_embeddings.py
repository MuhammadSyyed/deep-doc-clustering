import pandas as pd
import random
import numpy as np
from pathlib import Path
import sys
import torch
import re

from glob import glob
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for _pkg in ("punkt", "stopwords", "wordnet", "averaged_perceptron_tagger", "omw-1.4", "punkt_tab"):
    nltk.download(_pkg, quiet=True)


root = Path.cwd()
if not (root / "modules").is_dir():
    REPO_ROOT = root.parent
sys.path.insert(0, str(REPO_ROOT))


seed = 42
cap = 150000
batch_size = 64
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
encoder_name = "sbert"
sbert_model_name = "all-mpnet-base-v2"
_stopwords = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


class BaseEncoder(ABC):
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size

    @abstractmethod
    def fit_transform(
        self, device,
        cleaned_texts: Optional[List[str]] = None,
        tokenized_texts: Optional[List[List[str]]] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def transform(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support transform()")


class SBERTEncoder(BaseEncoder):
    def fit_transform(self, device, cleaned_texts=None, **kwargs):
        model = SentenceTransformer(self.model_name, device=device)
        return model.encode(
            cleaned_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)


def clean_text(text: str, lowercase=True) -> str:
    if lowercase:
        text = text.lower()

    # Remove URLs, email addresses, numbers
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans(
        string.punctuation, " " * len(string.punctuation)))

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, min_token_len=2, remove_stopwords=True, lemmatize=True) -> List[str]:
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _stopwords and len(
            t) >= min_token_len]
    else:
        tokens = [t for t in tokens if len(t) >= min_token_len]
    if lemmatize:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def preprocess(texts: List[str], max_seq_len=None):
    cleaned, tokenized = [], []
    for text in texts:
        cleaned_text = clean_text(text)
        toks = tokenize(cleaned_text)
        if max_seq_len:
            toks = toks[:max_seq_len]
        joined = " ".join(toks)
        cleaned.append(joined)
        tokenized.append(toks)

    return cleaned, tokenized


def main():
    dataset_dirs = [str(dataset)
                    for dataset in list((REPO_ROOT / "datasets").glob("*/"))]
    for dataset in dataset_dirs:
        dataset_name = dataset.split("/")[-1]
        df = pd.read_csv(f"{dataset}/data.csv")
        df = df.dropna(subset=['text', 'label'])
        df['text'] = df['text'].astype(str).str.strip()
        df = df[df['text'].str.len() > 0]
        if cap is not None and cap < len(df):
            df = df.sample(n=cap, random_state=42)

        embeddings_path = Path(
            f"../embeddings/emb_{dataset_name}_{encoder_name}.npy")
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)

        texts = df['text'].tolist()

        if not embeddings_path.exists():
            print("Generating embeddings")
            cleaned, tokenized = preprocess(texts)
            encoder = SBERTEncoder(
                model_name=sbert_model_name, batch_size=batch_size)
            embeddings = encoder.fit_transform(
                device, cleaned, tokenized_texts=tokenized)

            np.save(embeddings_path, embeddings)
        else:
            print(f"Embeddings already exist for {dataset_name}")

        print(f"Dataset: {dataset_name}, Embeddings shape: {embeddings.shape}")

if __name__ == "__main__":
    main()
