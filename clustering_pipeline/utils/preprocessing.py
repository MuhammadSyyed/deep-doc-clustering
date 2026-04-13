import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for _pkg in ("punkt", "stopwords", "wordnet", "averaged_perceptron_tagger",
             "omw-1.4", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

_STOP = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


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
        tokens = [t for t in tokens if t not in _STOP and len(
            t) >= min_token_len]
    else:
        tokens = [t for t in tokens if len(t) >= min_token_len]
    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
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
