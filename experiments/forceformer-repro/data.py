"""Download and prepare the NLTK movie_reviews corpus."""
import os
import random
import re
from collections import Counter
from pathlib import Path

import nltk
from nltk.corpus import movie_reviews

TOKEN_RE = re.compile(r"[a-z']+")
DEFAULT_NLTK_DIR = Path.home() / ".cache" / "forceformer-repro" / "nltk_data"

def ensure_corpus():
    data_dir = Path(os.environ.get("NLTK_DATA", DEFAULT_NLTK_DIR)).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(data_dir))
    try:
        movie_reviews.fileids()
    except LookupError:
        if not nltk.download("movie_reviews", download_dir=str(data_dir), quiet=True):
            raise RuntimeError(
                "Could not download the NLTK movie_reviews corpus. "
                "Check network access or set NLTK_DATA to a directory containing it."
            )
        nltk.data.path.insert(0, str(data_dir))
        try:
            movie_reviews.fileids()
        except LookupError as exc:
            raise RuntimeError(
                f"The movie_reviews corpus was not found under {data_dir}."
            ) from exc

def tokenize(text):
    return TOKEN_RE.findall(text.lower())

def build_vocab(train_texts, vocab_size=10000):
    counter = Counter()
    for text in train_texts:
        counter.update(tokenize(text))
    words = [word for word, _ in counter.most_common(vocab_size - 2)]
    itos = ["<pad>", "<unk>", *words]
    stoi = {word: index for index, word in enumerate(itos)}
    return stoi, itos

def encode(text, stoi, max_len):
    ids = [stoi.get(token, 1) for token in tokenize(text)[:max_len]]
    return ids + [0] * (max_len - len(ids))

def get_splits(max_len=256, vocab_size=10000, val_frac=0.1, seed=42):
    """Return the original deterministic train/validation split."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1")
    ensure_corpus()
    docs = []
    for category, label in (("pos", 1), ("neg", 0)):
        fileids = sorted(movie_reviews.fileids(categories=[category]))
        docs.extend((movie_reviews.raw(fileid), label) for fileid in fileids)
    random.Random(seed).shuffle(docs)
    n_val = int(len(docs) * val_frac)
    val_docs, train_docs = docs[:n_val], docs[n_val:]
    stoi, itos = build_vocab([text for text, _ in train_docs], vocab_size)

    def encode_split(docs):
        features = [encode(text, stoi, max_len) for text, _ in docs]
        labels = [label for _, label in docs]
        return features, labels

    return encode_split(train_docs), encode_split(val_docs), stoi, itos

if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), stoi, _ = get_splits()
    print(f"train={len(x_train)} val={len(x_val)} vocab={len(stoi)}")
