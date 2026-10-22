"""Dense retriever over PubMed abstracts. Each abstract is one chunk: abstracts are short (median ~200 words)
and self-contained, so splitting them would only cut sentences away from their context."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this instruction in front of queries (not in front of passages)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def device():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class DenseRetriever:
    def __init__(self, passages: pd.DataFrame):
        from sentence_transformers import SentenceTransformer
        self.passages = passages
        self.ids = passages.index.to_numpy()
        self.model = SentenceTransformer(EMBED_MODEL, device=device())
        cache = CACHE / f"bge_small_{len(passages)}.npy"
        if cache.exists():
            self.vectors = np.load(cache)
        else:
            CACHE.mkdir(parents=True, exist_ok=True)
            self.vectors = self.model.encode(passages["passage"].tolist(), batch_size=64, normalize_embeddings=True,
                                             show_progress_bar=True).astype(np.float32)
            np.save(cache, self.vectors)

    def search(self, query: str, k: int = 3) -> list:
        q = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [{"id": int(self.ids[i]), "score": float(scores[i]), "text": self.passages["passage"].iloc[i]}
                for i in top]

    def get(self, ids: list) -> list:
        return [{"id": int(i), "score": None, "text": self.passages.loc[i, "passage"]} for i in ids]
