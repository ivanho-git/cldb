"""Semantic embeddings for SQL queries.

The primary backend is a sentence-transformers model. When the model is not
available (for example, offline environments or a missing model cache) the
embedder degrades gracefully to a deterministic hashing-based encoder that
preserves the 384-dimensional, L2-normalized contract the rest of the system
relies on.
"""
import hashlib
import os
import re

import numpy as np


EMBEDDING_DIM = 384


class _HashingEmbedder:
    """Deterministic, dependency-free fallback embedder.

    Tokenizes SQL into alphanumeric tokens and accumulates each into a signed
    hashing bucket. Near-identical queries therefore produce near-identical
    vectors, while the output remains a unit-norm 384-dim vector.
    """

    _token_pattern = re.compile(r"[a-z0-9_]+")

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim

    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in self._token_pattern.findall(text.lower()):
            digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = digest % self.dim
            sign = 1.0 if (digest >> 8) & 1 else -1.0
            vec[idx] += sign
        if normalize_embeddings:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec


class QueryEmbedder:
    """
    Generates semantic embeddings for SQL queries using sentence-transformers.
    Falls back to a deterministic hashing encoder when the model is unavailable.
    A SQL-aware encoder like CodeBERT could be swapped in here later.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.backend = "hashing"
        self.model = _HashingEmbedder(dim)
        try:
            from sentence_transformers import SentenceTransformer

            # Only consult the local cache unless the caller explicitly allows
            # network access. This keeps offline/test runs from hanging.
            local_only = os.environ.get("CLDB_EMBEDDING_ALLOW_DOWNLOAD") != "1"
            self.model = SentenceTransformer(model_name, local_files_only=local_only)
            self.backend = "sentence-transformers"
        except Exception:
            # Model unavailable: keep the deterministic fallback.
            self.model = _HashingEmbedder(dim)

    def embed(self, query: str) -> np.ndarray:
        # Normalize the query (lowercase, strip extra whitespace) to improve semantic matching
        normalized = " ".join(query.split()).lower()
        if self.backend == "sentence-transformers":
            vec = self.model.encode(normalized, normalize_embeddings=True)
        else:
            vec = self.model.encode(normalized, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)
