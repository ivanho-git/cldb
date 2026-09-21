import pytest
from cldb.embeddings import QueryEmbedder
import numpy as np


def test_embedder_init():
    embedder = QueryEmbedder()
    assert embedder.model is not None


def test_embedder_produces_fixed_dim():
    embedder = QueryEmbedder()
    vec = embedder.embed("SELECT * FROM orders WHERE customer_id = 42;")
    assert vec.shape[0] == 384
    assert isinstance(vec, np.ndarray)


def test_embedder_deterministic():
    embedder = QueryEmbedder()
    q1 = "SELECT * FROM orders WHERE customer_id = 42;"
    q2 = "SELECT * FROM orders WHERE customer_id = 42;"
    v1 = embedder.embed(q1)
    v2 = embedder.embed(q2)
    assert np.allclose(v1, v2)


def test_embedder_normalizes():
    embedder = QueryEmbedder()
    vec = embedder.embed("SELECT * FROM orders;")
    norm = np.linalg.norm(vec)
    assert 0.99 < norm < 1.01


def test_embedder_similar_queries_close():
    embedder = QueryEmbedder()
    q1 = "SELECT * FROM orders WHERE customer_id = 42;"
    q2 = "SELECT * FROM orders WHERE customer_id = 99;"
    v1 = embedder.embed(q1)
    v2 = embedder.embed(q2)
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    assert 0.5 < similarity < 1.0