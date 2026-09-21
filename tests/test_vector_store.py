import pytest
import numpy as np
from cldb.vector_store import VectorStore


def test_vector_store_init():
    vs = VectorStore(collection_name="test_cldb_queries")
    assert vs.collection_name == "test_cldb_queries"


def test_vector_store_ensure_collection():
    vs = VectorStore(collection_name="test_ensure")
    client = vs.client
    coll = client.get_collection("test_ensure")
    assert coll.vectors_count == 0


def test_vector_store_store_and_find():
    vs = VectorStore(collection_name="test_store_find")
    embedding = np.random.rand(384).astype(np.float32)

    vs.store_experience(
        query_id=1,
        embedding=embedding,
        action="CREATE_INDEX",
        outcome=15.5,
    )

    results = vs.find_similar_queries(embedding, k=5)
    assert len(results) >= 1
    assert results[0]["action"] == "CREATE_INDEX"
    assert results[0]["outcome"] == 15.5


def test_vector_store_multiple_experiences():
    vs = VectorStore(collection_name="test_multi")
    emb1 = np.random.rand(384).astype(np.float32)
    emb2 = np.random.rand(384).astype(np.float32)

    vs.store_experience(1, emb1, "CREATE_INDEX", 10.0)
    vs.store_experience(2, emb2, "DROP_INDEX", -5.0)

    results = vs.find_similar_queries(emb1, k=2)
    assert len(results) >= 1
    assert results[0]["query_id"] == 1