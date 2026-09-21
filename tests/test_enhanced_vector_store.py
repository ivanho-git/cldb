"""
Tests for enhanced vector store.
"""
import pytest
import numpy as np
from cldb.enhanced_vector_store import EnhancedVectorStore, VectorStoreConfig


def test_vector_store_init_in_memory():
    # When Qdrant is unavailable, should fall back to memory
    store = EnhancedVectorStore(collection_name="test_fallback")
    assert store is not None
    assert store.collection_name == "test_fallback"


def test_vector_store_store_experience():
    store = EnhancedVectorStore(collection_name="test_store")
    
    embedding = np.random.rand(384).astype(np.float32)
    
    result = store.store_experience(
        query_id=1,
        embedding=embedding,
        action="CREATE_INDEX",
        outcome=15.5
    )
    
    assert result == True


def test_vector_store_find_similar():
    store = EnhancedVectorStore(collection_name="test_find_similar")
    
    embedding = np.random.rand(384).astype(np.float32)
    store.store_experience(query_id=1, embedding=embedding, action="CREATE_INDEX", outcome=10.0)
    
    results = store.find_similar_queries(embedding, k=5)
    
    assert isinstance(results, list)


def test_vector_store_batch():
    store = EnhancedVectorStore(collection_name="test_batch")
    
    experiences = [
        {"query_id": i, "embedding": np.random.rand(384).astype(np.float32), "action": "CREATE_INDEX", "outcome": 10.0}
        for i in range(5)
    ]
    
    count = store.store_batch(experiences)
    
    assert count == 5


def test_vector_store_health_check():
    store = EnhancedVectorStore(collection_name="test_health")
    
    health = store.health_check()
    
    assert "status" in health
    assert "connected" in health
    assert "collection" in health


def test_vector_store_clear():
    store = EnhancedVectorStore(collection_name="test_clear")
    
    store.store_experience(query_id=1, embedding=np.random.rand(384), action="CREATE", outcome=1.0)
    
    result = store.clear_collection()
    
    assert result == True


def test_vector_store_config():
    config = VectorStoreConfig(host="custom-host", port=9999, collection_name="custom")
    
    assert config.host == "custom-host"
    assert config.port == 9999
    assert config.collection_name == "custom"