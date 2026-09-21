"""
Tests for enhanced CL engine.
"""
import pytest
import torch
from cldb.enhanced_cl_engine import (
    CLEngine, EnhancedCLEngine, ImprovedPolicyNetwork, ResidualPolicyNetwork,
    AttentionPolicyNetwork, EMBEDDING_DIM, CONTEXT_DIM
)


def test_improved_policy_network_forward():
    model = ImprovedPolicyNetwork(input_dim=74 + EMBEDDING_DIM + CONTEXT_DIM, hidden_dims=[128, 64])
    x = torch.randn(4, 74 + EMBEDDING_DIM + CONTEXT_DIM)
    output = model(x)
    assert output.shape == (4, 3)


def test_residual_policy_network_forward():
    model = ResidualPolicyNetwork(input_dim=74 + EMBEDDING_DIM + CONTEXT_DIM, hidden_dim=128)
    x = torch.randn(4, 74 + EMBEDDING_DIM + CONTEXT_DIM)
    output = model(x)
    assert output.shape == (4, 3)


def test_attention_policy_network_forward():
    model = AttentionPolicyNetwork(input_dim=74 + EMBEDDING_DIM + CONTEXT_DIM, hidden_dim=128, num_heads=4)
    x = torch.randn(4, 74 + EMBEDDING_DIM + CONTEXT_DIM)
    output = model(x)
    assert output.shape == (4, 3)


def test_cl_engine_init():
    engine = CLEngine(feature_dim=74)
    assert engine.feature_dim == 74
    assert engine.model is not None


def test_cl_engine_train_on_batch():
    engine = CLEngine(feature_dim=74)
    
    batch_size = 4
    x_feat = torch.randn(batch_size, 74)
    x_emb = torch.randn(batch_size, EMBEDDING_DIM)
    x_ctx = torch.randn(batch_size, CONTEXT_DIM)
    y = torch.tensor([1, 0, 2, 1], dtype=torch.long)
    
    loss = engine.train_on_batch(x_feat, x_emb, x_ctx, y)
    assert loss >= 0


def test_cl_engine_predict():
    engine = CLEngine(feature_dim=74)
    
    x_feat = torch.randn(2, 74)
    x_emb = torch.randn(2, EMBEDDING_DIM)
    x_ctx = torch.randn(2, CONTEXT_DIM)
    
    actions, confidences = engine.predict(x_feat, x_emb, x_ctx, use_cache=False)
    
    assert len(actions) == 2
    assert all(0 <= a <= 2 for a in actions)
    assert all(0.0 <= c <= 1.0 for c in confidences)


def test_cl_engine_cache():
    engine = CLEngine(feature_dim=74, cache_size=100)
    
    x_feat = torch.randn(2, 74)
    x_emb = torch.randn(2, EMBEDDING_DIM)
    x_ctx = torch.randn(2, CONTEXT_DIM)
    
    # First call
    actions1, _ = engine.predict(x_feat, x_emb, x_ctx, use_cache=True)
    
    # Second call should hit cache
    actions2, _ = engine.predict(x_feat, x_emb, x_ctx, use_cache=True)
    
    assert engine._cache_hits >= 1


def test_cl_engine_save_load():
    engine1 = CLEngine(feature_dim=74, architecture="standard")
    
    # Train a bit
    x_feat = torch.randn(4, 74)
    x_emb = torch.randn(4, EMBEDDING_DIM)
    x_ctx = torch.randn(4, CONTEXT_DIM)
    y = torch.tensor([1, 0, 2, 1], dtype=torch.long)
    engine1.train_on_batch(x_feat, x_emb, x_ctx, y)
    
    # Save
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
    
    try:
        engine1.save(temp_path)
        
        # Load into new engine
        engine2 = CLEngine.load(temp_path, feature_dim=74)
        
        assert engine2.feature_dim == engine1.feature_dim
        assert engine2.embedding_dim == engine1.embedding_dim
    finally:
        os.unlink(temp_path)


def test_cl_engine_architecture_variants():
    for arch in ["standard", "residual", "attention"]:
        engine = CLEngine(feature_dim=74, architecture=arch)
        assert engine.model is not None
        assert engine.architecture == arch


def test_cl_engine_clear_cache():
    engine = CLEngine(feature_dim=74, cache_size=10)
    
    x_feat = torch.randn(2, 74)
    x_emb = torch.randn(2, EMBEDDING_DIM)
    x_ctx = torch.randn(2, CONTEXT_DIM)
    
    engine.predict(x_feat, x_emb, x_ctx, use_cache=True)
    assert len(engine._cache) > 0
    
    engine.clear_cache()
    assert len(engine._cache) == 0
    assert engine._cache_hits == 0