import pytest
import torch
from cldb.cl_engine import CLEngine, EMBEDDING_DIM, CONTEXT_DIM


def test_cl_engine_init():
    engine = CLEngine(feature_dim=74, ewc_lambda=0.4, replay_mem_size=100)
    assert engine.model is not None
    assert engine.ewc_plugin is not None
    assert engine.replay_plugin is not None


def test_cl_engine_train_and_predict():
    engine = CLEngine(feature_dim=74)

    batch_size = 2
    x_feat = torch.randn(batch_size, 74)
    x_emb = torch.randn(batch_size, EMBEDDING_DIM)
    x_ctx = torch.randn(batch_size, CONTEXT_DIM)
    y = torch.tensor([1, 0], dtype=torch.long)

    engine.train_on_batch(x_feat, x_emb, x_ctx, y)
    actions, confidences = engine.predict(x_feat, x_emb, x_ctx)

    assert len(actions) == batch_size
    assert len(confidences) == batch_size
    assert all(0 <= a <= 2 for a in actions)
    assert all(0.0 <= c <= 1.0 for c in confidences)


def test_cl_engine_predict_from_features_only():
    engine = CLEngine(feature_dim=74)
    x_feat = torch.randn(2, 74)

    actions, confidences = engine.predict_from_features_only(x_feat)

    assert len(actions) == 2
    assert all(0 <= a <= 2 for a in actions)
    assert all(0.0 <= c <= 1.0 for c in confidences)