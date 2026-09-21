"""
Tests for training module.
"""
import pytest
import torch
import numpy as np
from cldb.training import Normalizer, EarlyStopping, ModelCheckpointManager, Trainer


def test_normalizer_standard():
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    normalizer = Normalizer(method="standard")
    
    normalized = normalizer.fit_transform(data)
    
    assert normalized.shape == data.shape
    assert abs(normalized.mean(axis=0).max()) < 0.1


def test_normalizer_minmax():
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    normalizer = Normalizer(method="minmax")
    
    normalized = normalizer.fit_transform(data)
    
    assert normalized.min() >= 0
    assert normalized.max() <= 1


def test_normalizer_save_load(tmp_path):
    normalizer = Normalizer(method="standard")
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    normalizer.fit(data)
    
    save_path = tmp_path / "normalizer.json"
    normalizer.save(str(save_path))
    
    loaded = Normalizer.load(str(save_path))
    
    assert loaded._fitted == normalizer._fitted
    assert loaded.method == normalizer.method


def test_early_stopping_min_mode():
    stopper = EarlyStopping(patience=3, mode="min")
    
    assert stopper(10.0) == False
    assert stopper(9.0) == False
    assert stopper(9.5) == True  # 9.5 > 9 - min_delta


def test_early_stopping_max_mode():
    stopper = EarlyStopping(patience=3, mode="max")
    
    assert stopper(0.5) == False
    assert stopper(0.8) == False
    assert stopper(0.75) == True  # 0.75 < 0.8 + min_delta


def test_checkpoints_manager_save_load(tmp_path):
    import torch.nn as nn
    
    manager = ModelCheckpointManager(str(tmp_path))
    
    model = nn.Linear(10, 3)
    optimizer = torch.optim.Adam(model.parameters())
    
    path = manager.save(model, optimizer, epoch=1, task_id=0, metrics={"val_loss": 0.5}, config={})
    
    assert path is not None
    assert (tmp_path / path.split("/")[-1]).exists()


def test_trainer_fit():
    from cldb.enhanced_cl_engine import ImprovedPolicyNetwork, EMBEDDING_DIM, CONTEXT_DIM
    
    input_dim = 74 + EMBEDDING_DIM + CONTEXT_DIM
    model = ImprovedPolicyNetwork(input_dim=input_dim, hidden_dims=[64, 32])
    
    trainer = Trainer(model)
    
    features = torch.randn(20, 74)
    embeddings = torch.randn(20, EMBEDDING_DIM)
    context = torch.randn(20, CONTEXT_DIM)
    labels = torch.randint(0, 3, (20,))
    
    history = trainer.fit(features, embeddings, context, labels, val_split=0.2)
    
    assert "epochs" in history
    assert "final_train_loss" in history
    assert history["epochs"] > 0


def test_trainer_with_dataloader():
    from cldb.enhanced_cl_engine import ImprovedPolicyNetwork, EMBEDDING_DIM, CONTEXT_DIM
    
    input_dim = 74 + EMBEDDING_DIM + CONTEXT_DIM
    model = ImprovedPolicyNetwork(input_dim=input_dim, hidden_dims=[64, 32])
    
    trainer = Trainer(model)
    
    features = torch.randn(20, 74)
    embeddings = torch.randn(20, EMBEDDING_DIM)
    context = torch.randn(20, CONTEXT_DIM)
    labels = torch.randint(0, 3, (20,))
    
    history = trainer.train_with_dataloader(features, embeddings, context, labels, batch_size=8, epochs=2)
    
    assert len(history["loss"]) == 2
    assert len(history["accuracy"]) == 2