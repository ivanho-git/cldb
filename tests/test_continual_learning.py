import pytest
import numpy as np
import torch
import torch.nn as nn

from cldb.replay_buffer import DatabaseAwarePriorityReplay
from cldb.drift_detection import DriftDetector
from cldb.ewc import EWC

def test_priority_replay_buffer():
    buffer = DatabaseAwarePriorityReplay(capacity=10)
    
    # Add experiences
    for i in range(5):
        # We give the last one a much higher reward to test prioritization
        reward = 1.0 if i < 4 else 10.0
        buffer.add(f"q_{i}", np.random.rand(74), np.random.rand(384), np.random.rand(3), 1, reward)
        
    assert len(buffer) == 5
    
    # Priority of the last element should be higher
    assert buffer.buffer[-1].priority > buffer.buffer[0].priority
    
    # Sample and check dimensions
    feat, emb, ctx, actions, weights, indices = buffer.sample(3)
    assert feat.shape == (3, 74)
    assert emb.shape == (3, 384)
    assert actions.shape == (3,)
    assert weights.shape == (3,)


def test_drift_detection():
    detector = DriftDetector(window_size=10, drift_threshold=0.3)
    
    # Phase 1: All ones
    for _ in range(15):
        drifted = detector.add_experience(np.ones(10))
        assert not drifted
        
    assert detector.get_current_phase() == "phase_1"
    
    # Phase 2: Suddenly change distribution (all negative ones)
    drifted = False
    for _ in range(15):
        if detector.add_experience(np.ones(10) * -1):
            drifted = True
            break
            
    assert drifted
    assert detector.get_current_phase() == "phase_2"


def test_ewc_penalty():
    model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 3))
    ewc = EWC(model, lambda_param=1.0)
    
    # Mock data for registering a phase
    features = torch.randn(20, 4)
    embeddings = torch.randn(20, 3)
    context = torch.randn(20, 3)
    labels = torch.randint(0, 3, (20,))
    
    # Register phase 1
    ewc.register_workload_phase("phase_1", features, embeddings, context, labels)
    
    assert "phase_1" in ewc.optimal_weights
    assert "phase_1" in ewc.fisher_matrices
    
    # Penalty should be zero immediately after registration since weights haven't moved
    initial_penalty = ewc.compute_ewc_loss()
    assert torch.isclose(initial_penalty, torch.tensor(0.0))
    
    # Manually shift weights to simulate forgetting
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.5)
            
    # Penalty should now be > 0
    shifted_penalty = ewc.compute_ewc_loss()
    assert shifted_penalty.item() > 0.0
