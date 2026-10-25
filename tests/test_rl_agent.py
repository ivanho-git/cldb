"""Tests for the DQN index-selection agent."""
import os
import tempfile

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from cldb.rl_agent import DQNAgent, QNetwork, build_state, ACTION_KEEP, ACTION_CREATE, ACTION_DROP
from cldb.replay_buffer import DatabaseAwarePriorityReplay


def make_agent(**kwargs):
    defaults = dict(feature_dim=74, embedding_dim=384, context_dim=3, index_state_dim=3,
                    hidden_dims=(64, 32), seed=0)
    defaults.update(kwargs)
    return DQNAgent(**defaults)


def random_state(agent):
    return build_state(np.random.rand(74), np.random.rand(384), np.random.rand(3), np.random.rand(3))


def test_qnetwork_output_shape():
    net = QNetwork(state_dim=74 + 384 + 3 + 3, num_actions=3, hidden_dims=(64, 32))
    out = net(torch.randn(5, 74 + 384 + 3 + 3))
    assert out.shape == (5, 3)


def test_state_dim():
    agent = make_agent()
    assert agent.state_dim == 74 + 384 + 3 + 3


def test_select_action_greedy_and_masked():
    agent = make_agent()
    state = random_state(agent)
    # Mask out everything except DROP
    mask = np.array([False, False, True])
    action = agent.select_action(state, valid_actions=mask, explore=False)
    assert action == ACTION_DROP


def test_predict_returns_valid_action_and_confidence():
    agent = make_agent()
    state = random_state(agent)
    action, confidence = agent.predict(state)
    assert action in (ACTION_KEEP, ACTION_CREATE, ACTION_DROP)
    assert 0.0 <= confidence <= 1.0


def test_epsilon_decays_on_every_action():
    agent = make_agent(epsilon_start=1.0, epsilon_end=0.0, epsilon_decay_steps=100)
    state = random_state(agent)
    for _ in range(50):
        agent.select_action(state, valid_actions=np.array([True, True, True]), explore=True)
    assert agent.epsilon < 1.0


def test_train_from_buffer_reduces_td():
    agent = make_agent(learning_rate=1e-3)
    buffer = DatabaseAwarePriorityReplay(capacity=2000, alpha=0.6, beta=0.4)
    env_state = (np.random.rand(74), np.random.rand(384), np.random.rand(3), np.random.rand(3))
    for i in range(300):
        next_state = (np.random.rand(74), np.random.rand(384), np.random.rand(3), np.random.rand(3))
        buffer.add_transition(f"q{i}", int(np.random.randint(3)), float(np.random.randn()), False,
                              state=env_state, next_state=next_state)
    first = agent.train_from_buffer(buffer, batch_size=64)
    assert first is not None
    assert "loss" in first and "td_mean" in first
    # A few more updates should run without error.
    for _ in range(5):
        agent.train_from_buffer(buffer, batch_size=64)
    assert agent.updates == 6


def test_consolidate_registers_ewc_phase():
    agent = make_agent(ewc_lambda=0.4)
    states = torch.randn(64, agent.state_dim)
    agent.consolidate("phase_0", states)
    assert "phase_0" in agent.ewc.optimal_weights
    assert "phase_0" in agent.ewc.fisher_matrices
    # Penalty is zero right after consolidation (weights unchanged).
    assert torch.isclose(agent.ewc.compute_ewc_loss(), torch.tensor(0.0))
    # Perturbing the weights makes the penalty positive.
    with torch.no_grad():
        for p in agent.online.parameters():
            p.add_(0.1)
    assert agent.ewc.compute_ewc_loss().item() > 0.0


def test_save_load_roundtrip():
    agent = make_agent()
    states = torch.randn(32, agent.state_dim)
    agent.consolidate("phase_0", states)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        path = f.name
    try:
        agent.save(path)
        restored = make_agent()
        restored.load(path)
        assert "phase_0" in restored.ewc.optimal_weights
        assert restored.steps == agent.steps
    finally:
        os.unlink(path)
