"""Tests for Counterfactual What-If Learning (CWIL)."""
import os

import numpy as np
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from cldb.counterfactual import SimulatorOracle, build_counterfactual_set
from cldb.rl_agent import ACTION_CREATE, ACTION_KEEP, DQNAgent
from cldb.sim_env import IndexAdvisorEnv


def make_env(**kwargs):
    defaults = dict(seed=0, budget=2, horizon=20)
    defaults.update(kwargs)
    return IndexAdvisorEnv(**defaults)


def make_agent(**kwargs):
    defaults = dict(feature_dim=74, embedding_dim=384, context_dim=3,
                    index_state_dim=3, hidden_dims=(64, 32), seed=0,
                    ewc_lambda=0.0)
    defaults.update(kwargs)
    return DQNAgent(**defaults)


def _snapshot_equal(a, b):
    return (a["task_id"] == b["task_id"] and a["step_idx"] == b["step_idx"]
            and a["active"] == b["active"]
            and a["next_index_id"] == b["next_index_id"]
            and a["current_sql"] == b["current_sql"]
            and a["rng_state"] == b["rng_state"])


def test_what_if_does_not_mutate_state_or_rng():
    env = make_env()
    env.reset(task_id=0)
    env._current_sql = "SELECT * FROM orders WHERE customer_id = 42;"
    before = env.snapshot()

    branches = env.what_if()

    after = env.snapshot()
    assert _snapshot_equal(before, after)
    assert set(branches) == set(np.flatnonzero(env.valid_actions()))


def test_what_if_matches_real_step_and_restores_rng():
    env = make_env()
    env.reset(task_id=0)
    env._current_sql = "SELECT * FROM orders WHERE customer_id = 42;"

    branches = env.what_if()
    chosen = ACTION_CREATE
    expected_obs, expected_reward, _, _, _ = branches[chosen]

    obs, reward, _, _ = env.step(chosen)

    assert reward == pytest.approx(expected_reward)
    for actual, expected in zip(obs, expected_obs):
        assert np.allclose(actual, expected)


def test_counterfactual_set_shapes():
    env = make_env()
    env.reset(task_id=0)
    env._current_sql = "SELECT * FROM orders WHERE customer_id = 42;"

    cf = build_counterfactual_set(env.what_if(), env.obs_vector)

    assert len(cf) == 2  # KEEP and CREATE are valid on a fresh, uncovered predicate
    assert cf.actions.tolist() == [ACTION_KEEP, ACTION_CREATE]
    assert cf.rewards.shape == (2,)
    assert cf.next_states.shape == (2, 464)
    assert cf.dones.shape == (2,)
    assert cf.next_masks.shape == (2, 3)


def test_oracle_wrapper():
    env = make_env()
    env.reset(task_id=0)
    env._current_sql = "SELECT * FROM orders WHERE customer_id = 42;"
    oracle = SimulatorOracle(env)
    cf = oracle.counterfactuals(env.obs_vector)
    assert len(cf) == len(oracle.what_if())


def test_counterfactual_training_reduces_loss():
    env = make_env()
    env.reset(task_id=0)
    env._current_sql = "SELECT * FROM orders WHERE customer_id = 42;"
    state = env.obs_vector(env._build_obs(env._current_sql))
    cf = build_counterfactual_set(env.what_if(), env.obs_vector)

    agent = make_agent(learning_rate=1e-3)
    first = agent.train_counterfactual(state, cf)
    assert first is not None
    assert first["cf_actions"] == 2

    losses = [agent.train_counterfactual(state, cf)["cf_loss"] for _ in range(40)]
    assert losses[-1] < first["cf_loss"]
    assert agent.cf_updates == 41
    assert agent.updates == 41


def test_cwil_learns_to_keep_on_conflict_phase():
    """On an inverted-selectivity phase, what-if data must teach KEEP.

    Phase 2 contains the *same* query as phase 1 but marks the predicate
    non-selective, so creating an index is strictly worse. A replay-only
    agent has to try CREATE to discover this; CWIL sees the counterfactual
    cost immediately.
    """
    env = make_env(horizon=50)
    env.reset(task_id=2)
    env._current_sql = "SELECT * FROM orders WHERE status = 'PENDING';"
    state = env.obs_vector(env._build_obs(env._current_sql))
    mask = env.valid_actions()
    assert mask[ACTION_CREATE]

    cf = build_counterfactual_set(env.what_if(), env.obs_vector)
    create_reward = cf.rewards[cf.actions.tolist().index(ACTION_CREATE)]
    keep_reward = cf.rewards[cf.actions.tolist().index(ACTION_KEEP)]
    assert create_reward < keep_reward

    agent = make_agent(learning_rate=1e-3, epsilon_start=0.0, epsilon_end=0.0)
    for _ in range(200):
        agent.train_counterfactual(state, cf)

    action, _ = agent.predict(state, valid_actions=mask)
    assert action == ACTION_KEEP
