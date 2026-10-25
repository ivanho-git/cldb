"""Tests for the index-advisor simulator environment."""
import os

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from cldb.sim_env import IndexAdvisorEnv, PHASE_TEMPLATES, PHASE_SELECTIVITY
from cldb.rl_agent import ACTION_KEEP, ACTION_CREATE, ACTION_DROP


def make_env(**kwargs):
    defaults = dict(seed=0, budget=2, horizon=20)
    defaults.update(kwargs)
    return IndexAdvisorEnv(**defaults)


def test_reset_observation_shapes():
    env = make_env()
    obs = env.reset(task_id=0)
    f, e, c, s = obs
    assert f.shape[0] == env.feature_dim == 74
    assert e.shape[0] == 384
    assert c.shape[0] == 3
    assert s.shape[0] == 3


def test_num_tasks_matches_templates():
    env = make_env()
    assert env.num_tasks == len(PHASE_TEMPLATES)


def test_keep_baseline_is_worst():
    env = make_env()
    env.set_task(0)
    score = env.evaluate(lambda o, m: ACTION_KEEP)
    assert score <= -0.9  # never indexed -> expensive latency every step


def test_create_reduces_latency_on_selective_predicate():
    env = make_env()
    # Force a query whose predicate is high-selectivity and indexable.
    env.reset(task_id=0)
    env._current_sql = "SELECT * FROM orders WHERE customer_id = 42;"
    mask = env.valid_actions()
    assert mask[ACTION_CREATE]
    _, reward_create, _, info = env.step(ACTION_CREATE)
    assert info["n_indexes"] == 1
    assert info["latency"] == env.latency_cheap

    # Same predicate again -> still covered and cheap.
    env._current_sql = "SELECT * FROM orders WHERE customer_id = 7;"
    _, _, _, info2 = env.step(ACTION_KEEP)
    assert info2["latency"] == env.latency_cheap


def test_low_selectivity_column_not_helped():
    env = make_env()
    env.reset(task_id=2)  # phase 2 marks status/category as low-selectivity
    env._current_sql = "SELECT * FROM orders WHERE status = 'PENDING';"
    assert PHASE_SELECTIVITY[2]["status"] == "low"
    env.step(ACTION_CREATE)  # index an unhelpful column
    env._current_sql = "SELECT * FROM orders WHERE status = 'PENDING';"
    _, _, _, info = env.step(ACTION_KEEP)
    assert info["latency"] == env.latency_expensive


def test_budget_is_respected():
    env = make_env(budget=2)
    env.reset(task_id=0)
    queries = [
        "SELECT * FROM orders WHERE customer_id = 42;",
        "SELECT * FROM order_items WHERE product_id = 7;",
        "SELECT * FROM customers WHERE customer_id = 15;",
    ]
    for q in queries:
        env._current_sql = q
        if env.valid_actions()[ACTION_CREATE]:
            env.step(ACTION_CREATE)
    assert len(env.active) <= 2


def test_valid_actions_no_predicate_disallows_create():
    env = make_env()
    env.reset(task_id=2)
    env._current_sql = "SELECT COUNT(*) FROM orders;"
    mask = env.valid_actions()
    assert mask[ACTION_KEEP]
    assert not mask[ACTION_CREATE]
    assert not mask[ACTION_DROP]


def test_drop_requires_existing_index():
    env = make_env()
    env.reset(task_id=0)
    env._current_sql = "SELECT * FROM orders WHERE customer_id = 42;"
    assert not env.valid_actions()[ACTION_DROP]
    env.step(ACTION_CREATE)
    env._current_sql = "SELECT * FROM orders WHERE customer_id = 42;"
    assert env.valid_actions()[ACTION_DROP]


def test_greedy_create_beats_keep():
    env = make_env()
    env.set_task(0)
    keep = env.evaluate(lambda o, m: ACTION_KEEP)
    create = env.evaluate(lambda o, m: ACTION_CREATE if m[ACTION_CREATE] else ACTION_KEEP)
    assert create > keep
