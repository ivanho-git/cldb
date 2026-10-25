"""Counterfactual What-If Learning (CWIL) for index selection.

An index advisor is unusual among RL problems in that a *what-if oracle* is
often available: a cost model (offline) or a hypothetical-index planner
(online) can estimate the cost of an action without materialising it. CWIL
exploits this at every decision point: it rolls out *all* structurally valid
actions from an identical environment snapshot, producing one-step
counterfactual outcomes, and trains the Q-network to **rank** the actions in
the order the oracle observes.

This is the mechanism that teaches the agent the cost of a *non-selective*
index without it having to experience the regret first. Replay-based DQN
only ever learns from the action it took; CWIL learns from the actions it
did not take. The two objectives are complementary: replay provides
off-policy, temporally extended experience, while CWIL provides a dense,
immediate preference signal over the action set.

The oracle interface intentionally says nothing about how the counterfactual
cost is obtained. ``SimulatorOracle`` wraps the benchmark's analytic cost
model; a production deployment can implement the same protocol with
``EXPLAIN`` over hypothetical indexes.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Protocol, Tuple

import numpy as np


@dataclass
class CounterfactualSet:
    """One-step counterfactual transitions for every valid action at a state."""

    actions: np.ndarray       # (k,)      action ids
    rewards: np.ndarray       # (k,)      immediate reward of each action
    next_states: np.ndarray   # (k, d)    successor state for each action
    dones: np.ndarray         # (k,)      terminal flags
    next_masks: np.ndarray    # (k, |A|)  valid-action masks in the successors

    def __len__(self) -> int:
        return int(self.actions.shape[0])


class WhatIfOracle(Protocol):
    """Anything that can evaluate all valid actions at the current state."""

    def what_if(self) -> Dict[int, Tuple[object, float, bool, np.ndarray, Dict]]:
        ...


def build_counterfactual_set(
    branches: Dict[int, Tuple[object, float, bool, np.ndarray, Dict]],
    obs_vector: Callable[[object], np.ndarray],
) -> CounterfactualSet:
    """Convert oracle branches into stacked arrays for the Q-network."""
    if not branches:
        raise ValueError("what-if oracle returned no valid actions")

    actions = sorted(branches)
    rewards = []
    next_states = []
    dones = []
    next_masks = []
    for action in actions:
        next_obs, reward, done, next_mask, _ = branches[action]
        rewards.append(float(reward))
        next_states.append(np.asarray(obs_vector(next_obs), dtype=np.float32))
        dones.append(float(done))
        next_masks.append(np.asarray(next_mask, dtype=bool))

    return CounterfactualSet(
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        next_states=np.stack(next_states),
        dones=np.asarray(dones, dtype=np.float32),
        next_masks=np.stack(next_masks),
    )


class SimulatorOracle:
    """What-if oracle backed by :class:`cldb.sim_env.IndexAdvisorEnv`."""

    def __init__(self, env):
        self.env = env

    def what_if(self) -> Dict[int, Tuple[object, float, bool, np.ndarray, Dict]]:
        return self.env.what_if()

    def counterfactuals(self, obs_vector: Callable[[object], np.ndarray]) -> CounterfactualSet:
        return build_counterfactual_set(self.what_if(), obs_vector)
