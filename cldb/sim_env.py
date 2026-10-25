"""
Reproducible index-advisor benchmark environment.

Uses a metadata cost model instead of a live PostgreSQL instance so that RL
experiments are deterministic, fast, and reproducible. Each step exposes one
query, and the agent chooses KEEP / CREATE_INDEX / DROP_INDEX on the query's
predicate columns. The reward is the negative operating cost of the system
after the action (query latency + index maintenance), so the agent learns to
build indexes that pay off and drop indexes that no longer do.

The environment is a genuine MDP: the action mutates the index configuration,
which is part of the next state, and index creation is amortised over the
recurring queries that follow.
"""
import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cldb.parser import SQLParser
from cldb.features import FeatureEncoder
from cldb.embeddings import QueryEmbedder
from cldb.rl_agent import ACTION_KEEP, ACTION_CREATE, ACTION_DROP, NUM_ACTIONS, build_state

CONTEXT_DIM = 3
INDEX_STATE_DIM = 3

Obs = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

# Per-phase column selectivity. A high-selectivity predicate benefits from an
# index; a low-selectivity predicate does not. Crucially, the same predicate
# can be high-selectivity in one phase and low-selectivity in another, so the
# optimal action for an *identical* observation flips across phases. The only
# cue distinguishing such phases is the workload-context signal, which makes
# this a genuinely catastrophic-forgetting-prone sequence (a permuted bandit /
# permuted-MNIST analogue). Which columns are selective is not revealed to the
# agent; it must be learned from reward feedback.
PHASE_SELECTIVITY: Dict[int, Dict[str, str]] = {
    0: {"customer_id": "high", "product_id": "high"},
    1: {"category": "high", "status": "high"},
    2: {"category": "low", "status": "low"},      # conflicts with phase 1
    3: {"order_date": "high", "region": "high"},
    4: {"order_date": "low", "region": "low"},    # conflicts with phase 3
    5: {"customer_id": "low", "product_id": "low"},  # conflicts with phase 0
    6: {"customer_id": "high", "product_id": "high"},  # return to phase 0
}


# Workload phases (tasks). Each is a query template list; the predicate
# columns differ across phases so that indexes useful in one phase become
# useless in another -- the source of catastrophic forgetting.
_OLTP_V1 = [
    "SELECT * FROM orders WHERE customer_id = 42;",
    "SELECT * FROM order_items WHERE product_id = 7;",
    "SELECT * FROM customers WHERE customer_id = 15;",
]
_FILTERS_V1 = [
    "SELECT * FROM products WHERE category = 'Electronics';",
    "SELECT * FROM orders WHERE status = 'PENDING';",
    "SELECT * FROM products WHERE category = 'Books';",
]
_ANALYTICS_V1 = [
    "SELECT * FROM orders WHERE order_date > '2024-01-01';",
    "SELECT * FROM customers WHERE region = 'EU';",
    "SELECT * FROM orders WHERE order_date > '2024-06-01';",
]

PHASE_TEMPLATES: Dict[int, List[str]] = {
    0: _OLTP_V1,       # customer_id / product_id  (high)
    1: _FILTERS_V1,    # category / status         (high)
    2: _FILTERS_V1,    # identical queries, opposite selectivity (low)  -- conflict
    3: _ANALYTICS_V1,  # order_date / region       (high)
    4: _ANALYTICS_V1,  # identical queries, opposite selectivity (low)  -- conflict
    5: _OLTP_V1,       # identical queries, opposite selectivity (low)  -- conflict
    6: _OLTP_V1,       # return to phase 0         (high)
}


class IndexAdvisorEnv:
    def __init__(self, embedder: Optional[QueryEmbedder] = None,
                 encoder: Optional[FeatureEncoder] = None, parser: Optional[SQLParser] = None,
                 budget: int = 4, build_cost: float = 0.5, resource_cost: float = 0.2,
                 latency_expensive: float = 1.0, latency_cheap: float = 0.1,
                 horizon: int = 60, seed: int = 0):
        self.parser = parser or SQLParser()
        self.encoder = encoder or FeatureEncoder()
        self.embedder = embedder or QueryEmbedder()
        self.budget = budget
        self.build_cost = build_cost
        self.resource_cost = resource_cost
        self.latency_expensive = latency_expensive
        self.latency_cheap = latency_cheap
        self.horizon = horizon
        self._rng = np.random.default_rng(seed)

        self.feature_dim = self.encoder.get_feature_dim()
        self.embedding_dim = 384

        self.num_tasks = len(PHASE_TEMPLATES)
        self.task_id = 0
        # Selectivity can be decoupled from the phase cue to model *silent*
        # drift: the query templates and the context signal look unchanged,
        # but which predicates benefit from an index has flipped.
        self.selectivity_phase: Optional[int] = None
        self.step_idx = 0

        # Active index configuration: id -> frozenset(predicate columns)
        self.active: Dict[int, frozenset] = {}
        self._next_index_id = 0
        self._current_sql: Optional[str] = None

        self._cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ helpers
    def _parse(self, sql: str):
        if sql not in self._cache:
            parsed = self.parser.parse(sql)
            self._cache[sql] = parsed
        return self._cache[sql]

    def _embed(self, sql: str) -> np.ndarray:
        key = "emb::" + sql
        if key not in self._cache:
            self._cache[key] = self.embedder.embed(sql)
        return self._cache[key]

    def _predicate_key(self, parsed) -> frozenset:
        return frozenset(parsed.predicate_columns)

    def _is_covered(self, predicate: frozenset) -> bool:
        if not predicate:
            return False
        return any(predicate.issubset(cols) for cols in self.active.values())

    def _is_selective(self, predicate: frozenset) -> bool:
        phase = self.task_id if self.selectivity_phase is None else self.selectivity_phase
        mapping = PHASE_SELECTIVITY.get(phase, {})
        return any(mapping.get(c, "high") == "high" for c in predicate)

    def _latency(self, predicate: frozenset) -> float:
        # An index only helps if it covers the predicate *and* the predicate is
        # selective. Indexing a low-selectivity column yields no benefit.
        if self._is_covered(predicate) and self._is_selective(predicate):
            return self.latency_cheap
        return self.latency_expensive

    def _build_obs(self, sql: str) -> Obs:
        parsed = self._parse(sql)
        features = self.encoder.encode(parsed)
        embedding = self._embed(sql)
        predicate = self._predicate_key(parsed)

        # Phase identity is exposed as a normalized cue (a real drift detector
        # assigns a new phase id), giving the agent an observable task signal.
        context = np.array([
            self.task_id / max(1, self.num_tasks - 1),
            self.step_idx / max(1, self.horizon),
            len(self.active) / max(1, self.budget),
        ], dtype=np.float32)

        index_state = np.array([
            len(self.active) / max(1, self.budget),
            1.0 if self._is_covered(predicate) else 0.0,
            max(0.0, (self.budget - len(self.active)) / max(1, self.budget)),
        ], dtype=np.float32)

        return (features, embedding, context, index_state)

    @staticmethod
    def obs_vector(obs: Obs) -> np.ndarray:
        return build_state(obs[0], obs[1], obs[2], obs[3])

    @property
    def current_query(self) -> Optional[str]:
        return self._current_sql

    @property
    def active_indexes(self) -> List[List[str]]:
        return [sorted(cols) for cols in self.active.values()]

    # --------------------------------------------------------------------- API
    def set_task(self, task_id: int):
        self.task_id = task_id

    def set_selectivity_phase(self, phase_id: Optional[int]):
        """Decouple true selectivity from the observed phase/task cue.

        ``None`` restores the default coupling (selectivity follows task_id).
        """
        self.selectivity_phase = phase_id

    def reset(self, task_id: Optional[int] = None,
              selectivity_phase: Optional[int] = None) -> Obs:
        if task_id is not None:
            self.task_id = task_id
        if selectivity_phase is not None:
            self.selectivity_phase = selectivity_phase
        self.step_idx = 0
        self.active = {}
        self._next_index_id = 0
        self._current_sql = self._sample_query()
        return self._build_obs(self._current_sql)

    def _sample_query(self) -> str:
        templates = PHASE_TEMPLATES[self.task_id]
        return templates[int(self._rng.integers(len(templates)))]

    def valid_actions(self) -> np.ndarray:
        parsed = self._parse(self._current_sql)
        predicate = self._predicate_key(parsed)
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        mask[ACTION_KEEP] = True
        mask[ACTION_CREATE] = bool(predicate) and len(self.active) < self.budget and not self._is_covered(predicate)
        mask[ACTION_DROP] = bool(predicate) and any(cols == predicate for cols in self.active.values())
        return mask

    def step(self, action: int) -> Tuple[Obs, float, bool, Dict]:
        parsed = self._parse(self._current_sql)
        predicate = self._predicate_key(parsed)
        mask = self.valid_actions()

        one_time_cost = 0.0
        if action == ACTION_CREATE and mask[ACTION_CREATE]:
            self.active[self._next_index_id] = predicate
            self._next_index_id += 1
            one_time_cost += self.build_cost
        elif action == ACTION_DROP and mask[ACTION_DROP]:
            for idx_id, cols in list(self.active.items()):
                if cols == predicate:
                    del self.active[idx_id]
                    break

        latency = self._latency(predicate)
        resource = self.resource_cost * len(self.active)
        reward = -(latency + resource) - one_time_cost

        self.step_idx += 1
        done = self.step_idx >= self.horizon
        info = {"latency": latency, "n_indexes": len(self.active), "action": action}

        self._current_sql = self._sample_query()
        next_obs = self._build_obs(self._current_sql)
        return next_obs, float(reward), done, info

    # ------------------------------------------------------------ what-if oracle
    def snapshot(self) -> Dict[str, Any]:
        """Capture the complete environment state, including the RNG stream."""
        return {
            "task_id": self.task_id,
            "selectivity_phase": self.selectivity_phase,
            "step_idx": self.step_idx,
            "active": dict(self.active),
            "next_index_id": self._next_index_id,
            "current_sql": self._current_sql,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def restore(self, snapshot: Dict[str, Any]):
        self.task_id = snapshot["task_id"]
        self.selectivity_phase = snapshot["selectivity_phase"]
        self.step_idx = snapshot["step_idx"]
        self.active = dict(snapshot["active"])
        self._next_index_id = snapshot["next_index_id"]
        self._current_sql = snapshot["current_sql"]
        self._rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])

    def what_if(self) -> Dict[int, Tuple[Obs, float, bool, np.ndarray, Dict]]:
        """Evaluate every structurally valid action without committing any.

        Each branch is rolled out from an identical snapshot, so all actions
        observe the same successor query and differ only in the index
        mutation and its reward. The environment is restored exactly,
        including the RNG stream, so callers can then take the action they
        actually choose.

        Returns a mapping ``action -> (next_obs, reward, done, next_mask, info)``.
        """
        if self._current_sql is None:
            raise RuntimeError("what_if() requires reset() to have been called")

        snapshot = self.snapshot()
        branches: Dict[int, Tuple[Obs, float, bool, np.ndarray, Dict]] = {}
        try:
            for action in np.flatnonzero(self.valid_actions()):
                self.restore(snapshot)
                next_obs, reward, done, info = self.step(int(action))
                branches[int(action)] = (next_obs, reward, done,
                                         self.valid_actions(), info)
        finally:
            self.restore(snapshot)
        return branches

    # --------------------------------------------------------------- evaluation
    def evaluate(self, policy_fn, eval_steps: Optional[int] = None, seed: int = 12345) -> float:
        """Greedy rollout of a policy over the current task; returns mean reward."""
        saved_rng = self._rng
        self._rng = np.random.default_rng(seed)
        try:
            obs = self.reset(task_id=self.task_id)
            total = 0.0
            steps = eval_steps or self.horizon
            for _ in range(steps):
                mask = self.valid_actions()
                action = policy_fn(obs, mask)
                obs, reward, done, _ = self.step(action)
                total += reward
                if done:
                    obs = self.reset(task_id=self.task_id)
            return total / steps
        finally:
            self._rng = saved_rng
