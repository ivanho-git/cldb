"""Real-time heavy-traffic simulator for the CLDB live app.

This module runs the actual DQN index-selection agent inside the benchmark
MDP at a configurable queries-per-second rate. It is the engine behind the
live dashboard: a background worker steps the environment, acts, learns from
the replay buffer, detects workload drift and consolidates with EWC, while a
thread-safe snapshot exposes everything needed to inspect the agent.

State is deliberately plain Python (no tensors, no numpy arrays) so the
snapshot can be sent as JSON over a WebSocket without further conversion.
"""
import copy
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

import numpy as np
import torch

from cldb.counterfactual import build_counterfactual_set
from cldb.drift_detection import DriftDetector
from cldb.replay_buffer import DatabaseAwarePriorityReplay
from cldb.rl_agent import ACTION_CREATE, ACTION_DROP, ACTION_KEEP, DQNAgent
from cldb.sim_env import IndexAdvisorEnv, PHASE_SELECTIVITY, PHASE_TEMPLATES

ACTION_NAMES = {ACTION_KEEP: "KEEP", ACTION_CREATE: "CREATE", ACTION_DROP: "DROP"}
TASK_SEQUENCE = sorted(PHASE_TEMPLATES.keys())
TASK_NAMES = {
    0: "OLTP",
    1: "Filters",
    2: "Filters (conflict)",
    3: "Analytics",
    4: "Analytics (conflict)",
    5: "OLTP (conflict)",
    6: "OLTP (return)",
}


@dataclass
class TrafficConfig:
    qps: float = 60.0
    steps_per_phase: int = 400
    budget: int = 2
    train_every: int = 4
    warmup: int = 64
    batch_size: int = 64
    learning: bool = True
    ewc: bool = True
    cwil: bool = False
    epsilon_end: float = 0.05
    reheat_epsilon: float = 0.35
    seed: int = 0
    history_size: int = 600
    recent_size: int = 40

    def clamped(self) -> "TrafficConfig":
        self.qps = float(min(1000.0, max(1.0, self.qps)))
        self.steps_per_phase = int(min(20000, max(20, self.steps_per_phase)))
        self.budget = int(min(8, max(1, self.budget)))
        self.train_every = int(min(100, max(1, self.train_every)))
        self.warmup = int(min(4096, max(8, self.warmup)))
        self.batch_size = int(min(512, max(8, self.batch_size)))
        self.epsilon_end = float(min(0.5, max(0.0, self.epsilon_end)))
        self.reheat_epsilon = float(min(1.0, max(0.0, self.reheat_epsilon)))
        return self


class HeavyTrafficSimulator:
    """Threaded, hot-loop traffic driver with a live JSON-safe snapshot."""

    def __init__(self, config: Optional[TrafficConfig] = None):
        self.config = (config or TrafficConfig()).clamped()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._paused = False
        self._error: Optional[str] = None
        self._started_at: Optional[float] = None
        self._build(self.config.seed)

    # ------------------------------------------------------------------ setup
    def _build(self, seed: int):
        cfg = self.config
        self.env = IndexAdvisorEnv(seed=seed, budget=cfg.budget, horizon=10 ** 9)
        self.agent = DQNAgent(
            feature_dim=self.env.feature_dim,
            embedding_dim=self.env.embedding_dim,
            context_dim=3,
            index_state_dim=3,
            hidden_dims=(128, 64),
            learning_rate=7e-4,
            gamma=0.97,
            batch_size=cfg.batch_size,
            epsilon_start=1.0,
            epsilon_end=cfg.epsilon_end,
            epsilon_decay_steps=2000,
            target_update_interval=150,
            ewc_lambda=0.4 if cfg.ewc else 0.0,
            seed=seed,
        )
        self.buffer = DatabaseAwarePriorityReplay(capacity=50000, alpha=0.6, beta=0.4)
        self.detector = DriftDetector(window_size=24, drift_threshold=0.22)

        self._phase_index = 0
        self._task = TASK_SEQUENCE[0]
        self._cycle = 0
        self._steps_in_phase = 0
        self._phase_had_consolidation = False
        self._phase_states: Deque[np.ndarray] = deque(maxlen=256)

        self._obs = self.env.reset(task_id=self._task)
        self._steps = 0
        self._total_queries = 0
        self._episodes = 0
        self._reward_sum = 0.0
        self._reward_window: Deque[float] = deque(maxlen=200)
        self._action_counts = Counter()
        self._phase_stats: Dict[int, Dict] = {}
        self._history: Deque[Dict] = deque(maxlen=cfg.history_size)
        self._recent: Deque[Dict] = deque(maxlen=cfg.recent_size)
        self._consolidations: List[Dict] = []
        self._drift_events = 0
        self._qps_marks: Deque[tuple] = deque(maxlen=512)
        self._qps_actual = 0.0
        self._current: Dict = {}
        self._last_loss = 0.0
        self._last_td = 0.0
        self._last_ewc = 0.0
        self._last_cf = 0.0
        self._last_qps_snapshot = time.time()
        self._history_stride = max(1, int(self.config.qps / 12)) if self.config.qps >= 12 else 1
        self._t0_monotonic = time.monotonic()
        self._error = None

    # --------------------------------------------------------------- lifecycle
    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    def start(self):
        with self._lock:
            self._paused = False
            if self._running:
                return
            self._stop_event.clear()
            self._running = True
            self._started_at = time.time()
            self._thread = threading.Thread(target=self._run, name="cldb-traffic", daemon=True)
            self._thread.start()

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self, timeout: float = 3.0):
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._lock:
            self._running = False
            self._thread = None

    def reset(self, seed: Optional[int] = None):
        was_running = self._running
        self.stop()
        with self._lock:
            if seed is not None:
                self.config.seed = int(seed)
            self._build(self.config.seed)
        if was_running:
            self.start()

    def configure(self, **kwargs) -> Dict:
        with self._lock:
            aliases = {"rate": "qps", "queries_per_second": "qps"}
            for key, value in kwargs.items():
                key = aliases.get(key, key)
                if not hasattr(self.config, key) or key in ("history_size", "recent_size"):
                    continue
                setattr(self.config, key, value)
            self.config.clamped()
            self._history_stride = max(1, int(self.config.qps / 12)) if self.config.qps >= 12 else 1
            self.agent.ewc.lambda_param = 0.4 if self.config.ewc else 0.0
            self.agent.epsilon_end = self.config.epsilon_end
            self.env.budget = self.config.budget
            return self.snapshot_locked()

    def run_for(self, seconds: float) -> Dict:
        self.start()
        try:
            time.sleep(max(0.0, seconds))
        finally:
            self.stop()
        return self.snapshot()

    # ------------------------------------------------------------------- loop
    def _run(self):
        tick = 0.02
        credit = 0.0
        last = time.perf_counter()
        while not self._stop_event.is_set():
            now = time.perf_counter()
            dt = now - last
            last = now

            if self._paused:
                credit = 0.0
                time.sleep(0.05)
                continue

            qps = max(0.1, float(self.config.qps))
            credit += dt * qps
            credit = min(credit, qps * 0.25)
            n = int(credit)
            if n <= 0:
                time.sleep(tick)
                continue

            credit -= n
            try:
                for _ in range(min(n, 500)):
                    if self._stop_event.is_set():
                        break
                    self._step()
            except Exception as exc:  # keep the app alive and surface the error
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
                self._stop_event.set()
                break
            self._refresh_rate()

        with self._lock:
            self._running = False

    def _step(self):
        cfg = self.config
        env = self.env
        obs = self._obs
        mask = env.valid_actions()
        state = env.obs_vector(obs)

        if cfg.learning and cfg.cwil:
            try:
                cf = build_counterfactual_set(env.what_if(), env.obs_vector)
                cf_stats = self.agent.train_counterfactual(state, cf)
                if cf_stats:
                    self._last_cf = cf_stats["cf_loss"]
            except Exception:
                pass

        sql = env.current_query or ""
        predicate = env._parse(sql).predicate_columns

        action, q, exploring = self.agent.act(state, valid_actions=mask, explore=True)
        greedy_action = int(np.argmax(np.where(mask, q, -1e9)))

        next_obs, reward, done, info = env.step(action)

        self.buffer.add_transition(
            f"q{self._total_queries}", action, reward, False,
            state=obs, next_state=next_obs)

        if cfg.learning and len(self.buffer) >= max(cfg.warmup, cfg.batch_size):
            if self._steps % cfg.train_every == 0:
                stats = self.agent.train_from_buffer(self.buffer, batch_size=cfg.batch_size)
                if stats:
                    self._last_loss = stats["loss"]
                    self._last_td = stats["td_mean"]
                    self._last_ewc = stats["ewc_penalty"]

        if cfg.ewc and self.detector.add_experience(obs[1]):
            with self._lock:
                self._drift_events += 1
            self._consolidate("drift")
            self._reheat_exploration()

        # ---- bookkeeping -------------------------------------------------
        q_map = {ACTION_NAMES[i]: round(float(q[i]), 4) for i in range(len(q))}
        mask_map = {ACTION_NAMES[i]: bool(mask[i]) for i in range(len(mask))}
        now = time.monotonic() - self._t0_monotonic

        with self._lock:
            self._steps += 1
            self._total_queries += 1
            self._steps_in_phase += 1
            self._reward_sum += reward
            self._reward_window.append(reward)
            self._action_counts[action] += 1
            self._phase_states.append(state)

            phase = self._phase_stats.setdefault(
                self._task, {"steps": 0, "reward_sum": 0.0, "actions": Counter()})
            phase["steps"] += 1
            phase["reward_sum"] += reward
            phase["actions"][action] += 1

            self._current = {
                "step": self._total_queries,
                "sql": sql,
                "phase": self._task,
                "phase_name": TASK_NAMES.get(self._task, str(self._task)),
                "action": ACTION_NAMES[action],
                "action_id": action,
                "greedy_action": ACTION_NAMES[greedy_action],
                "exploring": exploring,
                "confidence": round(float(np.exp(q[action] - np.max(q))), 4),
                "q_values": q_map,
                "mask": mask_map,
                "reward": round(float(reward), 4),
                "latency": round(float(info.get("latency", 0.0)), 4),
                "n_indexes": int(info.get("n_indexes", 0)),
                "active_indexes": env.active_indexes,
                "predicate_columns": list(predicate),
                "epsilon": round(float(self.agent.epsilon), 4),
                "uptime": round(now, 2),
            }

            if self._total_queries % self._history_stride == 0 or exploring:
                self._history.append({
                    "t": round(now, 2),
                    "reward": round(float(reward), 4),
                    "reward_ma": round(float(np.mean(self._reward_window)), 4),
                    "epsilon": round(float(self.agent.epsilon), 4),
                    "loss": round(float(self._last_loss), 5),
                    "td": round(float(self._last_td), 5),
                    "qps": round(float(self._qps_actual), 1),
                    "action": ACTION_NAMES[action],
                    "phase": self._task,
                    "phase_name": TASK_NAMES.get(self._task, str(self._task)),
                })
            self._recent.append({
                "step": self._total_queries,
                "query": sql,
                "phase": self._task,
                "phase_name": TASK_NAMES.get(self._task, str(self._task)),
                "action": ACTION_NAMES[action],
                "reward": round(float(reward), 4),
                "exploring": exploring,
                "q_values": q_map,
            })

            self._obs = next_obs
            advance = self._steps_in_phase >= self.config.steps_per_phase

        if advance:
            self._advance_phase()

        if done:
            with self._lock:
                self._episodes += 1
                self._obs = env.reset(task_id=self._task)

    def _advance_phase(self):
        if self.config.ewc and not self._phase_had_consolidation:
            self._consolidate("boundary")

        with self._lock:
            self._phase_index += 1
            if self._phase_index >= len(TASK_SEQUENCE):
                self._phase_index = 0
                self._cycle += 1
            self._task = TASK_SEQUENCE[self._phase_index]
            self._steps_in_phase = 0
            self._phase_had_consolidation = False
            self._phase_states.clear()
            self.env.set_task(self._task)
            self._obs = self.env.reset(task_id=self._task)

        self._reheat_exploration()

    def _reheat_exploration(self):
        """Restart epsilon-greedy exploration after a workload change.

        Without this the schedule decays to its floor within the first phase
        and every later phase is evaluated greedily with a policy learned on a
        different distribution -- which reads as a frozen decision readout.
        """
        target = self.config.reheat_epsilon
        if target > 0 and self.config.learning:
            self.agent.reheat_epsilon(target)

    def _consolidate(self, source: str):
        with self._lock:
            if not self.config.ewc or len(self._phase_states) < 32:
                return
            states = torch.tensor(np.stack(list(self._phase_states)), dtype=torch.float32)
            phase_id = f"{source}_c{self._cycle}_p{self._phase_index}"
            task = self._task
            steps = self._steps_in_phase
            queries = self._total_queries

        try:
            self.agent.consolidate(phase_id, states)
        except Exception:
            return

        with self._lock:
            self._last_ewc = float(self.agent.ewc.compute_ewc_loss().item())
            self._consolidations.append({
                "phase": phase_id,
                "source": source,
                "task": task,
                "steps": steps,
                "queries": queries,
            })
            self._phase_had_consolidation = True
            self._prune_ewc()

    def _prune_ewc(self, keep: int = 6):
        """Bound the per-update EWC penalty for long-running live sessions.

        The full benchmark keeps every phase; a live demo runs indefinitely, so
        retaining only the most recent phases keeps the training step cost flat.
        """
        weights = self.agent.ewc.optimal_weights
        while len(weights) > keep:
            oldest = next(iter(weights))
            weights.pop(oldest, None)
            self.agent.ewc.fisher_matrices.pop(oldest, None)

    def _refresh_rate(self):
        now = time.monotonic()
        self._qps_marks.append((now, self._total_queries))
        while len(self._qps_marks) > 2 and now - self._qps_marks[0][0] > 2.0:
            self._qps_marks.popleft()
        if len(self._qps_marks) >= 2:
            t0, q0 = self._qps_marks[0]
            t1, q1 = self._qps_marks[-1]
            if t1 > t0:
                self._qps_actual = (q1 - q0) / (t1 - t0)
        if now - self._last_qps_snapshot >= 0.5:
            self._last_qps_snapshot = now

    # ---------------------------------------------------------------- snapshot
    def snapshot(self) -> Dict:
        with self._lock:
            return self.snapshot_locked()

    def snapshot_locked(self) -> Dict:
        cfg = self.config
        now = time.time()
        uptime = (now - self._started_at) if (self._running and self._started_at) else 0.0
        history = list(self._history)
        recent = list(self._recent)
        current = copy.deepcopy(self._current)
        total = max(1, self._total_queries)
        actions = {
            name: {
                "count": int(self._action_counts.get(action, 0)),
                "rate": round(self._action_counts.get(action, 0) / total, 4),
            }
            for action, name in ACTION_NAMES.items()
        }
        phases = []
        for task in TASK_SEQUENCE:
            data = self._phase_stats.get(task)
            if not data or data["steps"] == 0:
                phases.append({
                    "task": task, "name": TASK_NAMES.get(task, str(task)),
                    "steps": 0, "mean_reward": None,
                    "keep_rate": 0.0, "create_rate": 0.0, "drop_rate": 0.0,
                    "is_conflict": any(v == "low" for v in PHASE_SELECTIVITY.get(task, {}).values()),
                })
                continue
            steps = data["steps"]
            act = data["actions"]
            phases.append({
                "task": task,
                "name": TASK_NAMES.get(task, str(task)),
                "steps": steps,
                "mean_reward": round(data["reward_sum"] / steps, 4),
                "keep_rate": round(act.get(ACTION_KEEP, 0) / steps, 4),
                "create_rate": round(act.get(ACTION_CREATE, 0) / steps, 4),
                "drop_rate": round(act.get(ACTION_DROP, 0) / steps, 4),
                "is_conflict": any(v == "low" for v in PHASE_SELECTIVITY.get(task, {}).values()),
            })

        return {
            "type": "state",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": {
                "running": self._running,
                "paused": self._paused,
                "error": self._error,
                "uptime": round(uptime, 2),
                "started_at": (datetime.fromtimestamp(self._started_at, timezone.utc).isoformat()
                               if self._started_at else None),
            },
            "config": {
                "qps": cfg.qps, "steps_per_phase": cfg.steps_per_phase,
                "budget": cfg.budget, "train_every": cfg.train_every,
                "warmup": cfg.warmup, "batch_size": cfg.batch_size,
                "learning": cfg.learning, "ewc": cfg.ewc, "cwil": cfg.cwil,
                "epsilon_end": cfg.epsilon_end, "reheat_epsilon": cfg.reheat_epsilon,
                "seed": cfg.seed,
            },
            "traffic": {
                "qps_target": cfg.qps,
                "qps_actual": round(self._qps_actual, 1),
                "total_queries": self._total_queries,
                "steps": self._steps,
                "episodes": self._episodes,
                "action_counts": {name: actions[name]["count"] for name in actions},
                "action_rates": {name: actions[name]["rate"] for name in actions},
            },
            "phase": {
                "task": self._task,
                "name": TASK_NAMES.get(self._task, str(self._task)),
                "index": self._phase_index,
                "cycle": self._cycle,
                "steps_in_phase": self._steps_in_phase,
                "steps_per_phase": cfg.steps_per_phase,
                "progress": round(self._steps_in_phase / max(1, cfg.steps_per_phase), 4),
                "is_conflict": any(v == "low" for v in PHASE_SELECTIVITY.get(self._task, {}).values()),
            },
            "agent": {
                "epsilon": round(float(self.agent.epsilon), 4),
                "updates": int(self.agent.updates),
                "cf_updates": int(self.agent.cf_updates),
                "replay_size": len(self.buffer),
                "replay_capacity": self.buffer.capacity,
                "loss": round(float(self._last_loss), 5),
                "td": round(float(self._last_td), 5),
                "ewc_penalty": round(float(self._last_ewc), 5),
                "cf_loss": round(float(self._last_cf), 5),
                "consolidations": len(self._consolidations),
                "drift_events": self._drift_events,
                "mean_reward": round(self._reward_sum / total, 4),
                "reward_ma": round(float(np.mean(self._reward_window)) if self._reward_window else 0.0, 4),
            },
            "current": current,
            "history": history,
            "recent": recent,
            "phases": phases,
            "consolidation_log": list(self._consolidations[-10:]),
        }

    def summary(self) -> Dict:
        state = self.snapshot()
        phases = [p for p in state["phases"] if p["steps"] > 0]
        return {
            "queries": state["traffic"]["total_queries"],
            "qps_target": state["traffic"]["qps_target"],
            "qps_actual": state["traffic"]["qps_actual"],
            "updates": state["agent"]["updates"],
            "replay_size": state["agent"]["replay_size"],
            "epsilon": state["agent"]["epsilon"],
            "mean_reward": state["agent"]["mean_reward"],
            "action_rates": state["traffic"]["action_rates"],
            "phases": phases,
            "drift_events": state["agent"]["drift_events"],
            "consolidations": state["agent"]["consolidations"],
            "error": state["status"]["error"],
        }
