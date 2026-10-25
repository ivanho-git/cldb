"""
Canonical CLDB online pipeline.

A single class, ``CLDBPipeline``, drives the whole system: parsing and
encoding, reinforcement-learning action selection, structural action
masking, safety validation, execution, latency measurement, transition
storage, background DQN updates, drift detection and EWC consolidation.
``ProductionPipeline`` is retained as an alias for callers that used the old
name; there is no second implementation.
"""
import hashlib
import math
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch

from cldb.config import get_config
from cldb.decision_engine import DecisionEngine
from cldb.drift_detection import DriftDetector
from cldb.embeddings import QueryEmbedder
from cldb.enhanced_cl_engine import CLEngine, CONTEXT_DIM, EMBEDDING_DIM
from cldb.enhanced_safety import SafetyLayer
from cldb.enhanced_vector_store import VectorStore
from cldb.executor import Executor
from cldb.features import FeatureEncoder
from cldb.feedback import FeedbackLoop
from cldb.logger import get_logger
from cldb.metadata_repo import MetadataRepo, OptimizationDecision, QueryLog, DBIndex, DBTable
from cldb.metrics import get_metrics
from cldb.online_scheduler import OnlineScheduler
from cldb.parser import QueryFeatures, SQLParser
from cldb.performance_collector import PerformanceCollector
from cldb.replay_buffer import DatabaseAwarePriorityReplay
from cldb.resilience import get_database_breaker, get_embedding_breaker
from cldb.reward import RewardCalculator
from cldb.rl_agent import (
    ACTION_CREATE,
    ACTION_DROP,
    ACTION_KEEP,
    NUM_ACTIONS,
    DQNAgent,
    build_state,
)
from cldb.workload_monitor import WorkloadMonitor


class CLDBPipeline:
    def __init__(self, repo: MetadataRepo, vector_store: VectorStore,
                 feature_encoder: Optional[FeatureEncoder] = None,
                 embedder: Optional[QueryEmbedder] = None,
                 cl_engine: Optional[CLEngine] = None,
                 confidence_threshold: float = 0.6, db_id: int = 1):
        self.config = get_config()
        self.logger = get_logger("pipeline")
        self.metrics = get_metrics()

        self.repo = repo
        self.vector_store = vector_store
        self.db_id = db_id

        self.parser = SQLParser()
        self.feature_encoder = feature_encoder or FeatureEncoder()
        self.embedder = embedder or QueryEmbedder()
        feature_dim = self.feature_encoder.get_feature_dim()

        # The RL agent is the decision maker. ``cl_engine`` is retained for
        # supervised warm-starting only.
        self.cl_engine = cl_engine or CLEngine(feature_dim=feature_dim)
        self.decision_engine = DecisionEngine()

        self.agent = DQNAgent(
            feature_dim=feature_dim, embedding_dim=EMBEDDING_DIM, context_dim=CONTEXT_DIM,
            index_state_dim=3, ewc_lambda=self.config.ewc.lambda_param,
        )
        self.max_indexes = self.config.safety.max_indexes_per_table
        self._recent_states = deque(maxlen=256)
        self._last_transition = None

        self.workload_monitor = WorkloadMonitor(repo)
        self.safety_layer = SafetyLayer(
            repo,
            confidence_threshold=confidence_threshold,
            max_indexes_per_table=self.config.safety.max_indexes_per_table,
            rollback_on_regression=self.config.safety.rollback_on_regression,
            regression_threshold_ms=self.config.safety.regression_threshold_ms,
        )
        self.executor = Executor(repo)
        self.performance_collector = PerformanceCollector(repo)
        self.feedback_loop = FeedbackLoop(repo, vector_store)

        self._context_buffer = np.zeros((1, CONTEXT_DIM), dtype=np.float32)
        self._current_task_id = 0

        self._db_breaker = get_database_breaker()
        self._emb_breaker = get_embedding_breaker()

        self.replay_buffer = DatabaseAwarePriorityReplay(
            capacity=self.config.replay.mem_size,
            alpha=self.config.replay.alpha,
            beta=self.config.replay.beta,
        )
        self.reward_calculator = RewardCalculator(
            latency_weight=self.config.reward.latency_weight,
            regression_penalty_weight=self.config.reward.regression_penalty_weight,
            resource_penalty=self.config.reward.resource_penalty,
        )
        self.drift_detector = DriftDetector(
            window_size=self.config.drift.window_size,
            drift_threshold=self.config.drift.drift_threshold,
        )
        self.online_scheduler = OnlineScheduler(
            train_callback=self._execute_online_training,
            drift_detector=self.drift_detector,
            check_interval_seconds=self.config.drift.check_interval_seconds,
            min_queries_before_train=self.config.drift.min_queries_before_train,
        )
        self.online_scheduler.start()

        self.logger.info("CLDB pipeline initialized", db_id=db_id,
                         confidence_threshold=confidence_threshold)

    # ------------------------------------------------------------- lifecycle
    def close(self):
        self.online_scheduler.stop()

    def _execute_online_training(self):
        start = time.time()
        try:
            stats = self.agent.train_from_buffer(
                self.replay_buffer, batch_size=self.config.replay.batch_size)
            if stats is not None:
                self.metrics.ewc_penalty.set(stats["ewc_penalty"])
                self.metrics.training_loss.observe(
                    stats["loss"], phase=f"task_{self._current_task_id}")
            self.metrics.training_time.observe(time.time() - start)
        except Exception as e:
            self.logger.error(f"Online training failed: {e}")

    # --------------------------------------------------------------- features
    def _get_context_features(self) -> np.ndarray:
        stats = self.workload_monitor.get_workload_stats(self.db_id)
        if stats["total_queries"] == 0:
            return np.zeros((1, CONTEXT_DIM), dtype=np.float32)

        avg_latency = stats["avg_latency_ms"]
        log_total = np.log1p(stats["total_queries"]) / 10.0
        log_latency = np.log1p(avg_latency) / 10.0 if math.isfinite(avg_latency) else 0.0
        rw_ratio = stats["rw_ratio"]
        rw_normalized = min(rw_ratio, 10.0) / 10.0 if rw_ratio != float("inf") else 1.0

        self._context_buffer[0] = [log_total, log_latency, rw_normalized]
        return self._context_buffer

    def _infer_table_name(self, features: QueryFeatures) -> str:
        return features.tables[0] if features.tables else "unknown"

    def _table_index_count(self, table_name: str) -> int:
        try:
            with self.repo.get_session() as session:
                table = session.query(DBTable).filter(DBTable.table_name == table_name).first()
                if not table:
                    return 0
                return session.query(DBIndex).filter(
                    DBIndex.table_id == table.table_id, DBIndex.is_active == True).count()
        except Exception:
            return 0

    def _index_state(self, features: QueryFeatures) -> np.ndarray:
        count = self._table_index_count(self._infer_table_name(features))
        covered = bool(features.predicate_columns) and count > 0
        return np.array([
            count / max(1, self.max_indexes),
            1.0 if covered else 0.0,
            max(0.0, (self.max_indexes - count) / max(1, self.max_indexes)),
        ], dtype=np.float32)

    def _valid_actions(self, features: QueryFeatures) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        mask[ACTION_KEEP] = True
        if features.predicate_columns:
            count = self._table_index_count(self._infer_table_name(features))
            mask[ACTION_CREATE] = count < self.max_indexes
            mask[ACTION_DROP] = count > 0
        return mask

    @staticmethod
    def _is_read_query(query: str) -> bool:
        head = query.strip().lstrip("(").upper()
        return head.startswith("SELECT") or head.startswith("WITH")

    @staticmethod
    def _confidence_bucket(confidence: float) -> str:
        if confidence >= 0.9:
            return "high"
        if confidence >= 0.7:
            return "medium"
        return "low"

    # ------------------------------------------------------------------ state
    def _observe(self, query: str):
        features = self.parser.parse(query)
        x_features = self.feature_encoder.encode(features)
        embedding = self.embedder.embed(query)
        context = self._get_context_features()[0]
        index_state = self._index_state(features)
        state = build_state(x_features, embedding, context, index_state)
        return features, embedding, (x_features, embedding, context, index_state), state

    # --------------------------------------------------------------- learning
    def _record_learning(self, embedding: np.ndarray):
        self.metrics.replay_buffer_size.set(len(self.replay_buffer))
        self.online_scheduler.increment_query_count()

        if self.drift_detector.add_experience(embedding):
            phase_id = self.drift_detector.get_current_phase()
            self.logger.info(f"Workload drift detected. Consolidating policy as {phase_id}.")
            if len(self._recent_states) >= 32:
                states = torch.tensor(np.stack(list(self._recent_states)), dtype=torch.float32)
                self.agent.consolidate(phase_id, states)
            self.online_scheduler.trigger_drift()

    # ------------------------------------------------------------------ acting
    def process_query(self, query: str, log_to_db: bool = True) -> dict:
        start_time = datetime.now()
        result = {"query": query, "action": "KEEP", "target_columns": [],
                  "table_name": None, "confidence": 0.0, "executed": False,
                  "error": None, "latency_ms": 0}

        is_read = self._is_read_query(query)
        query_id = 0
        try:
            features, embedding, state_tuple, state_vec = self._observe(query)
            result["table_name"] = self._infer_table_name(features)
            result["similar_outcomes"] = self.vector_store.find_similar_queries(embedding, k=3)

            # Finalize the previous transition now that its successor is known.
            if self._last_transition is not None:
                lt = self._last_transition
                self.replay_buffer.add_transition(
                    lt["hash"], lt["action"], lt["reward"], False,
                    state=lt["state"], next_state=state_tuple)
                self._recent_states.append(state_vec)
                self._last_transition = None

            mask = self._valid_actions(features)
            action_idx = self.agent.select_action(state_vec, valid_actions=mask, explore=True)
            _, confidence = self.agent.predict(state_vec, valid_actions=mask)

            decision = self.decision_engine.formulate_action(
                action_idx, confidence, features, result["table_name"])
            result["action"] = decision["action"]
            result["target_columns"] = decision["target_columns"]
            result["confidence"] = confidence

            self.metrics.model_predictions.inc(
                action=decision["action"],
                confidence_bucket=self._confidence_bucket(confidence))
            self.metrics.model_confidence.observe(confidence, action=decision["action"])

            latency_before = 0.0
            if is_read:
                latency_before = self.performance_collector.measure_latency(query, runs=1)
                if not math.isfinite(latency_before):
                    latency_before = 0.0

            if log_to_db:
                query_id = self._log_query(query, latency_before if is_read else None)

            reward = 0.0
            if decision["action"] == "KEEP":
                self.metrics.decisions_made.inc(action="KEEP", table=result["table_name"])
            else:
                safe = self.safety_layer.evaluate_decision(decision, query)
                if not safe:
                    result["action"] = "KEEP"
                    self.metrics.decisions_rejected.inc(reason="safety_check")
                else:
                    executed = self.executor.execute_decision(decision)
                    result["executed"] = executed
                    if executed:
                        self.metrics.decisions_executed.inc(action=decision["action"])
                        decision_id = self._persist_decision(query, decision, features) if log_to_db else 0

                        latency_after = 0.0
                        if is_read:
                            latency_after = self.performance_collector.measure_latency(query, runs=1)
                            if not math.isfinite(latency_after):
                                latency_after = 0.0
                            reward = self.reward_calculator.compute_reward(
                                decision["action"], latency_before, latency_after)

                        if query_id and decision_id:
                            try:
                                self.feedback_loop.record_outcome(
                                    decision_id, query_id, embedding, decision["action"],
                                    latency_before, latency_after)
                            except Exception as e:
                                self.logger.warning(f"Feedback loop store failed: {e}")
                    else:
                        self.metrics.decisions_rejected.inc(reason="execution_failed")

            effective_action = ACTION_KEEP if result["action"] == "KEEP" else action_idx
            self._last_transition = {
                "hash": hashlib.md5(query.encode()).hexdigest(),
                "action": effective_action,
                "reward": reward,
                "state": state_tuple,
            }
            self._record_learning(embedding)

        except Exception as e:
            self.logger.exception(f"Pipeline error on query '{query[:50]}...': {e}")
            result["error"] = str(e)
            self.metrics.query_errors.inc(error_type=type(e).__name__)

        result["latency_ms"] = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics.query_latency.observe(result["latency_ms"] / 1000)
        self.metrics.queries_processed.inc(
            phase=f"task_{self._current_task_id}", action=result["action"])
        return result

    def process_batch(self, queries: List[str]) -> List[dict]:
        return [self.process_query(query) for query in queries]

    # -------------------------------------------------------------- persisting
    def _log_query(self, query: str, execution_time_ms: Optional[float]) -> int:
        try:
            with self.repo.get_session() as session:
                log = QueryLog(db_id=self.db_id, query_text=query,
                               execution_time_ms=execution_time_ms)
                session.add(log)
                session.commit()
                return log.query_id
        except Exception as e:
            self.logger.error(f"Failed to log query: {e}")
            return 0

    def _persist_decision(self, query: str, decision: Dict, features: QueryFeatures) -> int:
        try:
            with self.repo.get_session() as session:
                table = session.query(DBTable).filter(
                    DBTable.db_id == self.db_id,
                    DBTable.table_name == decision["table_name"],
                ).first()
                if not table:
                    return 0
                opt_dec = OptimizationDecision(
                    table_id=table.table_id, action=decision["action"],
                    confidence_score=decision["confidence"], status="PENDING")
                session.add(opt_dec)
                session.commit()
                return opt_dec.decision_id
        except Exception as e:
            self.logger.error(f"Failed to persist decision: {e}")
            return 0

    # ------------------------------------------------- supervised warm-starting
    def train_on_queries(self, queries: List[str], labels: List[int], task_id: int = None):
        if task_id is None:
            task_id = self._current_task_id
        else:
            self._current_task_id = task_id

        x_features_list, x_emb_list, x_ctx_list = [], [], []
        for query in queries:
            features = self.parser.parse(query)
            x_features_list.append(self.feature_encoder.encode(features))
            x_emb_list.append(self.embedder.embed(query))
            x_ctx_list.append(self._get_context_features()[0])

        history = self.cl_engine.train_with_dataloader(
            torch.tensor(np.array(x_features_list), dtype=torch.float32),
            torch.tensor(np.array(x_emb_list), dtype=torch.float32),
            torch.tensor(np.array(x_ctx_list), dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            batch_size=self.config.ml.batch_size,
            epochs=self.config.ml.epochs,
        )
        if history["accuracy"]:
            self.metrics.model_accuracy.set(
                history["accuracy"][-1], phase=f"task_{task_id}", task_id=str(task_id))
        return history

    def detect_and_adapt(self):
        if self.workload_monitor.detect_workload_shift(self.db_id):
            self._current_task_id += 1
            self.logger.info(f"Workload shift detected. New task_id: {self._current_task_id}")
            return True
        return False

    # ----------------------------------------------------------------- reports
    def get_health_status(self) -> Dict:
        return {"status": "healthy", "task_id": self._current_task_id, "db_id": self.db_id}

    def get_metrics_summary(self) -> Dict:
        cache_stats = self.cl_engine.get_cache_stats() if hasattr(self.cl_engine, "get_cache_stats") else {}
        safety_stats = self.safety_layer.get_statistics() if hasattr(self.safety_layer, "get_statistics") else {}
        return {
            "cache": cache_stats,
            "safety": safety_stats,
            "current_task": self._current_task_id,
            "replay_size": len(self.replay_buffer),
            "epsilon": getattr(self.agent, "epsilon", None),
        }


# Backwards-compatible alias: there is one pipeline implementation.
ProductionPipeline = CLDBPipeline

__all__ = ["CLDBPipeline", "ProductionPipeline"]
