"""
Production-grade pipeline for CLDB with resilience and observability.
"""
import logging
import numpy as np
import torch
from typing import Optional, List, Dict
from datetime import datetime, timezone

from cldb.parser import SQLParser, QueryFeatures
from cldb.features import FeatureEncoder
from cldb.embeddings import QueryEmbedder
from cldb.enhanced_cl_engine import CLEngine, EMBEDDING_DIM, CONTEXT_DIM
from cldb.decision_engine import DecisionEngine
from cldb.workload_monitor import WorkloadMonitor
from cldb.enhanced_safety import SafetyLayer
from cldb.executor import Executor
from cldb.performance_collector import PerformanceCollector
from cldb.enhanced_vector_store import VectorStore
from cldb.metadata_repo import MetadataRepo, QueryLog, OptimizationDecision, DBTable, DBColumn
from cldb.feedback import FeedbackLoop
from cldb.config import get_config
from cldb.logger import get_logger
from cldb.metrics import get_metrics
from cldb.resilience import get_database_breaker, get_embedding_breaker
from cldb.replay_buffer import DatabaseAwarePriorityReplay
from cldb.reward import RewardCalculator
from cldb.drift_detection import DriftDetector
from cldb.online_scheduler import OnlineScheduler
import hashlib


class ProductionPipeline:
    def __init__(self, repo: MetadataRepo, vector_store: VectorStore, feature_encoder: Optional[FeatureEncoder] = None,
                 embedder: Optional[QueryEmbedder] = None, cl_engine: Optional[CLEngine] = None,
                 confidence_threshold: float = 0.6, db_id: int = 1):
        self.config = get_config()
        self.logger = get_logger("production_pipeline")
        self.metrics = get_metrics()
        
        self.repo = repo
        self.vector_store = vector_store
        self.db_id = db_id

        self.parser = SQLParser()
        self.feature_encoder = feature_encoder or FeatureEncoder()
        self.embedder = embedder or QueryEmbedder()
        feature_dim = self.feature_encoder.get_feature_dim()

        self.cl_engine = cl_engine or CLEngine(feature_dim=feature_dim)
        self.decision_engine = DecisionEngine()

        self.workload_monitor = WorkloadMonitor(repo)
        self.safety_layer = SafetyLayer(repo, confidence_threshold=confidence_threshold,
                                        max_indexes_per_table=self.config.safety.max_indexes_per_table,
                                        rollback_on_regression=self.config.safety.rollback_on_regression,
                                        regression_threshold_ms=self.config.safety.regression_threshold_ms)
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
            beta=self.config.replay.beta
        )
        self.reward_calculator = RewardCalculator(
            latency_weight=self.config.reward.latency_weight,
            regression_penalty_weight=self.config.reward.regression_penalty_weight,
            resource_penalty=self.config.reward.resource_penalty
        )
        self.drift_detector = DriftDetector(
            window_size=self.config.drift.window_size,
            drift_threshold=self.config.drift.drift_threshold
        )
        self.online_scheduler = OnlineScheduler(
            train_callback=self._execute_online_training,
            drift_detector=self.drift_detector,
            check_interval_seconds=self.config.drift.check_interval_seconds,
            min_queries_before_train=self.config.drift.min_queries_before_train
        )
        self.online_scheduler.start()
        
        self.logger.info("Production pipeline initialized", db_id=db_id, confidence_threshold=confidence_threshold)
        
    def _execute_online_training(self):
        import time
        start = time.time()
        try:
            total_loss, ewc_penalty = self.cl_engine.train_on_experience(self.replay_buffer, batch_size=self.config.replay.batch_size)
            self.metrics.ewc_penalty.set(ewc_penalty)
            duration = time.time() - start
            self.metrics.training_time.observe(duration)
        except Exception as e:
            self.logger.error(f"Online training failed: {e}")
    
    def _get_context_features(self) -> np.ndarray:
        stats = self.workload_monitor.get_workload_stats(self.db_id)
        if stats["total_queries"] == 0:
            return np.zeros((1, CONTEXT_DIM), dtype=np.float32)

        log_total = np.log1p(stats["total_queries"]) / 10.0
        log_latency = np.log1p(stats["avg_latency_ms"]) / 10.0
        rw_normalized = min(stats["rw_ratio"], 10.0) / 10.0 if stats["rw_ratio"] != float("inf") else 1.0

        self._context_buffer[0] = [log_total, log_latency, rw_normalized]
        return self._context_buffer
    
    def _infer_table_name(self, features: QueryFeatures) -> str:
        if features.tables:
            return features.tables[0]
        return "unknown"
    
    def _is_read_query(self, query: str) -> bool:
        head = query.strip().lstrip("(").upper()
        return head.startswith("SELECT") or head.startswith("WITH")
    
    def _log_query(self, query: str, execution_time_ms: Optional[float]) -> int:
        try:
            with self.repo.get_session() as session:
                log = QueryLog(db_id=self.db_id, query_text=query, execution_time_ms=execution_time_ms)
                session.add(log)
                session.commit()
                return log.query_id
        except Exception as e:
            self.logger.error(f"Failed to log query: {e}")
            return 0
    
    def _record_learning(self, query: str, x_features: np.ndarray, embedding: np.ndarray,
                         action_idx: int, reward: float):
        query_hash = hashlib.md5(query.encode()).hexdigest()
        self.replay_buffer.add(
            query_hash=query_hash,
            features=x_features,
            embedding=embedding,
            context=self._context_buffer[0].copy(),
            action=action_idx,
            reward=reward
        )
        self.metrics.replay_buffer_size.set(len(self.replay_buffer))
        self.online_scheduler.increment_query_count()
        
        is_drift = self.drift_detector.add_experience(embedding)
        if is_drift:
            phase_id = self.drift_detector.get_current_phase()
            self.logger.info(f"Workload Drift detected! Transitioning to {phase_id}")
            if len(self.replay_buffer) > 0:
                feat, emb, ctx, actions, _, _ = self.replay_buffer.sample(min(32, len(self.replay_buffer)))
                self.cl_engine.ewc.register_workload_phase(phase_id, feat, emb, ctx, actions)
            self.online_scheduler.trigger_drift()
    
    def process_query(self, query: str, log_to_db: bool = True) -> dict:
        start_time = datetime.now()
        result = {"query": query, "action": "KEEP", "target_columns": [], "table_name": None,
                  "confidence": 0.0, "executed": False, "error": None, "latency_ms": 0}
        
        is_read = self._is_read_query(query)
        query_id = 0
        try:
            features = self.parser.parse(query)
            result["table_name"] = self._infer_table_name(features)
            
            x_features = self.feature_encoder.encode(features)
            x_features_t = torch.tensor(x_features, dtype=torch.float32).unsqueeze(0)
            
            embedding = self.embedder.embed(query)
            x_emb_t = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            
            x_ctx_t = torch.tensor(self._get_context_features(), dtype=torch.float32)
            
            similar = self.vector_store.find_similar_queries(embedding, k=3)
            result["similar_outcomes"] = similar
            
            action_idx, confidence = self.cl_engine.predict(x_features_t, x_emb_t, x_ctx_t)
            action_idx = int(action_idx[0])
            confidence = float(confidence[0])
            
            decision = self.decision_engine.formulate_action(action_idx, confidence, features, result["table_name"])
            decision["confidence"] = confidence
            decision["table_name"] = result["table_name"]
            
            result["action"] = decision["action"]
            result["target_columns"] = decision["target_columns"]
            result["confidence"] = confidence
            
            self.metrics.model_predictions.inc(action=decision["action"], confidence_bucket=self._confidence_bucket(confidence))
            self.metrics.model_confidence.observe(confidence, action=decision["action"])
            
            latency_before = 0.0
            if is_read:
                latency_before = self.performance_collector.measure_latency(query, runs=1)
            
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
                        decision_id = 0
                        if log_to_db:
                            decision_id = self._persist_decision(query, decision, features)
                        
                        latency_after = 0.0
                        if is_read:
                            latency_after = self.performance_collector.measure_latency(query, runs=1)
                            reward = self.reward_calculator.compute_reward(decision["action"], latency_before, latency_after)
                        
                        if query_id and decision_id:
                            try:
                                self.feedback_loop.record_outcome(
                                    decision_id, query_id, embedding, decision["action"],
                                    latency_before, latency_after)
                            except Exception as e:
                                self.logger.warning(f"Feedback loop store failed: {e}")
                    else:
                        self.metrics.decisions_rejected.inc(reason="execution_failed")
            
            effective_action_idx = 0 if result["action"] == "KEEP" else action_idx
            self._record_learning(query, x_features, embedding, effective_action_idx, reward)
            
        except Exception as e:
            self.logger.exception(f"Pipeline error on query '{query[:50]}...': {e}")
            result["error"] = str(e)
            self.metrics.query_errors.inc(error_type=type(e).__name__)
        
        result["latency_ms"] = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics.query_latency.observe(result["latency_ms"] / 1000)
        self.metrics.queries_processed.inc(phase=f"task_{self._current_task_id}", action=result["action"])
        
        return result
    
    def _confidence_bucket(self, confidence: float) -> str:
        if confidence >= 0.9:
            return "high"
        elif confidence >= 0.7:
            return "medium"
        return "low"
    
    def process_batch(self, queries: List[str]) -> List[dict]:
        results = []
        for query in queries:
            result = self.process_query(query)
            results.append(result)
        return results
    
    def _persist_decision(self, query: str, decision: Dict, features: QueryFeatures) -> int:
        try:
            with self.repo.get_session() as session:
                table = session.query(DBTable).filter(
                    DBTable.db_id == self.db_id,
                    DBTable.table_name == decision["table_name"],
                ).first()

                if not table:
                    return 0

                opt_dec = OptimizationDecision(table_id=table.table_id, action=decision["action"],
                                               confidence_score=decision["confidence"], status="PENDING")
                session.add(opt_dec)
                session.commit()
                return opt_dec.decision_id
        except Exception as e:
            self.logger.error(f"Failed to persist decision: {e}")
            return 0
    
    def train_on_queries(self, queries: List[str], labels: List[int], task_id: int = None):
        if task_id is None:
            task_id = self._current_task_id
        else:
            self._current_task_id = task_id

        x_features_list = []
        x_emb_list = []
        x_ctx_list = []

        for query in queries:
            features = self.parser.parse(query)
            x_feat = self.feature_encoder.encode(features)
            x_emb = self.embedder.embed(query)
            x_ctx = self._get_context_features()

            x_features_list.append(x_feat)
            x_emb_list.append(x_emb)
            x_ctx_list.append(x_ctx[0])

        x_features_t = torch.tensor(np.array(x_features_list), dtype=torch.float32)
        x_emb_t = torch.tensor(np.array(x_emb_list), dtype=torch.float32)
        x_ctx_t = torch.tensor(np.array(x_ctx_list), dtype=torch.float32)
        y_t = torch.tensor(labels, dtype=torch.long)

        history = self.cl_engine.train_with_dataloader(x_features_t, x_emb_t, x_ctx_t, y_t,
                                                       batch_size=self.config.ml.batch_size, epochs=self.config.ml.epochs)
        
        self.metrics.model_accuracy.set(history["accuracy"][-1] if history["accuracy"] else 0, phase=f"task_{task_id}", task_id=str(task_id))
        
        return history
    
    def detect_and_adapt(self):
        if self.workload_monitor.detect_workload_shift(self.db_id):
            self._current_task_id += 1
            self.logger.info(f"Workload shift detected. New task_id: {self._current_task_id}")
            return True
        return False
    
    def get_health_status(self) -> Dict:
        return {"status": "healthy", "task_id": self._current_task_id, "db_id": self.db_id}
    
    def get_metrics_summary(self) -> Dict:
        cache_stats = self.cl_engine.get_cache_stats() if hasattr(self.cl_engine, 'get_cache_stats') else {}
        safety_stats = self.safety_layer.get_statistics() if hasattr(self.safety_layer, 'get_statistics') else {}
        
        return {"cache": cache_stats, "safety": safety_stats, "current_task": self._current_task_id}