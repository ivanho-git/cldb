import logging
import numpy as np
import torch
from typing import Optional

from cldb.parser import SQLParser, QueryFeatures
from cldb.features import FeatureEncoder
from cldb.embeddings import QueryEmbedder
from cldb.cl_engine import CLEngine, EMBEDDING_DIM, CONTEXT_DIM
from cldb.decision_engine import DecisionEngine
from cldb.workload_monitor import WorkloadMonitor
from cldb.safety_layer import SafetyLayer
from cldb.executor import Executor
from cldb.performance_collector import PerformanceCollector
from cldb.vector_store import VectorStore
from cldb.metadata_repo import MetadataRepo, QueryLog, OptimizationDecision, DBTable, DBColumn
from cldb.feedback import FeedbackLoop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLDBPipeline:
    def __init__(
        self,
        repo: MetadataRepo,
        vector_store: VectorStore,
        feature_encoder: Optional[FeatureEncoder] = None,
        embedder: Optional[QueryEmbedder] = None,
        cl_engine: Optional[CLEngine] = None,
        confidence_threshold: float = 0.6,
        db_id: int = 1,
    ):
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
        self.safety_layer = SafetyLayer(repo, confidence_threshold=confidence_threshold)
        self.executor = Executor(repo)
        self.performance_collector = PerformanceCollector(repo)
        self.feedback_loop = FeedbackLoop(repo, vector_store)

        self._context_buffer = np.zeros((1, CONTEXT_DIM), dtype=np.float32)
        self._current_task_id = 0

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

    def process_query(self, query: str, log_to_db: bool = True) -> dict:
        result = {
            "query": query,
            "action": "KEEP",
            "target_columns": [],
            "table_name": None,
            "confidence": 0.0,
            "executed": False,
            "error": None,
        }

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

            decision = self.decision_engine.formulate_action(
                action_idx, confidence, features, result["table_name"]
            )
            decision["confidence"] = confidence
            decision["table_name"] = result["table_name"]

            result["action"] = decision["action"]
            result["target_columns"] = decision["target_columns"]
            result["confidence"] = confidence

            if decision["action"] == "KEEP":
                result["executed"] = False
                return result

            safe = self.safety_layer.evaluate_decision(decision, query)
            if not safe:
                result["action"] = "KEEP"
                result["executed"] = False
                return result

            executed = self.executor.execute_decision(decision)
            result["executed"] = executed

            if executed and log_to_db:
                self._persist_decision(query, decision, features)

        except Exception as e:
            logger.error(f"Pipeline error on query '{query[:50]}...': {e}")
            result["error"] = str(e)

        return result

    def _persist_decision(self, query: str, decision: dict, features: QueryFeatures):
        with self.repo.get_session() as session:
            log = QueryLog(
                db_id=self.db_id,
                query_text=query,
                execution_time_ms=None,
            )
            session.add(log)
            session.commit()

            table = session.query(DBTable).filter(
                DBTable.db_id == self.db_id,
                DBTable.table_name == decision["table_name"],
            ).first()

            if not table:
                return

            opt_dec = OptimizationDecision(
                table_id=table.table_id,
                action=decision["action"],
                confidence_score=decision["confidence"],
                status="PENDING",
            )
            session.add(opt_dec)
            session.commit()

    def train_on_queries(
        self, queries: list[str], labels: list[int], task_id: int = None
    ):
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

        self.cl_engine.train_on_batch(x_features_t, x_emb_t, x_ctx_t, y_t, task_id=task_id)

    def detect_and_adapt(self):
        if self.workload_monitor.detect_workload_shift(self.db_id):
            self._current_task_id += 1
            logger.info(f"Workload shift detected. New task_id: {self._current_task_id}")
            return True
        return False