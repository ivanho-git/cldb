from cldb.metadata_repo import MetadataRepo, OptimizationDecision, ReplayBufferMetadata
from cldb.vector_store import VectorStore
import numpy as np


class FeedbackLoop:
    def __init__(self, repo: MetadataRepo, vector_store: VectorStore):
        self.repo = repo
        self.vector_store = vector_store

    def record_outcome(
        self,
        decision_id: int,
        query_id: int,
        embedding: np.ndarray,
        action: str,
        latency_before: float,
        latency_after: float,
    ):
        improvement_ms = latency_before - latency_after

        with self.repo.get_session() as session:
            decision = session.query(OptimizationDecision).get(decision_id)
            if decision:
                decision.status = "EXECUTED"
                session.commit()

            self.vector_store.store_experience(query_id, embedding, action, float(improvement_ms))

            if abs(improvement_ms) > 10.0:
                replay_entry = ReplayBufferMetadata(
                    query_id=query_id,
                    importance_score=float(abs(improvement_ms)),
                )
                session.add(replay_entry)
                session.commit()

        return improvement_ms