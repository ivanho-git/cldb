"""
Enhanced Vector Store with production-grade reliability.
"""
import os
import uuid
import time
from typing import List, Dict, Optional, Union
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from cldb.config import get_config
from cldb.logger import get_logger
from cldb.resilience import retry_with_backoff, get_vector_store_breaker


class VectorStoreConfig:
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "cldb_queries",
                 vector_size: int = 384, distance_metric: str = "Cosine", timeout: int = 10, retry_attempts: int = 3):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance_metric = distance_metric
        self.timeout = timeout
        self.retry_attempts = retry_attempts


class EnhancedVectorStore:
    def __init__(self, collection_name: str = "cldb_queries", config: Optional[VectorStoreConfig] = None):
        self.config = config or self._load_config()
        self.logger = get_logger("vector_store")
        self.collection_name = collection_name
        self.vector_size = self.config.vector_size
        self._init_client()
        self._ensure_collection()
        self._memory_store: Dict[str, Dict] = {}
        self._memory_vectors: Dict[str, np.ndarray] = {}
        self.circuit_breaker = get_vector_store_breaker()
    
    def _load_config(self) -> VectorStoreConfig:
        try:
            app_config = get_config()
            return VectorStoreConfig(host=app_config.vector_store.host, port=app_config.vector_store.port,
                                     collection_name=app_config.vector_store.collection_name, vector_size=app_config.vector_store.vector_size,
                                     timeout=app_config.vector_store.timeout, retry_attempts=app_config.vector_store.retry_attempts)
        except Exception:
            return VectorStoreConfig()
    
    def _init_client(self):
        try:
            self.client = QdrantClient(host=self.config.host, port=self.config.port, timeout=self.config.timeout)
            self.client.get_collections()
            self._connected = True
            self.logger.info(f"Connected to Qdrant at {self.config.host}:{self.config.port}")
        except Exception as e:
            self.logger.warning(f"Could not connect to Qdrant: {e}. Using in-memory fallback.")
            self._connected = False
            self.client = None
    
    def _ensure_collection(self):
        if not self._connected:
            return
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            distance = Distance.COSINE if self.config.distance_metric == "Cosine" else Distance.EUCLID
            self.client.create_collection(collection_name=self.collection_name,
                                           vectors_config=VectorParams(size=self.config.vector_size, distance=distance))
            self.logger.info(f"Created collection: {self.collection_name}")
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def _to_list(self, embedding: Union[np.ndarray, List[float]]) -> List[float]:
        if hasattr(embedding, 'tolist'):
            return embedding.tolist()
        return list(embedding)
    
    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    def store_experience(self, query_id: int, embedding: np.ndarray, action: str, outcome: float, metadata: Optional[Dict] = None) -> bool:
        point_id = str(uuid.uuid4())
        payload = {"query_id": query_id, "action": action, "outcome": outcome, "timestamp": time.time()}
        if metadata:
            payload.update(metadata)
        point = PointStruct(id=point_id, vector=self._to_list(embedding), payload=payload)
        
        if self._connected and self.circuit_breaker.allow_request():
            try:
                self.client.upsert(collection_name=self.collection_name, points=[point])
                self.circuit_breaker.record_success()
                return True
            except Exception as e:
                self.circuit_breaker.record_failure()
                self.logger.error(f"Failed to store to Qdrant: {e}")
        
        self._memory_store[point_id] = payload
        self._memory_vectors[point_id] = embedding
        return True
    
    def store_batch(self, experiences: List[Dict]) -> int:
        if not experiences:
            return 0
        stored = 0
        for exp in experiences:
            if self.store_experience(exp["query_id"], exp["embedding"], exp["action"], exp["outcome"], exp.get("metadata")):
                stored += 1
        return stored
    
    @retry_with_backoff(max_attempts=3, base_delay=0.5)
    def find_similar_queries(self, embedding: np.ndarray, k: int = 5, action_filter: Optional[str] = None,
                             min_outcome: Optional[float] = None) -> List[Dict]:
        results = []
        
        if self._connected and self.circuit_breaker.allow_request():
            try:
                search_params = {"limit": k}
                if action_filter or min_outcome is not None:
                    must_conditions = []
                    if action_filter:
                        must_conditions.append(FieldCondition(key="action", match=MatchValue(value=action_filter)))
                    if min_outcome is not None:
                        must_conditions.append(FieldCondition(key="outcome", range={"gte": min_outcome}))
                    search_params["filter"] = Filter(must=must_conditions)
                
                hits = self.client.search(collection_name=self.collection_name, query_vector=self._to_list(embedding), **search_params)
                for hit in hits:
                    results.append({"query_id": hit.payload.get("query_id"), "action": hit.payload.get("action"),
                                    "outcome": hit.payload.get("outcome"), "score": hit.score, "timestamp": hit.payload.get("timestamp")})
                self.circuit_breaker.record_success()
                return results
            except Exception as e:
                self.circuit_breaker.record_failure()
                self.logger.error(f"Search failed: {e}")
        
        return self._memory_search(embedding, k)
    
    def _memory_search(self, embedding: np.ndarray, k: int) -> List[Dict]:
        if not self._memory_vectors:
            return []
        
        scores = []
        embedding = np.array(embedding)
        for point_id, stored_emb in self._memory_vectors.items():
            dot = np.dot(stored_emb, embedding)
            norm = np.linalg.norm(stored_emb) * np.linalg.norm(embedding)
            score = dot / norm if norm > 0 else 0
            scores.append((point_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for point_id, score in scores[:k]:
            payload = self._memory_store.get(point_id, {})
            results.append({"query_id": payload.get("query_id"), "action": payload.get("action"),
                            "outcome": payload.get("outcome"), "score": float(score), "timestamp": payload.get("timestamp")})
        return results
    
    def get_experience_count(self) -> int:
        if self._connected:
            try:
                collection = self.client.get_collection(self.collection_name)
                return collection.vectors_count
            except Exception:
                pass
        return len(self._memory_store)
    
    def delete_old_experiences(self, before_timestamp: float) -> int:
        deleted = 0
        if self._connected:
            try:
                self.client.delete(collection_name=self.collection_name,
                                   points_selector=Filter(must=[FieldCondition(key="timestamp", range={"lt": before_timestamp})]))
                deleted = -1
            except Exception as e:
                self.logger.error(f"Delete failed: {e}")
        
        keys_to_delete = [pid for pid, payload in self._memory_store.items() if payload.get("timestamp", 0) < before_timestamp]
        for key in keys_to_delete:
            del self._memory_store[key]
            if key in self._memory_vectors:
                del self._memory_vectors[key]
            deleted += 1
        return deleted
    
    def health_check(self) -> Dict:
        health = {"connected": self._connected, "collection": self.collection_name,
                  "experience_count": self.get_experience_count(), "memory_fallback_size": len(self._memory_store)}
        
        if self._connected:
            try:
                self.client.get_collections()
                health["status"] = "healthy"
            except Exception as e:
                health["status"] = "unhealthy"
                health["error"] = str(e)
        else:
            health["status"] = "using_fallback"
        
        return health
    
    def clear_collection(self) -> bool:
        try:
            if self._connected:
                self.client.delete(collection_name=self.collection_name, points_selector=Filter(must=[]))
            self._memory_store.clear()
            self._memory_vectors.clear()
            return True
        except Exception as e:
            self.logger.error(f"Clear failed: {e}")
            return False


VectorStore = EnhancedVectorStore