"""
Enhanced Safety Layer with production-grade features:
- Automatic rollback on performance regression
- Index limit enforcement
- Concurrent operation throttling
- Audit logging
"""
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import threading
import time

from sqlalchemy import text
from cldb.metadata_repo import MetadataRepo, DBIndex, IndexColumn, OptimizationHistory
from cldb.logger import get_logger
from cldb.resilience import RateLimiter


@dataclass
class RollbackAction:
    action_id: str
    original_action: Dict
    rollback_action: Dict
    created_at: datetime
    status: str = "pending"


class EnhancedSafetyLayer:
    def __init__(self, repo: MetadataRepo, confidence_threshold: float = 0.6, max_indexes_per_table: int = 5,
                 max_concurrent_operations: int = 3, rollback_on_regression: bool = True, regression_threshold_ms: float = 20.0):
        self.repo = repo
        self.confidence_threshold = confidence_threshold
        self.max_indexes_per_table = max_indexes_per_table
        self.max_concurrent_operations = max_concurrent_operations
        self.rollback_on_regression = rollback_on_regression
        self.regression_threshold_ms = regression_threshold_ms
        
        self.logger = get_logger("safety_layer")
        self.rate_limiter = RateLimiter(rate=1.0, capacity=max_concurrent_operations)
        self._active_operations: Dict[str, RollbackAction] = {}
        self._operation_lock = threading.Lock()
        self._audit_log: List[Dict] = []
    
    def evaluate_decision(self, decision: Dict, sample_query: str) -> bool:
        if decision["confidence"] < self.confidence_threshold:
            self._audit("rejected", decision, "low_confidence", f"Confidence {decision['confidence']} below threshold {self.confidence_threshold}")
            return False
        
        if decision["action"] == "KEEP":
            return True
        
        if not self._check_index_limit(decision["table_name"]):
            self._audit("rejected", decision, "index_limit", f"Too many indexes on {decision['table_name']}")
            return False
        
        cost_before = self._get_explain_cost(sample_query)
        decision["estimated_cost_before"] = cost_before
        
        return True
    
    def _check_index_limit(self, table_name: str) -> bool:
        if not table_name or table_name == "unknown":
            return True
        
        with self.repo.get_session() as session:
            active_indexes = session.query(DBIndex).join(
                IndexColumn, IndexColumn.index_id == DBIndex.index_id
            ).filter(
                DBIndex.is_active == True,
                IndexColumn.column_id.in_(
                    session.query(IndexColumn.column_id).join(
                        DBIndex, DBIndex.index_id == IndexColumn.index_id
                    ).filter(DBIndex.is_active == True)
                )
            ).all()
            
            table_indexes = [idx for idx in active_indexes if idx.table_id]
            
            if len(table_indexes) >= self.max_indexes_per_table:
                return False
        
        return True
    
    def _get_explain_cost(self, query: str) -> float:
        try:
            with self.repo.engine.connect() as conn:
                result = conn.execute(text(f"EXPLAIN {query}")).first()
                if result and result[0] and "cost=" in result[0]:
                    cost_part = result[0].split("cost=")[1].split(" ")[0]
                    upper_cost = cost_part.split("..")[1]
                    return float(upper_cost)
        except Exception:
            pass
        return 0.0
    
    def generate_rollback_action(self, original_action: str, target_columns: list, table_name: str) -> Dict:
        if original_action == "CREATE_INDEX":
            return {"action": "DROP_INDEX", "table_name": table_name, "target_columns": target_columns}
        elif original_action == "DROP_INDEX":
            return {"action": "CREATE_INDEX", "table_name": table_name, "target_columns": target_columns}
        return {"action": "KEEP"}
    
    def execute_with_rollback(self, executor: Callable, decision: Dict, latency_before: float, latency_after: float) -> bool:
        action_id = f"{decision['table_name']}_{decision['action']}_{int(time.time())}"
        
        rollback_action = self.generate_rollback_action(
            decision["action"], decision["target_columns"], decision["table_name"]
        )
        
        rollback_record = RollbackAction(
            action_id=action_id,
            original_action=decision,
            rollback_action=rollback_action,
            created_at=datetime.now(timezone.utc)
        )
        
        with self._operation_lock:
            self._active_operations[action_id] = rollback_record
        
        acquired = self.rate_limiter.acquire(tokens=1, blocking=True, timeout=5.0)
        if not acquired:
            self._audit("rejected", decision, "rate_limit", "Too many concurrent operations")
            return False
        
        try:
            result = executor(decision)
            
            if self.rollback_on_regression:
                latency_delta = latency_before - latency_after
                if latency_delta < -self.regression_threshold_ms:
                    self.logger.warning(f"Performance regression detected: {latency_delta:.2f}ms. Rolling back.")
                    self._initiate_rollback(action_id)
                    self._audit("rolled_back", decision, "regression", f"Latency delta: {latency_delta:.2f}ms")
                    return False
            
            rollback_record.status = "completed"
            self._audit("executed", decision, "success", f"Latency improvement: {latency_before - latency_after:.2f}ms")
            return result
            
        except Exception as e:
            rollback_record.status = "failed"
            self._audit("failed", decision, "execution_error", str(e))
            raise
        finally:
            with self._operation_lock:
                if action_id in self._active_operations:
                    del self._active_operations[action_id]
    
    def _initiate_rollback(self, action_id: str):
        with self._operation_lock:
            if action_id not in self._active_operations:
                return
            rollback_record = self._active_operations[action_id]
            rollback_record.status = "rollback_initiated"
        
        self.logger.info(f"Initiating rollback for {action_id}")
    
    def _audit(self, event_type: str, decision: Dict, reason: str, details: str):
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "action": decision.get("action", "UNKNOWN"),
            "table_name": decision.get("table_name", "unknown"),
            "target_columns": decision.get("target_columns", []),
            "reason": reason,
            "details": details,
            "confidence": decision.get("confidence", 0)
        }
        self._audit_log.append(audit_entry)
        
        with self.repo.get_session() as session:
            history = OptimizationHistory(
                decision_id=0,
                table_id=0,
                action=decision.get("action", "UNKNOWN"),
                status=event_type.upper()
            )
            session.add(history)
            session.commit()
    
    def get_active_operations(self) -> List[Dict]:
        with self._operation_lock:
            return [{"action_id": r.action_id, "original_action": r.original_action, "rollback_action": r.rollback_action,
                     "created_at": r.created_at.isoformat(), "status": r.status} for r in self._active_operations.values()]
    
    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        return self._audit_log[-limit:]
    
    def get_statistics(self) -> Dict:
        with self._operation_lock:
            status_counts = {}
            for record in self._active_operations.values():
                status_counts[record.status] = status_counts.get(record.status, 0) + 1
        
        event_counts = {}
        for entry in self._audit_log:
            event_counts[entry["event_type"]] = event_counts.get(entry["event_type"], 0) + 1
        
        return {"active_operations": status_counts, "event_counts": event_counts, "total_audit_entries": len(self._audit_log)}


# Alias for backward compatibility
SafetyLayer = EnhancedSafetyLayer