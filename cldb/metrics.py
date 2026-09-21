"""
Prometheus metrics for CLDB observability.
"""
from typing import Dict, Optional, List
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from functools import wraps


@dataclass
class MetricValue:
    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    def __init__(self, name: str, description: str = "", labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()
    
    def inc(self, value: float = 1, **label_values):
        with self._lock:
            key = self._make_key(label_values)
            self._values[key] += value
    
    def get(self, **label_values) -> float:
        key = self._make_key(label_values)
        return self._values.get(key, 0.0)
    
    def _make_key(self, label_values: Dict[str, str]) -> tuple:
        if self.labels:
            return tuple(label_values.get(l, "") for l in self.labels)
        return ()
    
    def collect(self) -> Dict:
        return {"name": self.name, "type": "counter", "values": dict(self._values)}


class Gauge:
    def __init__(self, name: str, description: str = "", labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()
    
    def set(self, value: float, **label_values):
        with self._lock:
            key = self._make_key(label_values)
            self._values[key] = value
    
    def inc(self, value: float = 1, **label_values):
        with self._lock:
            key = self._make_key(label_values)
            self._values[key] += value
    
    def dec(self, value: float = 1, **label_values):
        with self._lock:
            key = self._make_key(label_values)
            self._values[key] -= value
    
    def get(self, **label_values) -> float:
        key = self._make_key(label_values)
        return self._values.get(key, 0.0)
    
    def _make_key(self, label_values: Dict[str, str]) -> tuple:
        if self.labels:
            return tuple(label_values.get(l, "") for l in self.labels)
        return ()
    
    def collect(self) -> Dict:
        return {"name": self.name, "type": "gauge", "values": dict(self._values)}


class Histogram:
    BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
    
    def __init__(self, name: str, description: str = "", labels: Optional[List[str]] = None,
                 buckets: Optional[List[float]] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self.buckets = buckets or self.BUCKETS
        self._counts: Dict[tuple, Dict[float, int]] = defaultdict(lambda: defaultdict(int))
        self._sums: Dict[tuple, float] = defaultdict(float)
        self._totals: Dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def observe(self, value: float, **label_values):
        with self._lock:
            key = self._make_key(label_values)
            self._sums[key] += value
            self._totals[key] += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[key][bucket] += 1
    
    def _make_key(self, label_values: Dict[str, str]) -> tuple:
        if self.labels:
            return tuple(label_values.get(l, "") for l in self.labels)
        return ()
    
    def collect(self) -> Dict:
        return {"name": self.name, "type": "histogram", "buckets": self.buckets,
                "values": {k: dict(v) for k, v in self._counts.items()}}


class Summary:
    def __init__(self, name: str, description: str = "", labels: Optional[List[str]] = None,
                 quantiles: Optional[List[float]] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self.quantiles = quantiles or [0.5, 0.9, 0.95, 0.99]
        self._values: Dict[tuple, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def observe(self, value: float, **label_values):
        with self._lock:
            key = self._make_key(label_values)
            self._values[key].append(value)
    
    def _make_key(self, label_values: Dict[str, str]) -> tuple:
        if self.labels:
            return tuple(label_values.get(l, "") for l in self.labels)
        return ()
    
    def get_quantile(self, quantile: float, **label_values) -> Optional[float]:
        key = self._make_key(label_values)
        values = self._values.get(key, [])
        if not values:
            return None
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * quantile)
        return sorted_values[min(idx, len(sorted_values) - 1)]
    
    def collect(self) -> Dict:
        result = {}
        for key, values in self._values.items():
            sorted_values = sorted(values)
            result[key] = {q: sorted_values[min(int(len(sorted_values) * q), len(sorted_values) - 1)] for q in self.quantiles}
        return {"name": self.name, "type": "summary", "values": result}


class MetricsRegistry:
    _instance: Optional['MetricsRegistry'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._summaries: Dict[str, Summary] = {}
    
    @classmethod
    def get_instance(cls) -> 'MetricsRegistry':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def counter(self, name: str, description: str = "", labels: Optional[List[str]] = None) -> Counter:
        if name not in self._counters:
            self._counters[name] = Counter(name, description, labels)
        return self._counters[name]
    
    def gauge(self, name: str, description: str = "", labels: Optional[List[str]] = None) -> Gauge:
        if name not in self._gauges:
            self._gauges[name] = Gauge(name, description, labels)
        return self._gauges[name]
    
    def histogram(self, name: str, description: str = "", labels: Optional[List[str]] = None,
                  buckets: Optional[List[float]] = None) -> Histogram:
        if name not in self._histograms:
            self._histograms[name] = Histogram(name, description, labels, buckets)
        return self._histograms[name]
    
    def summary(self, name: str, description: str = "", labels: Optional[List[str]] = None,
                quantiles: Optional[List[float]] = None) -> Summary:
        if name not in self._summaries:
            self._summaries[name] = Summary(name, description, labels, quantiles)
        return self._summaries[name]
    
    def collect_all(self) -> Dict:
        return {
            "counters": {k: v.collect() for k, v in self._counters.items()},
            "gauges": {k: v.collect() for k, v in self._gauges.items()},
            "histograms": {k: v.collect() for k, v in self._histograms.items()},
            "summaries": {k: v.collect() for k, v in self._summaries.items()}
        }
    
    def reset_all(self):
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._summaries.clear()


class CLDBMetrics:
    def __init__(self):
        registry = MetricsRegistry.get_instance()
        self.queries_processed = registry.counter("cldb_queries_processed_total", "Total queries processed", ["phase", "action"])
        self.query_latency = registry.histogram("cldb_query_latency_seconds", "Query latency in seconds", ["phase"])
        self.query_errors = registry.counter("cldb_query_errors_total", "Query errors", ["error_type"])
        self.model_predictions = registry.counter("cldb_model_predictions_total", "Model predictions", ["action", "confidence_bucket"])
        self.model_confidence = registry.histogram("cldb_model_confidence", "Model confidence distribution", ["action"])
        self.decisions_made = registry.counter("cldb_decisions_total", "Optimization decisions", ["action", "table"])
        self.decisions_executed = registry.counter("cldb_decisions_executed_total", "Executed decisions", ["action"])
        self.decisions_rejected = registry.counter("cldb_decisions_rejected_total", "Rejected decisions", ["reason"])
        self.training_batches = registry.counter("cldb_training_batches_total", "Training batches")
        self.training_loss = registry.summary("cldb_training_loss", "Training loss", ["phase"])
        self.model_accuracy = registry.gauge("cldb_model_accuracy", "Model accuracy", ["phase", "task_id"])
        self.vector_store_operations = registry.counter("cldb_vector_operations_total", "Vector store ops", ["operation", "status"])
        self.vector_search_latency = registry.histogram("cldb_vector_search_latency_seconds", "Vector search latency")
        self.active_indexes = registry.gauge("cldb_active_indexes", "Active indexes", ["table"])
        self.workload_queries_total = registry.gauge("cldb_workload_queries", "Workload queries", ["db_id"])
        self.workload_rw_ratio = registry.gauge("cldb_workload_rw_ratio", "RW ratio", ["db_id"])
        self.replay_buffer_size = registry.gauge("cldb_replay_buffer_size", "Current size of replay buffer")
        self.training_time = registry.histogram("cldb_training_time_seconds", "Time spent in online training")
        self.ewc_penalty = registry.gauge("cldb_ewc_penalty_value", "Value of the EWC penalty loss")
        self.drift_score = registry.gauge("cldb_drift_score", "Current detected workload drift distance")
        self.retention_score = registry.gauge("cldb_retention_score", "Model retention score on previous workload phases")


_metrics_instance: Optional[CLDBMetrics] = None
_metrics_lock = threading.Lock()


def get_metrics() -> CLDBMetrics:
    global _metrics_instance
    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = CLDBMetrics()
    return _metrics_instance


def time_function(metric: Histogram):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                metric.observe(duration)
        return wrapper
    return decorator


def get_prometheus_metrics() -> str:
    registry = MetricsRegistry.get_instance()
    all_metrics = registry.collect_all()
    output = []
    for name, data in all_metrics["counters"].items():
        for key, value in data["values"].items():
            labels = ""
            if key:
                label_parts = [f'{k}="{v}"' for k, v in zip(data.get("labels", []), key)]
                if label_parts:
                    labels = "{" + ",".join(label_parts) + "}"
            output.append(f"# TYPE {name} counter")
            output.append(f"{name}{labels} {value}")
    for name, data in all_metrics["gauges"].items():
        for key, value in data["values"].items():
            labels = ""
            if key:
                label_parts = [f'{k}="{v}"' for k, v in zip(data.get("labels", []), key)]
                if label_parts:
                    labels = "{" + ",".join(label_parts) + "}"
            output.append(f"# TYPE {name} gauge")
            output.append(f"{name}{labels} {value}")
    return "\n".join(output)