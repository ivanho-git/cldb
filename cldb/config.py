"""
Production-grade configuration management for CLDB.
Supports environment variables, validation, and multi-environment configurations.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import os
import json
from functools import lru_cache


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "postgres"
    name: str = "cldb"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def sqlite_url(self) -> str:
        return "sqlite:///:memory:"


@dataclass
class VectorStoreConfig:
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "cldb_queries"
    vector_size: int = 384
    distance_metric: str = "Cosine"
    timeout: int = 10
    retry_attempts: int = 3


@dataclass
class MLConfig:
    feature_dim: int = 74
    embedding_dim: int = 384
    context_dim: int = 3
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 5
    early_stopping_patience: int = 5
    validation_split: float = 0.2
    gradient_clip_value: float = 1.0
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6


@dataclass
class EWCConfig:
    lambda_param: float = 0.4
    importance: float = 1000.0
    decay_factor: Optional[float] = None


@dataclass
class ReplayConfig:
    mem_size: int = 1000
    batch_size: int = 32
    alpha: float = 0.6
    beta: float = 0.4
    min_samples: int = 50


@dataclass
class DriftConfig:
    window_size: int = 100
    drift_threshold: float = 0.3
    check_interval_seconds: float = 5.0
    min_queries_before_train: int = 50


@dataclass
class RewardConfig:
    latency_weight: float = 1.0
    regression_penalty_weight: float = 2.0
    resource_penalty: float = 0.1


@dataclass
class SafetyConfig:
    confidence_threshold: float = 0.6
    max_indexes_per_table: int = 5
    max_concurrent_operations: int = 3
    cost_threshold: float = 100.0
    enable_explain_validation: bool = True
    rollback_on_regression: bool = True
    regression_threshold_ms: float = 20.0


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    cache_dir: Optional[str] = None
    device: str = "auto"
    normalize: bool = True


@dataclass
class ObservabilityConfig:
    log_level: str = "INFO"
    log_format: str = "json"
    metrics_enabled: bool = True
    metrics_port: int = 9090
    health_port: int = 8080
    tracing_enabled: bool = False
    tracing_sample_rate: float = 0.1


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 5000
    workers: int = 4
    timeout: int = 60
    max_request_size_mb: int = 10


@dataclass
class Config:
    environment: str = "development"
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    ewc: EWCConfig = field(default_factory=EWCConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    model_dir: str = "/app/models"
    experiment_dir: str = "/app/experiments"
    checkpoint_dir: str = "/app/models/checkpoints"

    def validate(self) -> List[str]:
        errors = []
        if self.ml.learning_rate <= 0 or self.ml.learning_rate > 1:
            errors.append("ML learning_rate must be between 0 and 1")
        if self.ml.dropout < 0 or self.ml.dropout > 1:
            errors.append("ML dropout must be between 0 and 1")
        if self.safety.confidence_threshold < 0 or self.safety.confidence_threshold > 1:
            errors.append("Safety confidence_threshold must be between 0 and 1")
        if self.replay.mem_size < 10:
            errors.append("Replay memory size must be at least 10")
        if self.database.port < 1 or self.database.port > 65535:
            errors.append("Database port must be between 1 and 65535")
        if self.vector_store.port < 1 or self.vector_store.port > 65535:
            errors.append("Vector store port must be between 1 and 65535")
        return errors


def _get_env(key: str, default: Any) -> Any:
    value = os.environ.get(key)
    if value is None:
        return default
    if isinstance(default, bool):
        return value.lower() in ("true", "1", "yes")
    elif isinstance(default, int):
        try:
            return int(value)
        except ValueError:
            return default
    elif isinstance(default, float):
        try:
            return float(value)
        except ValueError:
            return default
    return value


def _get_env_list(key: str, default: List, separator: str = ",") -> List:
    value = os.environ.get(key)
    if value is None:
        return default
    return [int(x.strip()) if x.strip().isdigit() else x.strip() for x in value.split(separator)]


class ConfigLoader:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
    
    def load(self) -> Config:
        config = Config()
        config.environment = _get_env("CLDB_ENVIRONMENT", config.environment)
        config.database.host = _get_env("POSTGRES_HOST", config.database.host)
        config.database.port = _get_env("POSTGRES_PORT", config.database.port)
        config.database.user = _get_env("POSTGRES_USER", config.database.user)
        config.database.password = _get_env("POSTGRES_PASSWORD", config.database.password)
        config.database.name = _get_env("POSTGRES_DB", config.database.name)
        config.database.pool_size = _get_env("POSTGRES_POOL_SIZE", config.database.pool_size)
        config.vector_store.host = _get_env("QDRANT_HOST", config.vector_store.host)
        config.vector_store.port = _get_env("QDRANT_PORT", config.vector_store.port)
        config.ml.learning_rate = _get_env("ML_LR", config.ml.learning_rate)
        config.ml.batch_size = _get_env("ML_BATCH_SIZE", config.ml.batch_size)
        config.ml.hidden_dims = _get_env_list("ML_HIDDEN_DIMS", config.ml.hidden_dims)
        config.ewc.lambda_param = _get_env("EWC_LAMBDA", config.ewc.lambda_param)
        config.replay.mem_size = _get_env("REPLAY_MEM_SIZE", config.replay.mem_size)
        config.safety.confidence_threshold = _get_env("SAFETY_CONFIDENCE_THRESHOLD", config.safety.confidence_threshold)
        config.safety.enable_explain_validation = _get_env("SAFETY_EXPLAIN_VALIDATION", config.safety.enable_explain_validation)
        config.embedding.model_name = _get_env("EMBEDDING_MODEL", config.embedding.model_name)
        config.observability.log_level = _get_env("LOG_LEVEL", config.observability.log_level)
        config.observability.metrics_enabled = _get_env("METRICS_ENABLED", config.observability.metrics_enabled)
        config.model_dir = _get_env("CLDB_MODEL_DIR", config.model_dir)
        config.experiment_dir = _get_env("CLDB_EXPERIMENT_DIR", config.experiment_dir)
        config.checkpoint_dir = _get_env("CLDB_CHECKPOINT_DIR", config.checkpoint_dir)
        if self.config_path and os.path.exists(self.config_path):
            self._load_from_file(config)
        return config
    
    def _load_from_file(self, config: Config):
        try:
            with open(self.config_path, 'r') as f:
                file_config = json.load(f)
            env_config = file_config.get(config.environment, {})
            for section, values in env_config.items():
                if hasattr(config, section):
                    section_obj = getattr(config, section)
                    for key, value in values.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")


@lru_cache(maxsize=1)
def get_config(config_path: Optional[str] = None) -> Config:
    loader = ConfigLoader(config_path)
    config = loader.load()
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
    return config


def reset_config():
    get_config.cache_clear()