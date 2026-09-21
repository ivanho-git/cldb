# CLDB (Continual Learning Database Engine)

**Industrial-grade machine learning system for automatic database index optimization.**

CLDB observes incoming SQL queries over time, extracts structural features, computes semantic embeddings, and uses a PyTorch policy network to decide whether to **CREATE** or **DROP** indexes. To prevent catastrophic forgetting when workloads shift, CLDB integrates Elastic Weight Consolidation (EWC) and a replay buffer using `avalanche-lib`.

## Features

### Production-Grade Architecture
- **Resilient Design**: Circuit breakers, retry logic with exponential backoff, graceful degradation
- **Observability**: Structured JSON logging, Prometheus metrics, health endpoints
- **Configuration Management**: Environment variable support, validation, multi-environment configs
- **Model Training**: Checkpointing, early stopping, LR scheduling, MLflow integration

### Continual Learning
- **Elastic Weight Consolidation (EWC)**: Prevents catastrophic forgetting
- **Experience Replay**: Stores and replays important samples
- **Multiple Architectures**: Standard, Residual, and Attention-based policy networks
- **Vector Similarity Search**: Qdrant-powered semantic query matching

### Safety & Reliability
- **Automatic Rollback**: Reverts index changes on performance regression
- **Confidence Thresholds**: Rejects low-confidence decisions
- **Rate Limiting**: Prevents concurrent operation overload
- **Audit Logging**: Full decision trail

## Quickstart

```bash
# Development
docker-compose up -d

# Production
docker-compose -f docker-compose.yml up -d
```

## Configuration

All configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CLDB_ENVIRONMENT` | development | Environment (development/production) |
| `POSTGRES_HOST` | localhost | PostgreSQL host |
| `POSTGRES_PORT` | 5432 | PostgreSQL port |
| `POSTGRES_USER` | postgres | PostgreSQL user |
| `POSTGRES_PASSWORD` | postgres | PostgreSQL password |
| `POSTGRES_DB` | cldb | Database name |
| `QDRANT_HOST` | localhost | Qdrant host |
| `QDRANT_PORT` | 6333 | Qdrant port |
| `ML_LR` | 0.001 | Learning rate |
| `ML_BATCH_SIZE` | 32 | Training batch size |
| `EWC_LAMBDA` | 0.4 | EWC lambda parameter |
| `REPLAY_MEM_SIZE` | 500 | Replay buffer size |
| `SAFETY_CONFIDENCE_THRESHOLD` | 0.6 | Minimum confidence to execute |
| `LOG_LEVEL` | INFO | Logging level |
| `LOG_FORMAT` | text | Log format (text/json) |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | JSON health status |
| `GET /health/text` | Text health status |
| `GET /metrics` | Prometheus metrics |
| `GET /ready` | Kubernetes readiness probe |
| `GET /live` | Kubernetes liveness probe |

## Usage Example

```python
from cldb import ProductionPipeline, get_config

# Load configuration
config = get_config()

# Initialize pipeline
pipeline = ProductionPipeline(
    repo=metadata_repo,
    vector_store=vector_store,
    confidence_threshold=config.safety.confidence_threshold
)

# Process queries
result = pipeline.process_query("SELECT * FROM orders WHERE customer_id = 42")

# Batch processing
results = pipeline.process_batch([
    "SELECT * FROM products WHERE category = 'Electronics'",
    "SELECT * FROM orders WHERE status = 'PENDING'"
])

# Train on new workload
pipeline.train_on_queries(queries, labels, task_id=1)
```

## Development

```bash
# Run tests
pytest tests/ -v

# Run specific test
pytest tests/test_enhanced_cl_engine.py -v

# Run with coverage
pytest --cov=cldb tests/

# Start dashboard
streamlit run dashboard/app.py

# Run experiments
python experiments/run_evaluation.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLDB Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│  Query → Parser → Feature Encoder → Embedder → CL Engine        │
│              ↓              ↓              ↓                    │
│         SQL Features    Vector DB     Policy Network            │
│              ↓              ↓              ↓                    │
│         Decision Engine  Similarity   Action + Confidence        │
│              ↓                                 ↓                 │
│         Safety Layer ←────── Validation ←────────────────        │
│              ↓                                                 │
│         Executor → Database (CREATE/DROP INDEX)                 │
│              ↓                                                 │
│         Feedback → Vector Store + Replay Buffer                 │
└─────────────────────────────────────────────────────────────────┘
```

## Monitoring

Access metrics at:
- **Dashboard**: http://localhost:8501
- **Health**: http://localhost:8080/health
- **Prometheus**: http://localhost:9090
- **MLflow**: http://localhost:5001

## License

MIT