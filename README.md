# CLDB (Continual Learning Database Engine)

**Industrial-grade machine learning system for automatic database index optimization.**

CLDB observes incoming SQL queries over time, extracts structural features, computes semantic embeddings, and uses a **deep Q-network (DQN) reinforcement-learning agent** to decide whether to **KEEP**, **CREATE**, or **DROP** indexes. The agent is trained from the latency/plan-cost reward it observes by executing queries, and to prevent catastrophic forgetting when workloads drift it combines **drift-triggered Elastic Weight Consolidation (EWC)** with a **database-aware prioritized replay buffer**. Both are implemented natively in PyTorch.

Its distinguishing mechanism is **Counterfactual What-If Learning (CWIL)**: at every decision a what-if oracle evaluates *all* structurally valid actions from an identical state, and the agent is trained to rank those actions by their observed outcomes. Replay learns from the action that was taken; CWIL learns from the actions that were not.

## Features

### Production-Grade Architecture
- **Resilient Design**: Circuit breakers, retry logic with exponential backoff, graceful degradation
- **Observability**: Structured JSON logging, Prometheus metrics, health endpoints
- **Configuration Management**: Environment variable support, validation, multi-environment configs
- **Model Training**: Checkpointing, early stopping, LR scheduling, MLflow integration

### Reinforcement Learning
- **Off-policy DQN**: Q-network with a frozen target network and Huber TD loss
- **State**: query features (74) + semantic embedding (384) + workload context (3) + index-configuration state (3)
- **Actions**: `KEEP` / `CREATE_INDEX` / `DROP_INDEX`, with structural action masking (no `CREATE` without predicate columns, no `DROP` without an index)
- **Reward**: latency improvement minus build/resource/regression penalties (`RewardCalculator`)
- **Exploration**: decaying epsilon-greedy; greedy at evaluation time
- **CWIL**: a non-mutating what-if oracle (`IndexAdvisorEnv.what_if`, `cldb.counterfactual`) rolls out every valid action from one snapshot; `DQNAgent.train_counterfactual` applies a pairwise preference loss so the best what-if action out-scores every strictly worse one. This teaches the cost of a harmful index without paying for it, which is exactly what replay-only agents fail to learn on conflicting workload phases.

### Continual Learning
- **Drift-triggered EWC**: consolidates the policy only when workload drift is detected, not on fixed boundaries
- **Database-aware prioritized replay**: samples by latency impact x workload rarity and refines priorities with TD errors
- **Multiple Architectures**: standard, residual, and attention-based networks for supervised warm-starting
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

# Run the continual-RL benchmark (baselines + ablations + CL metrics)
python experiments/run_rl_evaluation.py

# Isolate CWIL on the conflict phases (replay-only vs replay+CWIL)
python experiments/run_cwil_ablation.py

# Run the supervised catastrophic-forgetting study
python experiments/run_evaluation.py
```

The RL benchmark is configurable via environment variables (`CLDB_SEEDS`,
`CLDB_EPISODES`, `CLDB_HORIZON`, `CLDB_BUDGET`, `CLDB_CF_WEIGHT`,
`CLDB_AGENTS`) and reports ACC, BWT, FWT, average forgetting, and a
**conflict-phase reward** for the DQN agent, EWC/replay ablations, and static
baselines. Results are exported to `experiments/results/cl_results.json`; the
dashboard reads that file directly.

CWIL is an opt-in ablation: the silent-drift experiment is
`experiments/run_cwil_ablation.py`, and it can also be added to the continual
benchmark with `CLDB_AGENTS=CWIL`. On the full sequence it did not improve the
headline agent, so it is not presented as doing so; it is a targeted mechanism
for adaptation before the drift detector assigns a new phase.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          CLDB Pipeline                               │
├──────────────────────────────────────────────────────────────────────┤
│  Query → Parser → Feature Encoder → Embedder → RL Agent (DQN)        │
│              ↓              ↓              ↓                         │
│         SQL Features    Vector DB     Q-Network + EWC                │
│              ↓              ↓              ↓                         │
│         Decision Engine  Similarity   Action + Confidence            │
│              ↓                                 ↓                     │
│         Safety Layer ←────── Validation ←────────────────            │
│              ↓                                 ↓                     │
│         Executor → Database (CREATE/DROP INDEX)                      │
│              ↓                                 ↓                     │
│         Feedback → Vector Store + Replay Buffer → Background DQN     │
│                                                       ↑              │
│              What-if oracle → all-action TD (CWIL) ───┘              │
└──────────────────────────────────────────────────────────────────────┘
```

## Project layout

There is exactly one pipeline (`CLDBPipeline`; `ProductionPipeline` is an alias)
and one implementation of each subsystem. The former `cl_engine`, `vector_store`,
`safety_layer`, and `production_pipeline` re-export modules were removed.

| Module | Responsibility |
|--------|----------------|
| `cldb/pipeline.py` | The online pipeline: decide, validate, execute, measure, learn |
| `cldb/rl_agent.py` | DQN, replay updates, EWC consolidation, CWIL update |
| `cldb/counterfactual.py` | What-if oracle protocol and counterfactual batching |
| `cldb/sim_env.py` | Reproducible cost-model MDP, non-mutating `what_if()` |
| `cldb/replay_buffer.py` | Database-aware prioritized replay |
| `cldb/ewc.py` | Native Elastic Weight Consolidation |
| `cldb/enhanced_cl_engine.py` | Supervised warm-start networks |
| `cldb/drift_detection.py` | Embedding-space workload drift detector |
| `cldb/enhanced_safety.py` | Confidence, index limits, rollback, audit |
| `cldb/enhanced_vector_store.py` | Qdrant store with in-memory fallback |

## Monitoring

Access metrics at:
- **Dashboard**: http://localhost:8501
- **Health**: http://localhost:8080/health
- **Prometheus**: http://localhost:9090
- **MLflow**: http://localhost:5001

## License

MIT