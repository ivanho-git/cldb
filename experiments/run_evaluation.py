"""
CLDB Continual Learning Evaluation.

Compares the quality of optimization decisions of CLDB as an optimization layer 
on top of PostgreSQL against the default PostgreSQL optimizer.

Scenarios evaluated:
1. Stable workload
2. Gradually evolving workload
3. Sudden workload shift
4. Returning to a previously seen workload (Catastrophic Forgetting test)
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from cldb.parser import SQLParser
from cldb.features import FeatureEncoder
from cldb.embeddings import QueryEmbedder
from cldb.enhanced_cl_engine import CLEngine, EMBEDDING_DIM, CONTEXT_DIM
from cldb.replay_buffer import DatabaseAwarePriorityReplay
from cldb.config import get_config

# 1. Stable Workload (OLTP)
PHASE_1_QUERIES = [
    "SELECT * FROM orders WHERE customer_id = 42;",
    "SELECT * FROM order_items WHERE product_id = 7;",
    "SELECT * FROM customers WHERE customer_id = 15;",
    "SELECT o.*, c.name FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE o.customer_id = 23;",
] * 5

# 2. Gradually Evolving (Mixed OLTP + simple analytical)
PHASE_2_QUERIES = [
    "SELECT * FROM products WHERE category = 'Electronics';",
    "SELECT * FROM orders WHERE status = 'PENDING';",
    "SELECT p.name, p.price FROM products p WHERE p.category = 'Electronics';",
    "SELECT order_id, status FROM orders WHERE status = 'PENDING';",
] * 5

# 3. Sudden Workload Shift (Heavy Analytics)
PHASE_3_QUERIES = [
    "SELECT c.region, COUNT(o.order_id), SUM(o.total_amount) FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.region;",
    "SELECT DATE_TRUNC('month', order_date) as month, SUM(total_amount) FROM orders GROUP BY DATE_TRUNC('month', order_date);",
    "SELECT customer_id, COUNT(*), SUM(total_amount) FROM orders GROUP BY customer_id ORDER BY SUM(total_amount) DESC LIMIT 10;",
] * 6

# 4. Return to seen workload (Phase 1 repeated)
PHASE_4_QUERIES = PHASE_1_QUERIES.copy()


PHASE_1_LABELS = [1, 1, 0, 0] * 5
PHASE_2_LABELS = [1, 1, 1, 1] * 5
PHASE_3_LABELS = [1, 1, 2] * 6
PHASE_4_LABELS = PHASE_1_LABELS.copy()


def prepare_phase_data(queries, labels):
    parser = SQLParser()
    encoder = FeatureEncoder()
    embedder = QueryEmbedder()

    x_features_list = []
    x_emb_list = []
    x_ctx_list = []

    for q in queries:
        features = parser.parse(q)
        x_feat = encoder.encode(features)
        x_emb = embedder.embed(q)
        x_ctx = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        x_features_list.append(x_feat)
        x_emb_list.append(x_emb)
        x_ctx_list.append(x_ctx)

    return (
        torch.tensor(np.array(x_features_list), dtype=torch.float32),
        torch.tensor(np.array(x_emb_list), dtype=torch.float32),
        torch.tensor(np.array(x_ctx_list), dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def evaluate_model(model, x_feat, x_emb, x_ctx, y_true):
    with torch.no_grad():
        if hasattr(model, "predict"):
            preds, _ = model.predict(x_feat, x_emb, x_ctx)
        else:
            preds = np.zeros(len(x_feat))
        correct = (preds == y_true.numpy()).sum()
    return correct / len(y_true)


def run():
    feature_dim = FeatureEncoder().get_feature_dim()
    config = get_config()

    # Define the 4 variants to compare
    # 1. Postgres Baseline (Dummy model that always predicts 0 / KEEP / no index)
    class PostgresBaseline:
        def predict(self, x_feat, x_emb, x_ctx):
            return np.zeros(len(x_feat)), np.ones(len(x_feat))
            
    postgres_baseline = PostgresBaseline()
    
    # 2. CLDB without replay (Standard finetuning, forgets quickly)
    cldb_no_replay = CLEngine(feature_dim=feature_dim)
    
    # 3. CLDB + Replay Buffer
    cldb_replay = CLEngine(feature_dim=feature_dim)
    replay_buffer = DatabaseAwarePriorityReplay(capacity=100)
    
    # 4. CLDB + Replay + EWC (Full System)
    cldb_full = CLEngine(feature_dim=feature_dim)
    full_replay_buffer = DatabaseAwarePriorityReplay(capacity=100)
    # EWC is already initialized in CLEngine due to our updates

    phases = [
        (PHASE_1_QUERIES, PHASE_1_LABELS, "Phase 1: Stable Workload"),
        (PHASE_2_QUERIES, PHASE_2_LABELS, "Phase 2: Evolving Workload"),
        (PHASE_3_QUERIES, PHASE_3_LABELS, "Phase 3: Sudden Shift"),
        (PHASE_4_QUERIES, PHASE_4_LABELS, "Phase 4: Return to Phase 1"),
    ]

    results = {
        "Postgres Default": [],
        "CLDB (No Replay)": [],
        "CLDB (+Replay)": [],
        "CLDB (Replay+EWC)": []
    }
    
    # For getting "Forgetting Measure", we need to test Phase 1 accuracy at every step
    phase1_retention = {
        "Postgres Default": [],
        "CLDB (No Replay)": [],
        "CLDB (+Replay)": [],
        "CLDB (Replay+EWC)": []
    }

    p1_feat, p1_emb, p1_ctx, p1_y = prepare_phase_data(PHASE_1_QUERIES, PHASE_1_LABELS)

    for idx, (queries, labels, name) in enumerate(phases):
        print(f"\n--- {name} ---")

        x_feat, x_emb, x_ctx, y = prepare_phase_data(queries, labels)
        
        # Train No Replay (just current batch)
        cldb_no_replay.train_on_batch(x_feat, x_emb, x_ctx, y)
        
        # Train +Replay
        for i in range(len(y)):
            replay_buffer.add(f"q_{idx}_{i}", x_feat[i].numpy(), x_emb[i].numpy(), x_ctx[i].numpy(), y[i].item(), 1.0)
        # We manually disable EWC penalty for this variant by setting lambda to 0
        original_lambda = cldb_replay.ewc.lambda_param
        cldb_replay.ewc.lambda_param = 0.0
        cldb_replay.train_on_experience(replay_buffer, batch_size=len(y))
        cldb_replay.ewc.lambda_param = original_lambda
        
        # Train +Replay+EWC
        # Register new workload phase in EWC before training
        if idx > 0:
            cldb_full.ewc.register_workload_phase(f"phase_{idx-1}", p1_feat, p1_emb, p1_ctx, p1_y)
            
        for i in range(len(y)):
            full_replay_buffer.add(f"q_{idx}_{i}", x_feat[i].numpy(), x_emb[i].numpy(), x_ctx[i].numpy(), y[i].item(), 1.0)
        cldb_full.train_on_experience(full_replay_buffer, batch_size=len(y))

        # Evaluate current phase accuracy
        results["Postgres Default"].append(evaluate_model(postgres_baseline, x_feat, x_emb, x_ctx, y))
        results["CLDB (No Replay)"].append(evaluate_model(cldb_no_replay, x_feat, x_emb, x_ctx, y))
        results["CLDB (+Replay)"].append(evaluate_model(cldb_replay, x_feat, x_emb, x_ctx, y))
        results["CLDB (Replay+EWC)"].append(evaluate_model(cldb_full, x_feat, x_emb, x_ctx, y))
        
        # Evaluate retention on Phase 1
        phase1_retention["Postgres Default"].append(evaluate_model(postgres_baseline, p1_feat, p1_emb, p1_ctx, p1_y))
        phase1_retention["CLDB (No Replay)"].append(evaluate_model(cldb_no_replay, p1_feat, p1_emb, p1_ctx, p1_y))
        phase1_retention["CLDB (+Replay)"].append(evaluate_model(cldb_replay, p1_feat, p1_emb, p1_ctx, p1_y))
        phase1_retention["CLDB (Replay+EWC)"].append(evaluate_model(cldb_full, p1_feat, p1_emb, p1_ctx, p1_y))
        
        print(f"  CLDB (Replay+EWC) Phase 1 Retention: {phase1_retention['CLDB (Replay+EWC)'][-1]:.2%}")

    os.makedirs("experiments/results", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Catastrophic Forgetting (Phase 1 Retention over time)
    x = [1, 2, 3, 4]
    axes[0].plot(x, phase1_retention["CLDB (Replay+EWC)"], marker="o", linewidth=3, label="CLDB (Replay+EWC)", color="#00ffaa")
    axes[0].plot(x, phase1_retention["CLDB (+Replay)"], marker="^", linewidth=2, label="CLDB (Replay Only)", color="#00aaff")
    axes[0].plot(x, phase1_retention["CLDB (No Replay)"], marker="x", linewidth=2, label="CLDB (No Replay)", color="#ff4b4b")
    axes[0].plot(x, phase1_retention["Postgres Default"], marker="s", linewidth=2, label="Postgres Baseline", color="#888888", linestyle="--")
    
    axes[0].set_title("Catastrophic Forgetting: Phase 1 Retention", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Training Phase")
    axes[0].set_ylabel("Accuracy on Phase 1 Workload")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Phase 1\n(Stable)", "Phase 2\n(Evolving)", "Phase 3\n(Shift)", "Phase 4\n(Return)"])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.1)

    # Plot 2: Per-Phase Accuracy Comparison
    phase_names = ["Stable", "Evolving", "Sudden Shift", "Return to P1"]
    x_bars = np.arange(4)
    width = 0.2
    
    axes[1].bar(x_bars - 1.5*width, results["Postgres Default"], width, label="Postgres Default", color="#888888", alpha=0.8)
    axes[1].bar(x_bars - 0.5*width, results["CLDB (No Replay)"], width, label="CLDB (No Replay)", color="#ff4b4b", alpha=0.8)
    axes[1].bar(x_bars + 0.5*width, results["CLDB (+Replay)"], width, label="CLDB (+Replay)", color="#00aaff", alpha=0.8)
    axes[1].bar(x_bars + 1.5*width, results["CLDB (Replay+EWC)"], width, label="CLDB (Replay+EWC)", color="#00ffaa", alpha=1.0)
    
    axes[1].set_title("Quality of Optimization Decisions per Phase", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Workload Phase")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xticks(x_bars)
    axes[1].set_xticklabels(phase_names)
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim(0, 1.1)

    plt.tight_layout()
    fig.savefig("experiments/results/forgetting_test.png", dpi=150, bbox_inches="tight")

    print("\n--- Summary ---")
    print(f"CLDB (Replay+EWC) Final Phase 1 retention: {phase1_retention['CLDB (Replay+EWC)'][-1]:.2%}")
    print(f"CLDB (No Replay) Final Phase 1 retention: {phase1_retention['CLDB (No Replay)'][-1]:.2%}")
    print("Chart saved to experiments/results/forgetting_test.png")
    
    # Calculate Forgetting Measure
    max_p1 = max(phase1_retention["CLDB (Replay+EWC)"])
    final_p1 = phase1_retention["CLDB (Replay+EWC)"][-1]
    forgetting_measure = max_p1 - final_p1
    print(f"Forgetting Measure for Full CLDB: {forgetting_measure:.2%}")


if __name__ == "__main__":
    run()