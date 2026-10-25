"""
Continual RL benchmark for CLDB.

Trains a DQN index-selection agent sequentially across non-stationary workload
phases and measures standard continual-learning metrics against baselines and
ablations, averaged over multiple seeds.

Agents
------
* CLDB-RL          : DQN + drift-triggered EWC + database-aware prioritized replay
* RL (no EWC)      : DQN + prioritized replay (catastrophic-forgetting control)
* RL (uniform)     : DQN + EWC + uniform replay (replay-priority ablation)

Baselines
---------
* KEEP            : never modify indexes
* RANDOM          : uniform random valid action
* GREEDY-CREATE   : always create an index when structurally valid

Metrics (Lopez-Paz & Ranzato 2017 / Chaudhry et al. 2018)
---------------------------------------------------------
* ACC  : average final performance over all tasks
* BWT  : backward transfer (does learning new tasks help/hurt old tasks?)
* FWT  : forward transfer (does the agent learn faster than from scratch?)
* F    : average forgetting (peak performance minus final performance)
"""
import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cldb.sim_env import IndexAdvisorEnv, PHASE_SELECTIVITY, PHASE_TEMPLATES
from cldb.counterfactual import SimulatorOracle, build_counterfactual_set
from cldb.drift_detection import DriftDetector
from cldb.rl_agent import DQNAgent
from cldb.replay_buffer import DatabaseAwarePriorityReplay


TASKS = sorted(PHASE_TEMPLATES.keys())
# Phases whose predicates invert an earlier phase's selectivity: the optimal
# action for an identical query flips, so these are the forgetting-critical
# phases and the ones a replay-only agent fails to KEEP on.
CONFLICT_TASKS = [t for t in TASKS
                  if any(v == "low" for v in PHASE_SELECTIVITY.get(t, {}).values())]
EPISODES_PER_TASK = int(os.environ.get("CLDB_EPISODES", "10"))
HORIZON = int(os.environ.get("CLDB_HORIZON", "30"))
SEEDS = [int(s) for s in os.environ.get("CLDB_SEEDS", "0,1,2,3,4,5,6,7,8,9").split(",")]
TRAIN_EVERY = 2
WARMUP = 48
INDEX_BUDGET = int(os.environ.get("CLDB_BUDGET", "2"))
CF_WEIGHT = float(os.environ.get("CLDB_CF_WEIGHT", "1.0"))

AGENT_CONFIGS = {
    "CLDB-RL (EWC+Prioritized)": dict(ewc=True, prioritized=True, capacity=20000),
    "CLDB-RL+CWIL (ablation)": dict(ewc=True, prioritized=True, capacity=20000,
                                    counterfactual=True, cf_weight=CF_WEIGHT),
    "RL (no EWC)": dict(ewc=False, prioritized=True, capacity=20000),
    "RL (online, no replay)": dict(ewc=False, prioritized=True, capacity=64),
}

# Optional subset, e.g. CLDB_AGENTS="CWIL,no EWC" for a focused ablation.
_agent_filter = os.environ.get("CLDB_AGENTS", "").strip()
if _agent_filter:
    wanted = [token.strip().lower() for token in _agent_filter.split(",") if token.strip()]
    AGENT_CONFIGS = {name: cfg for name, cfg in AGENT_CONFIGS.items()
                     if any(token in name.lower() for token in wanted)}


def make_policies():
    return {
        "KEEP": lambda obs, mask: 0,
        "RANDOM": lambda obs, mask: int(np.random.choice(np.flatnonzero(mask))),
        "GREEDY-CREATE": lambda obs, mask: 1 if mask[1] else 0,
    }


def greedy_policy(agent, env):
    return lambda obs, mask: agent.predict(env.obs_vector(obs), valid_actions=mask)[0]


def make_agent(env, cfg, seed):
    return DQNAgent(
        feature_dim=env.feature_dim, embedding_dim=env.embedding_dim,
        context_dim=3, index_state_dim=3, hidden_dims=(128, 64),
        learning_rate=5e-4, gamma=0.97, batch_size=64,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=2500,
        target_update_interval=150, ewc_lambda=0.4 if cfg["ewc"] else 0.0, seed=seed,
    )


def run_sequence(cfg, seed):
    """Train one agent sequentially over all tasks; return the R-matrix and info."""
    env = IndexAdvisorEnv(seed=seed, budget=INDEX_BUDGET)
    agent = make_agent(env, cfg, seed)
    buffer = DatabaseAwarePriorityReplay(
        capacity=cfg.get("capacity", 20000),
        alpha=0.6 if cfg["prioritized"] else 0.0,
        beta=0.4 if cfg["prioritized"] else 0.0,
    )
    detector = DriftDetector(window_size=24, drift_threshold=0.25)
    oracle = SimulatorOracle(env) if cfg.get("counterfactual") else None

    T = len(TASKS)
    R = np.full((T, T), np.nan)
    drift_events = 0
    global_step = 0
    pending_states = []

    for i, task in enumerate(TASKS):
        env.set_task(task)
        task_drift = 0
        for ep in range(EPISODES_PER_TASK):
            obs = env.reset(task_id=task)
            for t in range(HORIZON):
                mask = env.valid_actions()
                state_vec = env.obs_vector(obs)

                # CWIL: evaluate every valid action through the what-if oracle
                # *before* committing, and rank them with the preference loss.
                if oracle is not None:
                    cf = build_counterfactual_set(oracle.what_if(), env.obs_vector)
                    agent.train_counterfactual(state_vec, cf,
                                               weight=cfg.get("cf_weight", CF_WEIGHT))

                action = agent.select_action(state_vec, valid_actions=mask, explore=True)
                next_obs, reward, done, _ = env.step(action)
                buffer.add_transition(f"t{task}e{ep}s{t}", action, reward, done, state=obs, next_state=next_obs)
                pending_states.append(state_vec)

                if len(buffer) >= WARMUP and global_step % TRAIN_EVERY == 0:
                    agent.train_from_buffer(buffer, batch_size=64)

                # Drift-triggered consolidation (falls back to task boundary).
                if detector.add_experience(obs[1]):
                    task_drift += 1
                    drift_events += 1
                    if cfg["ewc"] and pending_states:
                        agent.consolidate(f"phase_{i}_{task_drift}",
                                          torch.tensor(np.stack(pending_states), dtype=torch.float32))
                        pending_states = []

                obs = next_obs
                global_step += 1
                if done:
                    break

        # Boundary fallback so EWC always sees each phase.
        if cfg["ewc"] and pending_states:
            agent.consolidate(f"phase_{i}_boundary",
                              torch.tensor(np.stack(pending_states), dtype=torch.float32))
            pending_states = []

        # Evaluate greedily on every task seen so far.
        policy = greedy_policy(agent, env)
        for j in range(i + 1):
            env.set_task(TASKS[j])
            R[i, j] = env.evaluate(policy)

    return R, {"drift_events": drift_events}


def baseline_matrix(seed):
    env = IndexAdvisorEnv(seed=seed + 777, budget=INDEX_BUDGET)
    T = len(TASKS)
    perf = {}
    for name, policy in make_policies().items():
        row = []
        for task in TASKS:
            env.set_task(task)
            row.append(env.evaluate(policy))
        perf[name] = np.array(row)
    return perf


def compute_metrics(R, baseline_row):
    T = R.shape[0]
    final = R[T - 1]
    acc = np.nanmean(final)
    bwt = np.nanmean([final[j] - R[j, j] for j in range(T)])
    fwt = np.nanmean([R[j, j] - baseline_row[j] for j in range(T)])
    forgetting = np.nanmean([np.nanmax(R[j:, j]) - final[j] for j in range(T)])
    conflict_idx = [j for j, task in enumerate(TASKS) if task in CONFLICT_TASKS]
    conflict = np.nanmean([final[j] for j in conflict_idx]) if conflict_idx else np.nan
    return {"ACC": acc, "BWT": bwt, "FWT": fwt, "F": forgetting, "Conflict": conflict}


def aggregator(values):
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def _export(results, mean_baseline, path):
    export = {
        "config": {"tasks": TASKS, "episodes": EPISODES_PER_TASK, "horizon": HORIZON,
                   "budget": INDEX_BUDGET, "seeds": SEEDS},
        "baselines": {name: [float(v) for v in mean_baseline[name]] for name in mean_baseline},
        "agents": {
            name: {
                "metrics": {k: [float(res["metrics"][k][0]), float(res["metrics"][k][1])] for k in res["metrics"]},
                "metrics_seeds": [{k: float(v) for k, v in m.items()} for m in res["metrics_seeds"]],
                "final_mean": [float(v) for v in res["final"]],
                "R_mean": [[None if np.isnan(v) else float(v) for v in row] for row in res["R_mean"]],
                "drift_events": res["drift_events"],
            } for name, res in results.items()
        },
    }
    with open(path, "w") as f:
        json.dump(export, f, indent=2)
    return export


def main():
    print("=" * 78)
    print("CLDB Continual RL Benchmark")
    print(f"tasks={TASKS}  episodes/task={EPISODES_PER_TASK}  horizon={HORIZON}  seeds={SEEDS}")
    print("=" * 78)

    # Baselines (no learning): evaluated once per seed, then averaged.
    baseline_metrics = {}
    baseline_rows = []
    for seed in SEEDS:
        perf = baseline_matrix(seed)
        baseline_rows.append(perf)
    mean_baseline = {name: np.mean([b[name] for b in baseline_rows], axis=0)
                     for name in next(iter(baseline_rows))}

    print("\nBaseline per-task performance (mean reward, higher is better):")
    for name, row in mean_baseline.items():
        print(f"  {name:14s} " + " ".join(f"{v:7.3f}" for v in row) + f"   mean={row.mean():7.3f}")

    results = {}
    for agent_name, cfg in AGENT_CONFIGS.items():
        metrics_seeds = []
        finals_seeds = []
        R_seeds = []
        drift_total = 0
        for seed in SEEDS:
            t0 = time.time()
            R, info = run_sequence(cfg, seed)
            ref = mean_baseline["RANDOM"]
            metrics_seeds.append(compute_metrics(R, ref))
            finals_seeds.append(R[-1])
            R_seeds.append(R)
            drift_total += info["drift_events"]
            print(f"  [{agent_name}] seed {seed} done in {time.time()-t0:5.1f}s  "
                  f"final={np.round(R[-1], 3)}")
        results[agent_name] = {
            "metrics": {k: aggregator([m[k] for m in metrics_seeds]) for k in metrics_seeds[0]},
            "metrics_seeds": metrics_seeds,
            "final": np.mean(finals_seeds, axis=0),
            "R_mean": np.nanmean(np.stack(R_seeds), axis=0),
        }
        results[agent_name]["drift_events"] = drift_total
        diag = np.diag(results[agent_name]["R_mean"])
        print(f"  -> {agent_name} mean diagonal R[j][j]={np.round(diag,3)}")
        print("     R_mean (rows=after training task i):")
        for i, row in enumerate(results[agent_name]["R_mean"]):
            print("       " + " ".join("  ----" if np.isnan(v) else f"{v:6.3f}" for v in row))
        os.makedirs("experiments/results", exist_ok=True)
        _export(results, mean_baseline, "experiments/results/cl_results_partial.json")

    print("\n" + "=" * 78)
    print("SUMMARY (mean over seeds)")
    print("=" * 78)
    metric_cols = ["ACC", "BWT", "FWT", "F", "Conflict"]
    header = f"{'agent':30s} " + " ".join(f"{m:>14s}" for m in metric_cols)
    print(header)
    for agent_name, res in results.items():
        m = res["metrics"]
        cells = " ".join(f"{m[k][0]:6.3f}±{m[k][1]:5.3f}" for k in metric_cols)
        print(f"{agent_name:30s} {cells}")

    # Pairwise significance vs the full CWIL model (paired t-test over seeds).
    full_name = "CLDB-RL+CWIL (ablation)"
    significance = {}
    if full_name in results:
        print("\nPaired t-tests vs " + full_name + " (two-sided):")
    for other in results:
        if other == full_name:
            continue
        entry = {}
        if full_name not in results:
            continue
        for metric in ("F", "ACC", "Conflict"):
            full_vals = [m[metric] for m in results[full_name]["metrics_seeds"]]
            other_vals = [m[metric] for m in results[other]["metrics_seeds"]]
            t_stat, p_val = stats.ttest_rel(full_vals, other_vals)
            entry[metric] = [float(t_stat), float(p_val)]
        significance[other] = entry
        print(f"  {other:30s}  F: t={entry['F'][0]:+.2f} p={entry['F'][1]:.4f} | "
              f"ACC: t={entry['ACC'][0]:+.2f} p={entry['ACC'][1]:.4f} | "
              f"Conflict: t={entry['Conflict'][0]:+.2f} p={entry['Conflict'][1]:.4f}")

    # ---- export results for the paper ----
    os.makedirs("experiments/results", exist_ok=True)
    export = {
        "config": {"tasks": TASKS, "episodes": EPISODES_PER_TASK, "horizon": HORIZON,
                   "budget": INDEX_BUDGET, "seeds": SEEDS},
        "baselines": {name: [float(v) for v in mean_baseline[name]] for name in mean_baseline},
        "agents": {
            name: {
                "metrics": {k: [float(res["metrics"][k][0]), float(res["metrics"][k][1])] for k in res["metrics"]},
                "metrics_seeds": [{k: float(v) for k, v in m.items()} for m in res["metrics_seeds"]],
                "final_mean": [float(v) for v in res["final"]],
                "R_mean": [[None if np.isnan(v) else float(v) for v in row] for row in res["R_mean"]],
                "drift_events": res["drift_events"],
            } for name, res in results.items()
        },
        "significance_vs_full": significance,
    }
    with open("experiments/results/cl_results.json", "w") as f:
        json.dump(export, f, indent=2)
    print("\nExported experiments/results/cl_results.json")

    # ---- plots ----
    os.makedirs("experiments/results", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    palette = ["#00c896", "#f59f00", "#ff6b6b", "#4dabf7", "#845ef7"]
    colors = [palette[i % len(palette)] for i in range(len(results))]
    task_names = {0: "OLTP", 1: "Filters", 2: "Filters*", 3: "Analytics", 4: "Analytics*",
                  5: "OLTP*", 6: "OLTP-ret"}
    task_labels = [f"T{i}\n({task_names.get(i, '')})" for i in range(len(TASKS))]
    for (agent_name, res), color in zip(results.items(), colors):
        axes[0].plot(range(len(TASKS)), res["final"], marker="o", linewidth=2, label=agent_name, color=color)
    axes[0].axhline(mean_baseline["KEEP"].mean(), linestyle="--", color="#888", label="KEEP baseline")
    axes[0].set_title("Final performance retained per task")
    axes[0].set_xlabel("Task")
    axes[0].set_ylabel("Mean reward (higher is better)")
    axes[0].set_xticks(range(len(TASKS)))
    axes[0].set_xticklabels(task_labels)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    metric_names = ["ACC", "BWT", "FWT", "F", "Conflict"]
    x = np.arange(len(metric_names))
    width = 0.8 / max(1, len(results))
    for k, ((agent_name, res), color) in enumerate(zip(results.items(), colors)):
        vals = [res["metrics"][m][0] for m in metric_names]
        errs = [res["metrics"][m][1] for m in metric_names]
        offset = (k - (len(results) - 1) / 2) * width
        axes[1].bar(x + offset, vals, width, yerr=errs, capsize=3,
                    label=agent_name, color=color, alpha=0.9)
    axes[1].set_title("Continual-learning metrics")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metric_names)
    axes[1].axhline(0, color="#444", linewidth=0.8)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig("experiments/results/rl_continual_learning.png", dpi=150, bbox_inches="tight")
    print("\nChart saved to experiments/results/rl_continual_learning.png")

    return results


if __name__ == "__main__":
    main()
