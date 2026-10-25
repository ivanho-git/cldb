"""
Generate all paper figures and result tables from the continual-RL benchmark.

Reads  : experiments/results/cl_results.json
Writes : paper/figures/*.png
         paper/tables/*.tex   (LaTeX table bodies)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIGS = os.path.join(ROOT, "paper", "figures")
TABLES = os.path.join(ROOT, "paper", "tables")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

TASK_NAMES = ["OLTP", "Filters", "Filters*", "Analytics", "Analytics*", "OLTP*", "OLTP-ret"]
AGENTS = ["CLDB-RL (EWC+Prioritized)", "RL (no EWC)", "RL (online, no replay)"]
COLORS = {"CLDB-RL (EWC+Prioritized)": "#00b37e", "RL (no EWC)": "#f08c00", "RL (online, no replay)": "#e03131"}


def load():
    with open(os.path.join(ROOT, "experiments", "results", "cl_results.json")) as f:
        return json.load(f)


def fig_retention(data):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(TASK_NAMES))
    keep = data["baselines"]["KEEP"]
    ax.plot(x, keep, linestyle="--", color="#868e96", label="KEEP baseline")
    for name in AGENTS:
        fm = data["agents"][name]["final_mean"]
        ax.plot(x, fm, marker="o", linewidth=2, color=COLORS[name], label=name)
    ax.set_xlabel("Workload phase (task)")
    ax.set_ylabel("Mean reward after final phase (higher is better)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}\n{n}" for i, n in enumerate(TASK_NAMES)], fontsize=7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_retention.png"), dpi=180)
    plt.close(fig)


def fig_metrics(data):
    metrics = ["ACC", "BWT", "FWT", "F"]
    x = np.arange(len(metrics))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, name in enumerate(AGENTS):
        means = [data["agents"][name]["metrics"][m][0] for m in metrics]
        stds = [data["agents"][name]["metrics"][m][1] for m in metrics]
        ax.bar(x + (k - 1) * w, means, w, yerr=stds, capsize=3, color=COLORS[name], label=name)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_metrics.png"), dpi=180)
    plt.close(fig)


def fig_rmatrix(data):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, name in zip(axes, AGENTS):
        R = np.array([[np.nan if v is None else v for v in row] for row in data["agents"][name]["R_mean"]])
        im = ax.imshow(R, cmap="viridis", vmin=-1.5, vmax=-0.4)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("Evaluated task j")
        ax.set_ylabel("After training task i")
        ax.set_xticks(range(len(TASK_NAMES)))
        ax.set_xticklabels(range(len(TASK_NAMES)))
        ax.set_yticks(range(len(TASK_NAMES)))
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if not np.isnan(R[i, j]):
                    ax.text(j, i, f"{R[i,j]:.1f}", ha="center", va="center", color="white", fontsize=6)
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean reward")
    fig.savefig(os.path.join(FIGS, "fig_rmatrix.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_learning_curve():
    """Short single-task learning curve (DQN on phase 0 vs baselines)."""
    from cldb.sim_env import IndexAdvisorEnv
    from cldb.rl_agent import DQNAgent
    from cldb.replay_buffer import DatabaseAwarePriorityReplay

    env = IndexAdvisorEnv(seed=0, budget=2, horizon=36)
    agent = DQNAgent(feature_dim=env.feature_dim, embedding_dim=env.embedding_dim,
                     context_dim=3, index_state_dim=3, hidden_dims=(128, 64), seed=0)
    buf = DatabaseAwarePriorityReplay(capacity=20000)
    rewards = []
    for ep in range(30):
        obs = env.reset(task_id=0)
        for t in range(env.horizon):
            mask = env.valid_actions()
            a = agent.select_action(env.obs_vector(obs), valid_actions=mask, explore=True)
            nobs, r, done, _ = env.step(a)
            buf.add_transition(f"e{ep}t{t}", a, r, done, state=obs, next_state=nobs)
            obs = nobs
            if len(buf) >= 48 and t % 2 == 0:
                agent.train_from_buffer(buf, batch_size=64)
        env.set_task(0)
        rewards.append(env.evaluate(lambda o, m: agent.predict(env.obs_vector(o), valid_actions=m)[0]))

    keep = env.evaluate(lambda o, m: 0)
    greedy = env.evaluate(lambda o, m: 1 if m[1] else 0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(rewards) + 1), rewards, marker="o", ms=3, color="#1971c2", label="DQN (greedy eval)")
    ax.axhline(keep, ls="--", color="#868e96", label="KEEP baseline")
    ax.axhline(greedy, ls=":", color="#f08c00", label="GREEDY-CREATE")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean reward on phase 0")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_learning_curve.png"), dpi=180)
    plt.close(fig)


def write_tables(data):
    lines = []
    lines.append(r"\begin{table}[t]\centering\small")
    lines.append(r"\caption{Baseline per-phase reward (mean over seeds). Higher is better.}")
    lines.append(r"\label{tab:baselines}")
    lines.append(r"\begin{tabular}{l" + "r" * (len(TASK_NAMES) + 1) + "}")
    lines.append(r"\toprule")
    lines.append("Method & " + " & ".join([f"T{i}" for i in range(len(TASK_NAMES))]) + r" & Mean \\")
    lines.append(r"\midrule")
    for name in ["KEEP", "RANDOM", "GREEDY-CREATE"]:
        row = data["baselines"][name]
        fmt = " & ".join(f"{v:.2f}" for v in row)
        lines.append(f"{name} & {fmt} & {np.mean(row):.2f} \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    with open(os.path.join(TABLES, "baselines.tex"), "w") as f:
        f.write("\n".join(lines))

    lines = []
    lines.append(r"\begin{table}[t]\centering\small")
    lines.append(r"\caption{Continual-learning metrics (mean $\pm$ std over %d seeds). "
                 r"ACC/BWT/FWT/F per Lopez-Paz \& Ranzato.}" % len(data["config"]["seeds"]))
    lines.append(r"\label{tab:main}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & ACC & BWT & FWT & Forgetting $\mathcal{F}$ \\")
    lines.append(r"\midrule")
    for name in AGENTS:
        m = data["agents"][name]["metrics"]
        lines.append(f"{name} & " + " & ".join(f"${m[k][0]:.3f}\\pm{m[k][1]:.3f}$" for k in ["ACC", "BWT", "FWT", "F"]) + r" \\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    with open(os.path.join(TABLES, "main_results.tex"), "w") as f:
        f.write("\n".join(lines))

    # significance table: all pairwise paired t-tests over seeds
    from scipy import stats as sps
    pairs = [("CLDB-RL (EWC+Prioritized)", "RL (no EWC)"),
             ("CLDB-RL (EWC+Prioritized)", "RL (online, no replay)"),
             ("RL (no EWC)", "RL (online, no replay)")]
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{Paired $t$-tests over 8 seeds on forgetting $\mathcal{F}$. "
             r"Bold $p$ are significant at $0.05$.}",
             r"\label{tab:sig}", r"\begin{tabular}{lrr}", r"\toprule",
             r"Comparison ($\mathcal{F}$) & $t$ & $p$ \\", r"\midrule"]
    for a, b in pairs:
        fa = [m["F"] for m in data["agents"][a]["metrics_seeds"]]
        fb = [m["F"] for m in data["agents"][b]["metrics_seeds"]]
        t, p = sps.ttest_rel(fa, fb)
        short_a = a.replace("CLDB-RL (EWC+Prioritized)", "CLDB-RL").replace("RL (no EWC)", "no-EWC").replace("RL (online, no replay)", "online")
        short_b = b.replace("CLDB-RL (EWC+Prioritized)", "CLDB-RL").replace("RL (no EWC)", "no-EWC").replace("RL (online, no replay)", "online")
        pfmt = (r"\textbf{%.3f}" % p) if p < 0.05 else ("%.3f" % p)
        lines.append(f"{short_a} vs. {short_b} & {t:+.2f} & {pfmt} \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    with open(os.path.join(TABLES, "significance.tex"), "w") as f:
        f.write("\n".join(lines))
    print("wrote tables to paper/tables/")


def main():
    data = load()
    fig_retention(data)
    fig_metrics(data)
    fig_rmatrix(data)
    fig_learning_curve()
    write_tables(data)
    print("figures -> paper/figures/")


if __name__ == "__main__":
    main()
