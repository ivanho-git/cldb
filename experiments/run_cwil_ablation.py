"""Silent-drift ablation for Counterfactual What-If Learning (CWIL).

In the continual benchmark, three effects are entangled: replay, EWC and
CWIL. This script isolates the part of the problem CWIL is designed for:
*silent* workload drift, where the query templates and the observable phase
cue stay the same but the predicate's selectivity flips, so indexing stops
paying off. The optimal response is to KEEP, and the only signal is reward.

Protocol per (warm phase, drifted phase) pair and seed:

1. Warm-start the agent on the selective phase, where CREATE is optimal.
2. Switch selectivity to the drifted phase *without* changing the task cue.
3. Continue online learning and evaluate greedily after every episode.

Two agents differing only in whether CWIL is enabled are compared on
zero-shot retention, adaptation speed (mean over the adaptation curve),
final KEEP rate and final reward.

Run:
    python experiments/run_cwil_ablation.py

Environment:
    CLDB_CWIL_PRE_EPISODES    warm-start episodes (default 6)
    CLDB_CWIL_ADAPT_EPISODES  post-drift episodes (default 10)
    CLDB_CWIL_HORIZON         steps per episode (default 30)
    CLDB_CWIL_SEEDS           comma-separated seeds (default 0,1,2)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cldb.counterfactual import SimulatorOracle, build_counterfactual_set
from cldb.replay_buffer import DatabaseAwarePriorityReplay
from cldb.rl_agent import ACTION_KEEP, DQNAgent
from cldb.sim_env import IndexAdvisorEnv

# (warm phase with selective predicates, drifted phase with inverted selectivity)
PAIRS = [(1, 2), (3, 4), (0, 5)]

PRE_EPISODES = int(os.environ.get("CLDB_CWIL_PRE_EPISODES", "6"))
ADAPT_EPISODES = int(os.environ.get("CLDB_CWIL_ADAPT_EPISODES", "10"))
HORIZON = int(os.environ.get("CLDB_CWIL_HORIZON", "30"))
SEEDS = [int(s) for s in os.environ.get("CLDB_CWIL_SEEDS", "0,1,2").split(",")]
EVAL_EPISODES = 3
TRAIN_EVERY = 2
WARMUP = 48


def make_agent(env, seed):
    return DQNAgent(
        feature_dim=env.feature_dim, embedding_dim=env.embedding_dim,
        context_dim=3, index_state_dim=3, hidden_dims=(128, 64),
        learning_rate=5e-4, gamma=0.97, batch_size=64,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=2500,
        target_update_interval=150, ewc_lambda=0.0, seed=seed,
    )


def train_episode(env, agent, buffer, oracle):
    obs = env.reset()
    for step in range(HORIZON):
        mask = env.valid_actions()
        state = env.obs_vector(obs)
        if oracle is not None:
            cf = build_counterfactual_set(oracle.what_if(), env.obs_vector)
            agent.train_counterfactual(state, cf)
        action = agent.select_action(state, valid_actions=mask, explore=True)
        next_obs, reward, done, _ = env.step(action)
        buffer.add_transition(f"s{step}", action, reward, done, state=obs, next_state=next_obs)
        if len(buffer) >= WARMUP and step % TRAIN_EVERY == 0:
            agent.train_from_buffer(buffer)
        obs = next_obs
        if done:
            break


def evaluate(agent, task, selectivity_phase, seed):
    """Greedy evaluation; returns (mean reward, KEEP rate)."""
    env = IndexAdvisorEnv(seed=seed, budget=2, horizon=HORIZON)
    env.set_task(task)
    env.set_selectivity_phase(selectivity_phase)
    total_reward, keeps, decisions = 0.0, 0, 0
    for _ in range(EVAL_EPISODES):
        obs = env.reset(task_id=task)
        for _ in range(HORIZON):
            mask = env.valid_actions()
            action, _ = agent.predict(env.obs_vector(obs), valid_actions=mask)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            keeps += int(action == ACTION_KEEP)
            decisions += 1
            if done:
                break
    return total_reward / max(1, decisions), keeps / max(1, decisions)


def run_pair(warm_task, drift_phase, seed, counterfactual):
    env = IndexAdvisorEnv(seed=seed, budget=2, horizon=HORIZON)
    env.set_task(warm_task)
    env.set_selectivity_phase(warm_task)

    agent = make_agent(env, seed)
    buffer = DatabaseAwarePriorityReplay(capacity=20000, alpha=0.6, beta=0.4)
    oracle = SimulatorOracle(env) if counterfactual else None

    for episode in range(PRE_EPISODES):
        train_episode(env, agent, buffer, oracle)

    pre_reward, pre_keep = evaluate(agent, warm_task, drift_phase, seed + 500)

    # Silent drift: templates and phase cue unchanged, selectivity inverted.
    env.set_selectivity_phase(drift_phase)
    rewards, keeps = [], []
    for episode in range(ADAPT_EPISODES):
        train_episode(env, agent, buffer, oracle)
        reward, keep = evaluate(agent, warm_task, drift_phase, seed + 500 + episode)
        rewards.append(reward)
        keeps.append(keep)

    return {
        "pre_reward": pre_reward,
        "pre_keep": pre_keep,
        "adapt_reward": float(np.mean(rewards)),
        "adapt_keep": float(np.mean(keeps)),
        "final_reward": float(rewards[-1]),
        "final_keep": float(keeps[-1]),
        "reward_curve": [float(v) for v in rewards],
        "keep_curve": [float(v) for v in keeps],
    }


def main():
    print("=" * 78)
    print("CWIL silent-drift ablation")
    print(f"pairs={PAIRS}  pre={PRE_EPISODES}  adapt={ADAPT_EPISODES}  "
          f"horizon={HORIZON}  seeds={SEEDS}")
    print("=" * 78)

    variants = {"Replay only": False, "Replay + CWIL": True}
    results = {name: {} for name in variants}

    for warm_task, drift_phase in PAIRS:
        label = f"{warm_task}->{drift_phase}"
        for name, use_cwil in variants.items():
            runs = [run_pair(warm_task, drift_phase, seed, use_cwil) for seed in SEEDS]
            results[name][label] = {
                key: float(np.mean([r[key] for r in runs]))
                for key in ("pre_reward", "pre_keep", "adapt_reward",
                            "adapt_keep", "final_reward", "final_keep")
            }
            results[name][label]["reward_curve"] = [
                float(np.mean([r["reward_curve"][i] for r in runs]))
                for i in range(ADAPT_EPISODES)
            ]
            results[name][label]["keep_curve"] = [
                float(np.mean([r["keep_curve"][i] for r in runs]))
                for i in range(ADAPT_EPISODES)
            ]
            row = results[name][label]
            print(f"  {label}  {name:14s} zero-shot KEEP={row['pre_keep']:.3f}  "
                  f"adapt KEEP={row['adapt_keep']:.3f}  final KEEP={row['final_keep']:.3f}  "
                  f"adapt reward={row['adapt_reward']:+.3f}")

    summary = {}
    for name in variants:
        labels = [f"{w}->{d}" for w, d in PAIRS]
        summary[name] = {
            key: float(np.mean([results[name][label][key] for label in labels]))
            for key in ("pre_keep", "adapt_keep", "final_keep", "adapt_reward")
        }

    print("\nSummary over drift pairs:")
    for name, values in summary.items():
        print(f"  {name:14s} adapt KEEP={values['adapt_keep']:.3f}  "
              f"final KEEP={values['final_keep']:.3f}  "
              f"adapt reward={values['adapt_reward']:+.3f}")

    deltas = {key: summary["Replay + CWIL"][key] - summary["Replay only"][key]
              for key in summary["Replay only"]}
    print("\n  CWIL deltas: " + "  ".join(f"{k}={v:+.3f}" for k, v in deltas.items()))

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/cwil_ablation.json", "w") as handle:
        json.dump({
            "config": {"pairs": PAIRS, "pre_episodes": PRE_EPISODES,
                       "adapt_episodes": ADAPT_EPISODES, "horizon": HORIZON,
                       "seeds": SEEDS},
            "results": results,
            "summary": summary,
            "cwil_deltas": deltas,
        }, handle, indent=2)
    print("\nExported experiments/results/cwil_ablation.json")
    return results


if __name__ == "__main__":
    main()
