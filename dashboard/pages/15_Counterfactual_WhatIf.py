"""Dashboard page for Counterfactual What-If Learning (CWIL)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import (agent_metric_rows, agent_names, load_cwil_ablation,  # noqa: E402
                  load_results, r_matrix, tasks)

st.set_page_config(page_title="Counterfactual What-If", page_icon="🔀", layout="wide")

st.title("🔀 Counterfactual What-If Learning (CWIL)")

st.markdown("""
CLDB is evaluated at every decision by a **what-if oracle**: a cost model offline,
or `EXPLAIN` over hypothetical indexes online. Instead of learning only from the
action it actually took, the agent rolls out **every structurally valid action**
from an identical state and is trained to rank those actions by their observed
outcomes with a pairwise preference loss.

That is the mechanism that teaches the agent a *non-selective* index is harmful
without first paying for the regression. Replay answers *"what happened when I
acted?"*; CWIL answers *"what would have happened if I had acted differently?"*.
The two objectives are complementary and trained jointly.
""")

st.divider()

results = load_results()
if results is None:
    st.warning("No benchmark artifact found. Run `python experiments/run_rl_evaluation.py` first.")
    st.stop()

rows = agent_metric_rows(results)
df = pd.DataFrame(rows).set_index("agent")
st.subheader("Measured continual-learning metrics")
st.dataframe(df, width="stretch")
st.caption("Conflict = mean final reward on the inverted-selectivity phases (2, 4, 5), "
           "where CREATE is harmful. F = average forgetting (lower is better).")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Conflict-phase reward")
    st.bar_chart(df[["Conflict"]], height=320)
    st.caption("Higher (closer to zero) means the agent learned to KEEP when indexing does not pay off.")

with col2:
    st.subheader("Average forgetting")
    st.bar_chart(df[["F"]], height=320)
    st.caption("Lower is better. CWIL is expected to complement, not replace, replay and EWC.")

st.divider()

st.subheader("Final reward per phase")
agents = agent_names(results)
selected = st.multiselect("Agents", agents, default=agents[:2])
phase_ids = tasks(results)
chart = {}
for agent in selected:
    final = results["agents"][agent].get("final_mean", [])
    if final:
        chart[agent] = pd.Series(final, index=[f"T{t}" for t in phase_ids])
if chart:
    st.line_chart(pd.DataFrame(chart), height=340)
    st.caption("Phases T2, T4 and T5 are the conflict phases. A robust continual agent "
               "should not collapse to the KEEP level on them.")

st.divider()

st.subheader("Evaluation matrix R")
agent = st.selectbox("Agent", agents, index=0)
matrix = r_matrix(results, agent)
if matrix:
    frame = pd.DataFrame(matrix, index=[f"after T{t}" for t in phase_ids],
                         columns=[f"eval T{t}" for t in phase_ids])
    st.dataframe(frame.style.background_gradient(cmap="viridis"), width="stretch")
    st.caption("Row i = policy after training phase i; column j = reward on phase j. "
               "Bright cells on earlier columns indicate retained competence.")

st.divider()

st.header("Silent-drift ablation")
st.markdown("""
The mechanism is isolated by warm-starting on a selective workload, then
inverting the predicate's selectivity **without changing the templates or the
phase cue**. The agent must discover from reward alone that indexing has
stopped paying off. Curves are the greedy evaluation after each post-drift
episode, averaged over the three drift pairs.
""")

ablation = load_cwil_ablation()
if ablation is None:
    st.info("No CWIL ablation artifact found. Run `python experiments/run_cwil_ablation.py`.")
else:
    summary = pd.DataFrame(ablation["summary"]).T
    st.dataframe(summary, width="stretch")
    st.caption("adapt/final KEEP are fractions of greedy decisions that keep the index "
               "configuration unchanged; adapt reward is the mean over the adaptation window.")

    pairs = list(ablation["results"]["Replay only"].keys())
    reward_curves, keep_curves = {}, {}
    for variant in ("Replay only", "Replay + CWIL"):
        reward_curves[variant] = np.mean(
            [ablation["results"][variant][pair]["reward_curve"] for pair in pairs], axis=0)
        keep_curves[variant] = np.mean(
            [ablation["results"][variant][pair]["keep_curve"] for pair in pairs], axis=0)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Post-drift reward")
        st.line_chart(pd.DataFrame(reward_curves,
                                   index=[f"ep {i + 1}" for i in range(len(reward_curves['Replay only']))]),
                      height=320)
        st.caption("A flat curve from episode 1 means the agent recovered without a dip.")
    with col2:
        st.subheader("Post-drift KEEP rate")
        st.line_chart(pd.DataFrame(keep_curves,
                                   index=[f"ep {i + 1}" for i in range(len(keep_curves['Replay only']))]),
                      height=320)
        st.caption("CWIL starts higher because it observes the counterfactual cost before acting.")
