import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import agent_metric, best_agent, load_results, results_path  # noqa: E402

st.set_page_config(
    page_title="CLDB Research Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .stMetric {
        background-color: rgba(25, 25, 25, 0.5); 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 4px solid #00ffaa; 
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #00ffaa;
        color: #00ffaa;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 250px;
        background-color: #333;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -125px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.9em;
        line-height: 1.4;
        font-weight: normal;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .card {
        border: 1px solid #444;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        background: rgba(30,30,30,0.4);
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ CLDB")
st.subheader("Continual Learning Database Engine")

st.markdown("""
<div class="card">
    <h3>Welcome to the CLDB Demonstration Dashboard</h3>
    <p style="font-size: 1.1em; line-height: 1.6;">
        <b>CLDB continuously learns from database workloads and improves query optimization over time while retaining previous optimization knowledge.</b>
    </p>
    <p>
        Traditional database optimizers like PostgreSQL rely on static heuristics. CLDB acts as an intelligent optimization layer 
        that adapts dynamically using <span class="tooltip">Deep Reinforcement Learning<span class="tooltiptext">Neural networks that learn optimal actions (like index creation) by receiving rewards for reducing latency.</span></span>, 
        protected by <span class="tooltip">Continual Learning<span class="tooltiptext">Techniques that allow AI to learn new workloads without forgetting how to optimize old ones.</span></span> mechanisms.
    </p>
    <p>
        Use the sidebar to explore the architecture, view live query processing, understand the continual learning engine, 
        and evaluate performance metrics against baseline PostgreSQL.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()
st.header("Measured System Status")

results = load_results()
if results is None:
    st.warning(
        "No benchmark artifact found. Run `python experiments/run_rl_evaluation.py` "
        "to populate `experiments/results/cl_results.json`; this page renders those "
        "measurements rather than placeholder values."
    )
else:
    best = best_agent(results, "Conflict")
    config = results.get("config", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best conflict-phase reward", f"{agent_metric(results, best, 'Conflict'):.3f}",
                  help="Mean final reward on the inverted-selectivity phases (2, 4, 5).")
    with col2:
        st.metric("Best ACC", f"{agent_metric(results, best, 'ACC'):.3f}",
                  help="Average final performance over all phases.")
    with col3:
        st.metric("Best forgetting F", f"{agent_metric(results, best, 'F'):.3f}",
                  help="Average drop from each phase's peak to final performance.")
    with col4:
        st.metric("Seeds", f"{len(config.get('seeds', []))}",
                  help="Seeds the reported metrics are averaged over.")

    st.divider()
    col5, col6, col7 = st.columns(3)
    with col5:
        st.markdown(f"**Leading agent**  \n{best}")
    with col6:
        st.markdown(f"**Benchmark**  \n{len(config.get('tasks', []))} phases · "
                    f"{config.get('episodes', '?')} episodes/task · horizon {config.get('horizon', '?')}")
    with col7:
        st.markdown(f"**Config**  \nIndex budget {config.get('budget', '?')} · "
                    f"CWIL weight via `CLDB_CF_WEIGHT`")
    st.caption(f"Source: `{results_path()}`")