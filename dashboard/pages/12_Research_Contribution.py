import streamlit as st

st.set_page_config(page_title="Research Contribution", page_icon="🎓", layout="wide")

st.title("🎓 Research Contribution")

st.markdown("""
<style>
.tooltip { position: relative; display: inline-block; border-bottom: 1px dotted #00ffaa; color: #00ffaa; cursor: help; }
.tooltip .tooltiptext { visibility: hidden; width: 250px; background-color: #333; color: #fff; text-align: left; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -125px; opacity: 0; transition: opacity 0.3s; font-size: 0.9em; }
.tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
</style>
""", unsafe_allow_html=True)

st.markdown("This section details the formal academic contributions of the CLDB architecture and explicitly contrasts it against existing state-of-the-art database optimizers.")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### ❌ The Problem")
    st.info("""
    Modern autonomous databases successfully use Deep Reinforcement Learning (DRL) to recommend indexes. However, these models assume a stationary workload. When deployed in production environments where workloads evolve (e.g., shifting from daytime OLTP to nighttime Analytics), the DRL agent must be retrained. 
    
    Standard fine-tuning causes **Catastrophic Forgetting**—the model completely overwrites its knowledge of the OLTP workload to learn the Analytics workload. When daytime returns, the database suffers massive performance regressions.
    """)
    
with col2:
    st.markdown("### ✅ The Solution (CLDB)")
    st.success("""
    CLDB introduces a **Database-Aware Continual Learning Framework** operating as an optimization layer on top of PostgreSQL. It enables the DRL agent to sequentially learn new workload phases indefinitely without forgetting how to optimize previously seen workloads.
    
    This is achieved through a hybrid approach combining dynamic memory replay with structural parameter regularization.
    """)
    
st.divider()

st.header("✨ Key Novelties")

st.markdown("#### 1. Database-Aware Priority Replay Buffer")
st.markdown("Traditional ML replay buffers use Temporal Difference (TD) error to prioritize samples. **CLDB's novelty** is prioritizing samples based on *database-specific metrics*: raw latency impact, storage cost penalties, and embedding rarity. This ensures that a rare, highly expensive analytical query is replayed frequently to prevent its index from being erroneously dropped.")

st.markdown("#### 2. Native EWC for Workload Phases")
st.markdown("Elastic Weight Consolidation (EWC) usually requires strictly delineated 'Tasks'. Databases do not have tasks; they have continuous streams of queries. **CLDB's novelty** is coupling EWC with an unsupervised Drift Detector. The detector automatically identifies phase boundaries in the embedding space and snapshots the Fisher Information Matrix dynamically.")

st.markdown("#### 3. Adaptive Drift Detection")
st.markdown("Instead of retraining the model on every query (which causes extreme overhead) or retraining on a fixed schedule (which misses sudden workload shifts), CLDB tracks the cosine distance of the incoming workload centroid. Training is mathematically triggered only when the underlying distribution shifts.")

st.markdown("#### 4. Online Asynchronous Continual Learning")
st.markdown("CLDB decouples the critical execution path from the learning path. The DRL inference happens in milliseconds, while the Continual Learning updates execute asynchronously in background threads, ensuring 0% blocking overhead on PostgreSQL query execution.")

st.divider()

st.header("⚖️ How this differs from traditional database optimizers")

comparison_data = {
    "Feature": ["Optimization Strategy", "Workload Adaptability", "Knowledge Retention", "Overhead"],
    "PostgreSQL Optimizer": ["Static Heuristics / Cost Models", "None (Blindly executes)", "N/A", "Near Zero"],
    "Standard ML Optimizer (SageDB, etc)": ["Deep Reinforcement Learning", "Requires full offline retraining", "Catastrophic Forgetting", "High (Blocks during training)"],
    "CLDB (Ours)": ["Deep RL + Vector DB Context", "Online Adaptive", "Maintains performance via EWC+Replay", "Low (Asynchronous Background)"]
}

import pandas as pd
st.table(pd.DataFrame(comparison_data).set_index("Feature"))
