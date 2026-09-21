import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Drift Detection", page_icon="📡", layout="wide")

st.title("📡 Workload Drift Detection")

# Custom CSS
st.markdown("""
<style>
.tooltip { position: relative; display: inline-block; border-bottom: 1px dotted #00ffaa; color: #00ffaa; cursor: help; }
.tooltip .tooltiptext { visibility: hidden; width: 250px; background-color: #333; color: #fff; text-align: left; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -125px; opacity: 0; transition: opacity 0.3s; font-size: 0.9em; }
.tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
**Concept Drift** in databases occurs when the types of queries being executed change significantly. For example, migrating from heavy `INSERT` operations (OLTP) during the day to complex `JOIN` reports (Analytics) at night.

**Adaptive Continual Learning**: Training the neural network on every single query is computationally expensive and causes overfitting. Therefore, CLDB uses an <span class="tooltip">Adaptive Online Scheduler<span class="tooltiptext">A background monitor that measures the Cosine Distance of incoming query embeddings against the historical centroid.</span></span>.

> **Training is triggered ONLY when significant workload drift is detected.**
""", unsafe_allow_html=True)

st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Current Drift Score", "0.41", delta="Above Threshold", delta_color="inverse")
col2.metric("Drift Threshold", "0.30")
col3.metric("Last Drift Event", "12 mins ago")

st.divider()

col_timeline, col_scatter = st.columns(2)

with col_timeline:
    st.subheader("Drift Score Timeline")
    st.caption("Cosine distance of the sliding window centroid compared to the baseline.")
    
    # Generate drift score timeline
    time_idx = np.arange(100)
    base_drift = 0.1 + np.random.normal(0, 0.05, 100)
    # Simulate a drift event at index 70
    base_drift[70:] += 0.25 + np.random.normal(0, 0.05, 30)
    base_drift = np.clip(base_drift, 0, 1)
    
    df_drift = pd.DataFrame({
        "Time": time_idx,
        "Drift Score": base_drift,
        "Threshold": 0.30
    }).set_index("Time")
    
    st.line_chart(df_drift, color=["#00aaff", "#ff4b4b"])
    st.markdown("Notice the spike at t=70. This cross over the threshold triggered a Phase Shift, EWC registration, and model training.")

with col_scatter:
    st.subheader("Embedding Distribution (Centroid Shift)")
    st.caption("A 2D PCA projection of the 384-dimensional query embeddings.")
    
    # Generate mock 2D embeddings for two phases
    np.random.seed(42)
    phase1_x = np.random.normal(0, 1, 100)
    phase1_y = np.random.normal(0, 1, 100)
    
    phase2_x = np.random.normal(4, 1.5, 100)
    phase2_y = np.random.normal(4, 1.5, 100)
    
    df_scatter = pd.DataFrame({
        "X": np.concatenate([phase1_x, phase2_x]),
        "Y": np.concatenate([phase1_y, phase2_y]),
        "Phase": ["Phase 1 (Stable)"] * 100 + ["Phase 2 (Drifted)"] * 100
    })
    
    st.scatter_chart(df_scatter, x="X", y="Y", color="Phase")

st.divider()

st.subheader("Recent Drift Events")
events = pd.DataFrame([
    {"Timestamp": "2026-07-29 14:22:10", "Score": 0.41, "Action": "Phase 2 Initiated. EWC Snapshotted. Training started."},
    {"Timestamp": "2026-07-28 09:00:00", "Score": 0.35, "Action": "Phase 1 Initiated. EWC Snapshotted. Training started."},
])
st.dataframe(events, width="stretch", hide_index=True)
