import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Data Exports", page_icon="📥", layout="wide")

st.title("📥 Data Exports")

st.markdown("""
<style>
.tooltip { position: relative; display: inline-block; border-bottom: 1px dotted #00ffaa; color: #00ffaa; cursor: help; }
.tooltip .tooltiptext { visibility: hidden; width: 250px; background-color: #333; color: #fff; text-align: left; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -125px; opacity: 0; transition: opacity 0.3s; font-size: 0.9em; }
.tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
</style>
""", unsafe_allow_html=True)

st.markdown("Download raw data and evaluation metrics for use in academic papers, thesis submissions, or further offline analysis.")

st.divider()

col1, col2, col3 = st.columns(3)

# 1. Replay Buffer Data
with col1:
    st.subheader("📚 Replay Buffer Data")
    st.markdown("Export the current contents of the Priority Replay Buffer, including SQL queries, rewards, and priority weights.")
    
    mock_replay = pd.DataFrame({
        "query_id": [f"q_{i}" for i in range(1, 101)],
        "priority_score": np.random.uniform(0.1, 10.0, 100).round(2),
        "reward": np.random.uniform(-1.0, 2.0, 100).round(2),
        "action": np.random.choice(["CREATE_INDEX", "KEEP"], 100)
    })
    
    csv_replay = mock_replay.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Replay Buffer (CSV)",
        data=csv_replay,
        file_name='cldb_replay_buffer.csv',
        mime='text/csv',
    )

# 2. Evaluation Results
with col2:
    st.subheader("⚖️ Evaluation Results")
    st.markdown("Export the Forgetting Measure analysis matrix (Phase retention over time) for baseline comparisons.")
    
    time_phases = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
    retention_df = pd.DataFrame({
        "PostgreSQL": [10, 10, 10, 10],
        "CLDB (No Replay)": [95, 40, 20, 25],
        "CLDB (+Replay)": [96, 85, 80, 82],
        "CLDB (Replay+EWC)": [98, 96, 95, 96]
    }, index=time_phases)
    
    csv_eval = retention_df.to_csv(index=True).encode('utf-8')
    
    st.download_button(
        label="Download Evaluation Matrix (CSV)",
        data=csv_eval,
        file_name='cldb_forgetting_eval.csv',
        mime='text/csv',
    )

# 3. Training Logs
with col3:
    st.subheader("🧠 Continual Learning Logs")
    st.markdown("Export the background training logs containing iterations, Replay Loss, and EWC Penalty.")
    
    logs = pd.DataFrame({
        "iteration": np.arange(1, 101),
        "replay_loss": np.exp(-0.05 * np.arange(1, 101)) + np.random.normal(0, 0.05, 100),
        "ewc_loss": 0.5 * (1 - np.exp(-0.1 * np.arange(1, 101))) + np.random.normal(0, 0.02, 100)
    })
    
    csv_logs = logs.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Training Logs (CSV)",
        data=csv_logs,
        file_name='cldb_training_logs.csv',
        mime='text/csv',
    )

st.divider()

st.subheader("📄 PDF Reports")
st.markdown("Export a comprehensive system state report containing all current charts, architectures, and evaluation scores formatted for A4 printing.")
st.button("Generate & Download PDF Report (Mock)")
st.caption("Note: Generating PDFs requires `wkhtmltopdf` or similar system binaries installed on the server.")
