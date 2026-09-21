import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Reward Engine", page_icon="🏆", layout="wide")

st.title("🏆 Multi-Objective Reward Engine")

st.markdown("""
<style>
.tooltip { position: relative; display: inline-block; border-bottom: 1px dotted #00ffaa; color: #00ffaa; cursor: help; }
.tooltip .tooltiptext { visibility: hidden; width: 250px; background-color: #333; color: #fff; text-align: left; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -125px; opacity: 0; transition: opacity 0.3s; font-size: 0.9em; }
.tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
The Reward Engine evaluates the real-world impact of the model's optimization decisions (e.g., `CREATE_INDEX`) by measuring the database's performance before and after the change.

Unlike standard RL environments, database optimization requires a **Multi-Objective Reward Function**. We cannot just reward raw latency improvement, as creating too many indexes consumes disk space and slows down `INSERT` operations. We also heavily penalize regressions.
""")

st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Reward", "+1.45", delta="Positive Impact")
col2.metric("Avg Latency Improvement", "85ms")
col3.metric("Resource Cost Penalty", "-0.10", delta_color="inverse")
col4.metric("Regression Penalty Weight", "2.0x")

st.divider()

st.subheader("Anatomy of the Latest Reward")
st.markdown("For the query: `SELECT * FROM orders WHERE status = 'PENDING';` | Action: `CREATE_INDEX`")

col_bar, col_explain = st.columns(2)

with col_bar:
    # Waterfall-like data
    components = pd.DataFrame({
        "Component": ["Latency Improvement", "Throughput Gain", "Resource (Disk) Penalty", "Regression Penalty", "Final Reward"],
        "Value": [1.20, 0.35, -0.10, 0.00, 1.45]
    })
    
    # We can use a simple bar chart
    st.bar_chart(components.set_index("Component"))

with col_explain:
    st.markdown("#### Reward Formula")
    st.latex(r"""
    R = \alpha \log\left(\frac{\text{Latency}_{before}}{\text{Latency}_{after}}\right) 
    - \beta (\text{Resource Cost}) 
    - \gamma (\text{Regression})
    """)
    
    st.markdown("""
    - **Latency Improvement (+1.20)**: The index reduced query time from 150ms to 15ms.
    - **Throughput Gain (+0.35)**: The reduced CPU load allowed the system to process more concurrent queries.
    - **Resource Penalty (-0.10)**: A flat penalty for consuming 15MB of disk space for the new index.
    - **Regression Penalty (0.00)**: No performance degradation occurred. (If latency had worsened, this would be doubled!).
    """)

st.divider()
st.subheader("Recent Rewards History")
history = pd.DataFrame([
    {"Action": "CREATE_INDEX", "Lat_Before": 150, "Lat_After": 15, "Raw_Delta": 135, "Penalty": -0.10, "Final Reward": 1.45},
    {"Action": "DROP_INDEX", "Lat_Before": 20, "Lat_After": 22, "Raw_Delta": -2, "Penalty": +0.10, "Final Reward": 0.05},
    {"Action": "CREATE_INDEX", "Lat_Before": 40, "Lat_After": 65, "Raw_Delta": -25, "Penalty": -0.10, "Final Reward": -1.80},
    {"Action": "KEEP", "Lat_Before": 12, "Lat_After": 12, "Raw_Delta": 0, "Penalty": 0.00, "Final Reward": 0.00},
])

st.dataframe(history.style.highlight_max(subset=["Final Reward"], color="green").highlight_min(subset=["Final Reward"], color="red"), width="stretch")
