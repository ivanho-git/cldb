import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Replay Buffer", page_icon="📚", layout="wide")

st.title("📚 Database-Aware Priority Replay Buffer")
st.markdown("""
<style>
.tooltip { position: relative; display: inline-block; border-bottom: 1px dotted #00ffaa; color: #00ffaa; cursor: help; }
.tooltip .tooltiptext { visibility: hidden; width: 250px; background-color: #333; color: #fff; text-align: left; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -125px; opacity: 0; transition: opacity 0.3s; font-size: 0.9em; }
.tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
The Replay Buffer stores representative database workloads so that the model can revisit previous optimization experiences during continual learning. 
**This prevents catastrophic forgetting.**
Our novel <span class="tooltip">Database-Aware Priority Sampling<span class="tooltiptext">Prioritizes sampling experiences based on latency impact and workload rarity rather than generic ML loss.</span></span> ensures that rare but critical analytical queries are not forgotten.
""", unsafe_allow_html=True)

st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Size", "542 / 1000", delta="+12 recently added")
col2.metric("Rare Workloads", "42", delta="Priority 2.0x")
col3.metric("Common Workloads", "500", delta="Priority 0.5x", delta_color="inverse")
col4.metric("Avg Sampling Frequency", "4.2 times/exp")

st.divider()

col_hist1, col_hist2 = st.columns(2)

# Generate mock data for distributions
np.random.seed(42)
priority_data = np.random.exponential(scale=2.0, size=542)
priority_data = np.clip(priority_data, 0, 10)

workload_freq = np.random.lognormal(mean=2.0, sigma=1.0, size=542)

with col_hist1:
    st.subheader("Priority Histogram")
    st.caption("Distribution of sampling priorities across all stored experiences.")
    hist_values = np.histogram(priority_data, bins=20, range=(0,10))[0]
    st.bar_chart(pd.DataFrame({"Count": hist_values}))

with col_hist2:
    st.subheader("Replay Distribution (Importance vs Frequency)")
    st.caption("Shows how rare workloads (left) are artificially boosted in importance compared to common ones (right).")
    chart_data = pd.DataFrame({
        "Workload Frequency": workload_freq,
        "Priority Score": priority_data
    })
    st.scatter_chart(chart_data, x="Workload Frequency", y="Priority Score", size=20, color="#00ffaa")

st.divider()
st.subheader("Stored Experiences")
st.caption("Expand any row to see why the experience is important and view its internal representation.")

mock_experiences = [
    {
        "id": "exp_892",
        "sql": "SELECT c.region, SUM(o.total_amount) FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.region;",
        "type": "Analytics (Rare)",
        "priority": 8.4,
        "reward": 1.95,
        "action": "CREATE_INDEX",
        "timestamp": "2026-07-29 14:22:10",
        "importance": "High priority because this is a rare monthly analytical query. Dropping its associated index would cause a massive latency regression (measured Reward: +1.95)."
    },
    {
        "id": "exp_891",
        "sql": "SELECT * FROM orders WHERE customer_id = 42;",
        "type": "OLTP (Common)",
        "priority": 1.2,
        "reward": 0.45,
        "action": "KEEP",
        "timestamp": "2026-07-29 14:22:05",
        "importance": "Low priority. This is a very common point lookup. The model has seen it thousands of times, so its sampling weight is artificially decayed."
    },
    {
        "id": "exp_890",
        "sql": "SELECT p.name, p.price FROM products p WHERE p.category = 'Electronics';",
        "type": "OLTP (Common)",
        "priority": 2.1,
        "reward": 0.8,
        "action": "CREATE_INDEX",
        "timestamp": "2026-07-29 14:21:40",
        "importance": "Medium priority. Common query but index creation yielded a moderate reward."
    }
]

for exp in mock_experiences:
    with st.expander(f"**{exp['timestamp']}** | {exp['type']} | Priority: {exp['priority']} | Reward: {exp['reward']}"):
        st.code(exp["sql"], language="sql")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Action Taken", exp["action"])
        c2.metric("Reward Computed", f"+{exp['reward']}")
        c3.metric("Priority Score", exp["priority"])
        
        st.markdown("##### Why is this important?")
        st.info(exp["importance"])
        
        st.markdown("##### Internal Semantic Embedding (Preview)")
        st.caption("First 10 dimensions of the 384-d vector space")
        st.code(f"[{np.random.rand():.4f}, {np.random.rand():.4f}, {np.random.rand():.4f}, ...]", language="json")
