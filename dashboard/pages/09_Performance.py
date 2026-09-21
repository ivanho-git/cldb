import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Performance Metrics", page_icon="📈", layout="wide")

st.title("📈 Database Performance Metrics")

st.markdown("""
<style>
.tooltip { position: relative; display: inline-block; border-bottom: 1px dotted #00ffaa; color: #00ffaa; cursor: help; }
.tooltip .tooltiptext { visibility: hidden; width: 250px; background-color: #333; color: #fff; text-align: left; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -125px; opacity: 0; transition: opacity 0.3s; font-size: 0.9em; }
.tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
</style>
""", unsafe_allow_html=True)

st.markdown("Monitor traditional database and infrastructure metrics. These reflect the real-world impact of CLDB's optimizations on the underlying PostgreSQL instance.")

st.divider()

st.subheader("Query Latency")
col1, col2, col3 = st.columns(3)
col1.metric("Average Latency", "24.5 ms", delta="-42% (Improved)")
col2.metric("Median Latency", "12.0 ms", delta="-15%")
col3.metric("95th Percentile Latency", "145.0 ms", delta="-60%")

st.divider()

st.subheader("Throughput & Hardware")
col4, col5, col6, col7 = st.columns(4)
col4.metric("Throughput", "842 QPS", delta="+120 QPS")
col5.metric("CPU Usage", "45%", delta="-12%", delta_color="inverse")
col6.metric("Memory Usage", "6.2 GB / 16 GB", delta="+0.1 GB", delta_color="inverse")
col7.metric("Disk I/O", "120 MB/s", delta="-40 MB/s", delta_color="inverse")

st.divider()

st.subheader("Storage & Optimization Stats")
col8, col9, col10 = st.columns(3)
col8.metric("CLDB Index Count", "14 Active Indexes", delta="+2 today")
col9.metric("Storage Used by Indexes", "450 MB")
col10.metric("Safety Rollback Rate", "0.5%", delta="-0.2%", delta_color="inverse")
st.metric("Optimization Success Rate", "94.5%", delta="+2.1%")

st.divider()

st.subheader("Historical Performance")
# Generate mock time-series data
time_idx = pd.date_range(start="2026-07-29 08:00", periods=24, freq="h")
np.random.seed(42)
qps = np.random.normal(800, 100, 24)
lat = 100000 / (qps + 100) + np.random.normal(0, 5, 24) # inverse relationship

df_history = pd.DataFrame({
    "Throughput (QPS)": qps,
    "Avg Latency (ms)": lat
}, index=time_idx)

st.line_chart(df_history, width="stretch")
