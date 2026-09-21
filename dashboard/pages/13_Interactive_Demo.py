import streamlit as st
import time

st.set_page_config(page_title="Interactive Demonstration", page_icon="🕹️", layout="wide")

st.title("🕹️ Interactive Demonstration")

st.markdown("""
<style>
.tooltip { position: relative; display: inline-block; border-bottom: 1px dotted #00ffaa; color: #00ffaa; cursor: help; }
.tooltip .tooltiptext { visibility: hidden; width: 250px; background-color: #333; color: #fff; text-align: left; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -125px; opacity: 0; transition: opacity 0.3s; font-size: 0.9em; }
.tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
.demo-console { background-color: #111; color: #00ffaa; font-family: monospace; padding: 20px; border-radius: 8px; height: 300px; overflow-y: auto; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

st.markdown("Use these interactive macros to simulate specific workload scenarios and observe how CLDB reacts in real-time.")

st.divider()

col1, col2, col3, col4 = st.columns(4)

scenario = None

with col1:
    if st.button("▶️ Run Stable Workload (OLTP)", width="stretch"):
        scenario = "stable"
    if st.button("▶️ Run Sudden Shift (Analytics)", width="stretch"):
        scenario = "shift"

with col2:
    if st.button("▶️ Run Returning Workload", width="stretch"):
        scenario = "return"
    if st.button("▶️ Run Catastrophic Forgetting Demo", width="stretch"):
        scenario = "forgetting"

with col3:
    if st.button("▶️ Run Full Continual Learning Demo", width="stretch"):
        scenario = "cl_full"

st.divider()

st.subheader("Live Simulation Console")

console_placeholder = st.empty()
progress_bar = st.progress(0)

def animate_scenario(lines):
    console_text = ""
    for i, line in enumerate(lines):
        progress_bar.progress(int(((i + 1) / len(lines)) * 100))
        console_text += f"> {line}<br>"
        console_placeholder.markdown(f'<div class="demo-console">{console_text}</div>', unsafe_allow_html=True)
        time.sleep(0.6)
    
    console_text += f"<br>> <b>✅ Simulation Complete</b>"
    console_placeholder.markdown(f'<div class="demo-console">{console_text}</div>', unsafe_allow_html=True)

if not scenario:
    console_placeholder.markdown('<div class="demo-console">> Ready... Select a scenario above to begin execution.</div>', unsafe_allow_html=True)
elif scenario == "stable":
    lines = [
        "Initializing Stable OLTP Workload...",
        "Executing 500 Point-Lookup Queries (Phase 1)...",
        "Drift Detector: Distance 0.05 (Below Threshold 0.3)",
        "CLDB Action: MAINTAIN_INDEX on orders.customer_id",
        "Reward: +0.0 (No action needed, stable latency 12ms)",
        "Replay Buffer: Added experiences (Priority 0.1)",
        "Online Scheduler: No drift detected. Skipping background training."
    ]
    animate_scenario(lines)
elif scenario == "shift":
    lines = [
        "Initializing Sudden Shift to Heavy Analytics Workload...",
        "Executing 150 complex GROUP BY / JOIN Queries (Phase 2)...",
        "Drift Detector: Distance 0.42 (Exceeds Threshold 0.3!)",
        "ALERT: Workload Drift Detected!",
        "EWC Engine: Snapshotting Fisher Information Matrix for Phase 1...",
        "CLDB Action: CREATE_INDEX on order_items.product_id",
        "Reward: +1.85 (Latency dropped from 1200ms -> 45ms)",
        "Replay Buffer: Added experiences (Priority 9.2)",
        "Online Scheduler: Triggering asynchronous Backpropagation..."
    ]
    animate_scenario(lines)
elif scenario == "return":
    lines = [
        "Initializing Return to Workload (Phase 1)...",
        "Executing 500 Point-Lookup Queries...",
        "Drift Detector: Distance 0.35 (Exceeds Threshold 0.3)",
        "ALERT: Workload Drift Detected! Returning to historical centroid.",
        "CLDB Action: MAINTAIN_INDEX on orders.customer_id",
        "Observation: Index was preserved by EWC during Phase 2!",
        "Reward: +0.0 (Latency maintained at 12ms)",
        "Conclusion: Forward/Backward transfer successful."
    ]
    animate_scenario(lines)
elif scenario == "forgetting":
    lines = [
        "[SIMULATING STANDARD ML BASELINE WITHOUT REPLAY/EWC]",
        "Training on Phase 1 (OLTP)... Accuracy: 98%",
        "Training on Phase 2 (Analytics)... Accuracy: 96%",
        "Evaluating on Phase 1...",
        "Catastrophic Forgetting Detected! Phase 1 Accuracy: 22%",
        "Reason: Neural weights for Phase 1 were overwritten to minimize Phase 2 loss.",
        "Result: Severe latency regression when returning to Phase 1."
    ]
    animate_scenario(lines)
elif scenario == "cl_full":
    lines = [
        "Running Full CLDB Pipeline...",
        "Phase 1: OLTP | Accuracy: 96% | EWC Snapshotted",
        "Phase 2: Analytics | Accuracy 94% | Replay Buffer active",
        "Phase 3: Evolving | Accuracy 95% | Priority sampling active",
        "Evaluating Phase 1 Retention...",
        "Result: Phase 1 Accuracy retained at 95%!",
        "Forgetting Measure: 0.01 (Near Zero)",
        "Database is safely learning sequentially without human intervention."
    ]
    animate_scenario(lines)
