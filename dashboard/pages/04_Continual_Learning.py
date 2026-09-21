import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Continual Learning", page_icon="🧠", layout="wide")

st.title("🧠 Continual Learning Engine")
st.markdown("Monitor the background training process. When workload drift is detected, the Online Scheduler triggers an asynchronous backpropagation update using samples from the Priority Replay Buffer, regularized by EWC.")

st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Workload Phase", "Phase 2")
col2.metric("Model Version", "v2.4.1")
col3.metric("Total Iterations", "14,020")
col4.metric("Learning Rate", "1e-4")

st.divider()

st.subheader("Live Training Progress")
st.markdown("""
<style>
.tooltip { position: relative; display: inline-block; border-bottom: 1px dotted #00ffaa; color: #00ffaa; cursor: help; }
.tooltip .tooltiptext { visibility: hidden; width: 250px; background-color: #333; color: #fff; text-align: left; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -125px; opacity: 0; transition: opacity 0.3s; font-size: 0.9em; }
.tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
The total loss is a combination of:
1. **Replay Loss**: Cross-entropy loss on the batches sampled from the Replay Buffer.
2. **EWC Loss**: The penalty applied for moving weights too far from the optimal weights of Phase 1.
""")

if "training_data" not in st.session_state:
    st.session_state.training_data = pd.DataFrame(columns=["Iteration", "Replay Loss", "EWC Loss", "Total Loss"])
    
chart_placeholder = st.empty()
metrics_placeholder = st.empty()

if st.button("▶️ Simulate Background Training Run"):
    st.session_state.training_data = pd.DataFrame(columns=["Iteration", "Replay Loss", "EWC Loss", "Total Loss"])
    
    for i in range(1, 51):
        # Generate realistic looking loss curves
        replay_loss = 2.5 * np.exp(-0.05 * i) + np.random.normal(0, 0.1)
        ewc_loss = 0.5 * (1 - np.exp(-0.1 * i)) + np.random.normal(0, 0.05)
        total_loss = replay_loss + ewc_loss
        
        new_row = pd.DataFrame({
            "Iteration": [i],
            "Replay Loss": [max(0.1, replay_loss)],
            "EWC Loss": [max(0.01, ewc_loss)],
            "Total Loss": [total_loss]
        })
        
        st.session_state.training_data = pd.concat([st.session_state.training_data, new_row], ignore_index=True)
        
        # Plot
        chart_data = st.session_state.training_data.set_index("Iteration")
        chart_placeholder.line_chart(chart_data)
        
        with metrics_placeholder.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("Replay Loss", f"{replay_loss:.3f}")
            c2.metric("EWC Loss", f"{ewc_loss:.3f}")
            c3.metric("Total Loss", f"{total_loss:.3f}")
            
        time.sleep(0.05)
        
    st.success("Background training iteration complete. Model weights updated.")
else:
    if not st.session_state.training_data.empty:
        chart_data = st.session_state.training_data.set_index("Iteration")
        chart_placeholder.line_chart(chart_data)
        
        last_row = st.session_state.training_data.iloc[-1]
        with metrics_placeholder.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("Replay Loss", f"{last_row['Replay Loss']:.3f}")
            c2.metric("EWC Loss", f"{last_row['EWC Loss']:.3f}")
            c3.metric("Total Loss", f"{last_row['Total Loss']:.3f}")
    else:
        st.info("Click the button above to simulate a background training run.")
