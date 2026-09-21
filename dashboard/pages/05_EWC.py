import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Elastic Weight Consolidation", page_icon="🛡️", layout="wide")

st.title("🛡️ Elastic Weight Consolidation (EWC)")

# Custom CSS
st.markdown("""
<style>
.tooltip { position: relative; display: inline-block; border-bottom: 1px dotted #00ffaa; color: #00ffaa; cursor: help; }
.tooltip .tooltiptext { visibility: hidden; width: 250px; background-color: #333; color: #fff; text-align: left; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -125px; opacity: 0; transition: opacity 0.3s; font-size: 0.9em; }
.tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
</style>
""", unsafe_allow_html=True)

with st.expander("📖 Educational Primer: What is EWC?", expanded=True):
    st.markdown("""
    #### What is Catastrophic Forgetting?
    When a neural network learns a new task (e.g., optimizing Analytics queries), it brutally overwrites the weights it used to solve the previous task (e.g., optimizing OLTP queries). The model completely "forgets" how to optimize the old workload.

    #### What is EWC?
    Elastic Weight Consolidation (EWC) is an algorithm inspired by neuroscience. It identifies which specific weights in the neural network were most important for solving a previous workload phase. It then applies a mathematical penalty (like a spring) that makes it very hard to change those specific "protected" weights, while allowing unimportant weights to change freely to learn the new task.

    #### Why is EWC needed in CLDB?
    Databases undergo massive workload shifts (e.g., Black Friday sales vs End-of-month reporting). Without EWC, CLDB would optimize perfectly for Black Friday, but immediately destroy those optimizations once December begins. EWC ensures the model accumulates knowledge instead of overwriting it.
    """)

st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Penalty Value", "0.412")
col2.metric("Total Parameters", "148,224")
col3.metric("Protected Parameters", "12,410", delta="High Fisher Info")
col4.metric("Active Workload Phases", "2 (Phase 1, Phase 2)")

st.divider()

st.subheader("Fisher Information Matrix Statistics")
st.markdown('The <span class="tooltip">Fisher Information Matrix<span class="tooltiptext">Calculates the second derivative of the loss to determine how sensitive the network is to changes in each parameter.</span></span> dictates importance. Higher values mean the weight is highly critical to Phase 1.', unsafe_allow_html=True)

# Generate mock heatmaps using Pandas Styler
np.random.seed(42)
fisher_matrix = np.random.exponential(scale=1.0, size=(10, 10))
fisher_df = pd.DataFrame(fisher_matrix, columns=[f"N_{i}" for i in range(10)])

st.dataframe(
    fisher_df.style.background_gradient(cmap='viridis', axis=None).format("{:.2f}"),
    width="stretch"
)

st.divider()

st.subheader("Weight Changes (Before vs After New Task)")
st.markdown("Observe how weights with High Fisher Information (protected) barely change, while others change significantly to accommodate Phase 2.")

col_before, col_after = st.columns(2)

weights_before = np.random.normal(0, 1, size=(10, 10))
# Mask where fisher is high, keep weights same. Else change them.
mask = fisher_matrix > 1.5
weights_after = np.copy(weights_before)
weights_after[~mask] += np.random.normal(0, 0.8, size=np.sum(~mask))

with col_before:
    st.markdown("**Weights at End of Phase 1**")
    df_before = pd.DataFrame(weights_before)
    st.dataframe(df_before.style.background_gradient(cmap='coolwarm', axis=None, vmin=-2, vmax=2).format("{:.2f}"))

with col_after:
    st.markdown("**Weights at Current Phase (Phase 2)**")
    df_after = pd.DataFrame(weights_after)
    st.dataframe(df_after.style.background_gradient(cmap='coolwarm', axis=None, vmin=-2, vmax=2).format("{:.2f}"))
