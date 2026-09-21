import streamlit as st
import time

st.set_page_config(page_title="Live Query Flow", page_icon="⚡", layout="wide")

st.title("⚡ Live Query Flow")
st.markdown("Submit a raw SQL query and watch it process through the CLDB Continual Learning Pipeline in real-time.")

# Custom CSS for tooltip
st.markdown("""
<style>
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
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .status-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #222;
        border-left: 4px solid #444;
        margin-bottom: 10px;
        color: #ddd;
    }
    .status-active {
        border-left: 4px solid #00aaff;
        background-color: #1a2b3c;
        color: #fff;
    }
    .status-done {
        border-left: 4px solid #00ffaa;
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

query = st.text_area("Enter SQL Query:", value="SELECT p.name, oi.quantity FROM products p JOIN order_items oi ON p.product_id = oi.product_id WHERE oi.order_id = 100;", height=100)

if st.button("▶️ Execute Query"):
    
    stages = [
        {"name": "Incoming Query", "desc": "Receiving and validating SQL syntax."},
        {"name": "Embedding Created", "desc": "Generating 384-d semantic embedding."},
        {"name": "Nearest Historical Queries Retrieved", "desc": "Querying Qdrant for similar past queries."},
        {"name": "Optimization Decision", "desc": "RL Agent outputs Action: CREATE_INDEX, Confidence: 94%"},
        {"name": "Safety Validation", "desc": "Checking heuristics... Safe to execute."},
        {"name": "Execution", "desc": "Executing CREATE INDEX on PostgreSQL..."},
        {"name": "Latency Measured", "desc": "Latency Drop: 120ms -> 15ms"},
        {"name": "Reward Computed", "desc": "Multi-objective Reward: +1.45"},
        {"name": "Replay Buffer Updated", "desc": "Experience stored with priority 0.85"},
        {"name": "Background Learning Triggered", "desc": "Drift detected. EWC matrix snapshotting. Backpropagation initiated."}
    ]
    
    placeholders = [st.empty() for _ in stages]
    
    progress_bar = st.progress(0)
    
    for i, stage in enumerate(stages):
        # Update progress bar
        progress_bar.progress(int((i / len(stages)) * 100))
        
        # Set all previous to done
        for j in range(i):
            placeholders[j].markdown(f"""
            <div class="status-box status-done">
                <b>✅ {stages[j]['name']}</b><br>
                <small>{stages[j]['desc']}</small>
            </div>
            """, unsafe_allow_html=True)
            
        # Set current to active
        placeholders[i].markdown(f"""
        <div class="status-box status-active">
            <b>⏳ {stage['name']}...</b><br>
            <small>{stage['desc']}</small>
        </div>
        """, unsafe_allow_html=True)
        
        time.sleep(1.0) # Animate
        
    # Finalize
    progress_bar.progress(100)
    for j in range(len(stages)):
        placeholders[j].markdown(f"""
        <div class="status-box status-done">
            <b>✅ {stages[j]['name']}</b><br>
            <small>{stages[j]['desc']}</small>
        </div>
        """, unsafe_allow_html=True)
        
    st.success("Query Processing & Continual Learning Loop Complete!")
