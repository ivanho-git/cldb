import streamlit as st

st.set_page_config(page_title="System Architecture", page_icon="🏗️", layout="wide")

st.title("🏗️ System Architecture")
st.markdown("Click on any component in the pipeline below to understand its role in the Continual Learning Database Engine.")

# Data dictionary for the architecture components
ARCHITECTURE_INFO = {
    "Incoming Query": {
        "Purpose": "The entry point for the system. A raw SQL query submitted by the application or user.",
        "Input": "None",
        "Output": "Raw SQL String",
        "Algorithms": "N/A",
        "Research Contribution": "N/A"
    },
    "SQL Parser": {
        "Purpose": "Parses the raw SQL string into an Abstract Syntax Tree (AST) to extract structural features (e.g., tables accessed, WHERE clauses, JOINs).",
        "Input": "Raw SQL String",
        "Output": "QueryFeatures object (74-dimensional numerical array)",
        "Algorithms": "AST Traversal via sqlglot",
        "Research Contribution": "Transforms unstructured text into a structured schema-aware vector representation."
    },
    "Embedding Generator": {
        "Purpose": "Generates a semantic embedding of the SQL query to capture intent and semantic similarity.",
        "Input": "Raw SQL String",
        "Output": "384-dimensional dense vector",
        "Algorithms": "SentenceTransformer (all-MiniLM-L6-v2)",
        "Research Contribution": "Allows the model to generalize across structurally different but semantically similar queries."
    },
    "Qdrant (Vector DB)": {
        "Purpose": "Stores historical query embeddings and their optimization outcomes for rapid similarity search.",
        "Input": "384-dimensional embedding",
        "Output": "Top-K similar historical queries and their rewards",
        "Algorithms": "HNSW (Hierarchical Navigable Small World) Index",
        "Research Contribution": "Provides context to the decision engine, bridging purely parametric neural networks with non-parametric memory."
    },
    "Decision Engine": {
        "Purpose": "The core neural network that decides whether to create an index, drop an index, or do nothing.",
        "Input": "Query Features + Semantic Embedding + Context Vector",
        "Output": "Action (CREATE_INDEX, DROP_INDEX, KEEP) & Confidence Score",
        "Algorithms": "Deep Neural Network with Residual Connections",
        "Research Contribution": "Learns mapping from query distributions to optimal database physical designs."
    },
    "Safety Layer": {
        "Purpose": "Prevents the Decision Engine from executing harmful or overly expensive operations.",
        "Input": "Proposed Action, Confidence Score",
        "Output": "Boolean (Approve/Reject)",
        "Algorithms": "Heuristic bounds (max indexes, confidence thresholds, explain cost validation)",
        "Research Contribution": "Crucial for autonomous databases to prevent catastrophic regressions in production environments."
    },
    "PostgreSQL": {
        "Purpose": "The underlying relational database management system that physically executes the query and schema changes.",
        "Input": "SQL Query, DDL Commands",
        "Output": "Query Results",
        "Algorithms": "PostgreSQL Default Optimizer (Cost-based)",
        "Research Contribution": "CLDB acts as a smart layer *on top* of standard PostgreSQL, requiring no engine modifications."
    },
    "Performance Metrics": {
        "Purpose": "Measures the actual execution latency of the query before and after the optimization.",
        "Input": "Executed Query",
        "Output": "Latency (ms), Resource Usage",
        "Algorithms": "EXPLAIN ANALYZE, System timers",
        "Research Contribution": "Gathers the ground-truth data necessary to evaluate if the AI's decision was actually beneficial."
    },
    "Reward Calculation": {
        "Purpose": "Translates performance metrics into a numerical reward signal for the Reinforcement Learning agent.",
        "Input": "Latency Before, Latency After, Resource Cost",
        "Output": "Reward [-2.0, 2.0]",
        "Algorithms": "Multi-objective normalization, Regression penalty weighting",
        "Research Contribution": "Balances raw latency improvements against storage costs and penalizes catastrophic regressions."
    },
    "Replay Buffer": {
        "Purpose": "Stores representative database workloads so that the model can revisit previous optimization experiences.",
        "Input": "Experience Tuple (State, Action, Reward)",
        "Output": "Batches of Experience for Training",
        "Algorithms": "Database-Aware Priority Sampling",
        "Research Contribution": "Novel algorithm prioritizing replay based on workload diversity, optimization impact, and rarity rather than generic ML metrics."
    },
    "Drift Detector": {
        "Purpose": "Monitors the incoming query distribution for concept drift (changes in workload patterns).",
        "Input": "Sequence of Query Embeddings",
        "Output": "Drift Flag (True/False), Phase ID",
        "Algorithms": "Sliding Window Centroid Tracking, Cosine Distance",
        "Research Contribution": "Prevents unnecessary training on stable workloads and automatically defines 'Workload Phases' for EWC."
    },
    "Continual Learning": {
        "Purpose": "Asynchronously updates the neural network using new experiences and historical replays without catastrophic forgetting.",
        "Input": "Replay Buffer Samples",
        "Output": "Updated Neural Network Weights",
        "Algorithms": "PyTorch Backpropagation, Elastic Weight Consolidation (EWC)",
        "Research Contribution": "Native EWC implementation maintains parameter importance across workload phases detected by the drift detector."
    }
}

if "selected_node" not in st.session_state:
    st.session_state.selected_node = "Decision Engine"

def select_node(node_name):
    st.session_state.selected_node = node_name

col_diagram, col_info = st.columns([1.5, 2])

with col_diagram:
    st.markdown("### Pipeline Flow")
    
    # Custom CSS for the buttons to make them look like a flowchart
    st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        margin: 2px 0px;
        background-color: #2b2b2b;
        color: #00ffaa;
        border: 1px solid #00ffaa;
        border-radius: 5px;
    }
    div.stButton > button:hover {
        background-color: #00ffaa;
        color: #000;
    }
    .arrow {
        text-align: center;
        color: #888;
        font-size: 24px;
        margin: -10px 0px;
    }
    </style>
    """, unsafe_allow_html=True)

    nodes = [
        "Incoming Query", "SQL Parser", "Embedding Generator", "Qdrant (Vector DB)", 
        "Decision Engine", "Safety Layer", "PostgreSQL", "Performance Metrics", 
        "Reward Calculation", "Replay Buffer", "Drift Detector", "Continual Learning"
    ]
    
    for i, node in enumerate(nodes):
        if st.button(node, key=f"btn_{i}"):
            select_node(node)
        if i < len(nodes) - 1:
            st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)

with col_info:
    selected = st.session_state.selected_node
    info = ARCHITECTURE_INFO.get(selected, {})
    
    st.markdown(f"## {selected}")
    st.divider()
    
    st.markdown("#### 🎯 Purpose")
    st.info(info.get("Purpose", ""))
    
    col_in, col_out = st.columns(2)
    with col_in:
        st.markdown("#### 📥 Input")
        st.code(info.get("Input", ""))
    with col_out:
        st.markdown("#### 📤 Output")
        st.code(info.get("Output", ""))
        
    st.markdown("#### ⚙️ Algorithms")
    st.success(info.get("Algorithms", ""))
    
    st.markdown("#### 🔬 Research Contribution")
    st.warning(info.get("Research Contribution", ""))
