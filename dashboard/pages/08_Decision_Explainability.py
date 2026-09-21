import streamlit as st

st.set_page_config(page_title="Decision Explainability", page_icon="🔍", layout="wide")

st.title("🔍 Decision Explainability (XAI)")

st.markdown("""
<style>
.tooltip { position: relative; display: inline-block; border-bottom: 1px dotted #00ffaa; color: #00ffaa; cursor: help; }
.tooltip .tooltiptext { visibility: hidden; width: 250px; background-color: #333; color: #fff; text-align: left; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -125px; opacity: 0; transition: opacity 0.3s; font-size: 0.9em; }
.tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
Deep neural networks are often black boxes, which is unacceptable for database administrators. CLDB uses a <span class="tooltip">Hybrid Contextual Architecture<span class="tooltiptext">Combines parametric neural networks with a non-parametric Vector Database (Qdrant) to retrieve historical evidence supporting the decision.</span></span> to explain **WHY** it made a specific decision.
""")

st.divider()

st.subheader("Latest Optimization Decisions")

decisions = [
    {
        "query": "SELECT p.name, oi.quantity FROM products p JOIN order_items oi ON p.product_id = oi.product_id WHERE oi.order_id = 100;",
        "action": "CREATE_INDEX",
        "confidence": "94%",
        "reason": "High latency detected on JOIN operation. Vector search revealed that similar past JOINs benefited significantly from indexing the foreign key.",
        "expected": "120ms -> 15ms",
        "actual": "120ms -> 14ms",
        "similar_queries": [
            {"sql": "SELECT * FROM users u JOIN posts p ON u.id = p.user_id WHERE p.post_id = 5;", "similarity": 0.89, "past_reward": "+1.30"},
            {"sql": "SELECT c.name, o.date FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.id = 12;", "similarity": 0.82, "past_reward": "+1.15"}
        ]
    },
    {
        "query": "SELECT * FROM orders WHERE status = 'PENDING';",
        "action": "KEEP",
        "confidence": "88%",
        "reason": "Query is very frequent but already executes in <10ms. Creating an index on a low-cardinality column ('status') would cause write amplification with minimal read benefit.",
        "expected": "10ms -> 10ms",
        "actual": "10ms -> 10ms",
        "similar_queries": [
            {"sql": "SELECT * FROM users WHERE active = true;", "similarity": 0.91, "past_reward": "0.00 (No action taken)"},
        ]
    }
]

for idx, dec in enumerate(decisions):
    icon = "✅" if dec["action"] == "CREATE_INDEX" else "⏸️"
    color = "#00ffaa" if dec["action"] == "CREATE_INDEX" else "#aaaaaa"
    
    with st.expander(f"{icon} {dec['action']} | Confidence: {dec['confidence']} | {dec['query'][:50]}...", expanded=(idx==0)):
        st.markdown(f"**Target Query:**")
        st.code(dec["query"], language="sql")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Action", dec["action"])
        c2.metric("Confidence", dec["confidence"])
        c3.metric("Expected vs Actual", f"{dec['expected']} vs {dec['actual']}")
        
        st.markdown("#### 🧠 Reasoning")
        st.info(dec["reason"])
        
        st.markdown("#### 📚 Retrieved Historical Evidence (Qdrant)")
        st.caption("The model used these similar past queries from the Vector DB as context to make its decision.")
        for sim in dec["similar_queries"]:
            st.markdown(f"- **Similarity: {sim['similarity']} | Past Reward: {sim['past_reward']}**")
            st.code(sim["sql"], language="sql")
