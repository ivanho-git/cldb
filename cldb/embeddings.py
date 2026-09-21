from sentence_transformers import SentenceTransformer

class QueryEmbedder:
    """
    Generates semantic embeddings for SQL queries using sentence-transformers.
    A SQL-aware encoder like CodeBERT could be swapped in here later.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Small general-purpose model suitable for the v1 prototype
        self.model = SentenceTransformer(model_name)
        
    def embed(self, query: str):
        # Normalize the query (lowercase, strip extra whitespace) to improve semantic matching
        normalized = " ".join(query.split()).lower()
        return self.model.encode(normalized)
