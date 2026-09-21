class BaselineStatic:
    """
    Static Heuristic Baseline: 
    Creates an index on any column that appears in a WHERE clause more than N times.
    Never drops indexes (except maybe standard Postgres maintenance, but naive).
    """
    def __init__(self, threshold_n: int = 5):
        self.threshold_n = threshold_n
        self.column_counts = {}
        
    def observe(self, query_features):
        """
        Updates the frequency of predicate columns observed.
        """
        for col in query_features.predicate_columns:
            self.column_counts[col] = self.column_counts.get(col, 0) + 1
            
    def recommend(self, query_features):
        """
        Recommends creating an index if a threshold is crossed.
        """
        targets = []
        for col in query_features.predicate_columns:
            if self.column_counts.get(col, 0) >= self.threshold_n:
                targets.append(col)
                
        if targets:
            return {
                "action": "CREATE_INDEX",
                "target_columns": targets
            }
        return {"action": "KEEP"}
