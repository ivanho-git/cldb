from cldb.parser import QueryFeatures

class DecisionEngine:
    """
    Translates the policy network's output logits into a concrete action:
    CREATE INDEX / DROP INDEX / KEEP.
    """
    ACTION_KEEP = 0
    ACTION_CREATE = 1
    ACTION_DROP = 2
    
    def formulate_action(self, action_idx: int, confidence: float, features: QueryFeatures, table_name: str):
        action_map = {
            self.ACTION_KEEP: "KEEP",
            self.ACTION_CREATE: "CREATE_INDEX",
            self.ACTION_DROP: "DROP_INDEX"
        }
        action_str = action_map.get(int(action_idx), "KEEP")
        
        target_columns = []
        if action_str != "KEEP":
            # For the prototype, we recommend an index on the predicate columns
            # extracted from the query. In a real system, we'd rank columns.
            target_columns = features.predicate_columns
            
        # If there are no predicates, we shouldn't recommend creating an index.
        if not target_columns and action_str == "CREATE_INDEX":
            action_str = "KEEP"
            
        return {
            "action": action_str,
            "target_columns": target_columns,
            "table_name": table_name,
            "confidence": float(confidence)
        }
