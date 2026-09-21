import pytest
from cldb.parser import QueryFeatures
from cldb.decision_engine import DecisionEngine

def test_decision_engine():
    engine = DecisionEngine()
    features = QueryFeatures(predicate_columns=["category"])
    
    # 0 = KEEP
    decision_keep = engine.formulate_action(0, 0.9, features, "products")
    assert decision_keep["action"] == "KEEP"
    
    # 1 = CREATE
    decision_create = engine.formulate_action(1, 0.85, features, "products")
    assert decision_create["action"] == "CREATE_INDEX"
    assert "category" in decision_create["target_columns"]
    assert decision_create["confidence"] == 0.85
    
    # CREATE with no predicates should fallback to KEEP
    features_empty = QueryFeatures()
    decision_fallback = engine.formulate_action(1, 0.8, features_empty, "products")
    assert decision_fallback["action"] == "KEEP"
