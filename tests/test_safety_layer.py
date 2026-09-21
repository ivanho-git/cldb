import pytest
from cldb.metadata_repo import MetadataRepo
from cldb.safety_layer import SafetyLayer

@pytest.fixture
def repo():
    import os
    if "POSTGRES_HOST" in os.environ:
        del os.environ["POSTGRES_HOST"]
    return MetadataRepo(db_url="sqlite:///:memory:")

def test_safety_layer_confidence(repo):
    layer = SafetyLayer(repo, confidence_threshold=0.8)
    
    # Low confidence should be rejected
    decision = {"action": "CREATE_INDEX", "confidence": 0.5}
    assert layer.evaluate_decision(decision, "SELECT 1") == False
    
    # High confidence should pass
    decision_high = {"action": "CREATE_INDEX", "confidence": 0.9}
    assert layer.evaluate_decision(decision_high, "SELECT 1") == True

def test_safety_layer_rollback(repo):
    layer = SafetyLayer(repo)
    
    rollback = layer.generate_rollback_action("CREATE_INDEX", ["id"], "users")
    assert rollback["action"] == "DROP_INDEX"
    assert rollback["table_name"] == "users"
    assert "id" in rollback["target_columns"]
    
    rollback_drop = layer.generate_rollback_action("DROP_INDEX", ["id"], "users")
    assert rollback_drop["action"] == "CREATE_INDEX"
