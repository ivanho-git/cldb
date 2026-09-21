"""
Tests for enhanced safety layer.
"""
import pytest
from cldb.metadata_repo import MetadataRepo
from cldb.enhanced_safety import EnhancedSafetyLayer, SafetyLayer


@pytest.fixture
def repo():
    import os
    if "POSTGRES_HOST" in os.environ:
        del os.environ["POSTGRES_HOST"]
    return MetadataRepo(db_url="sqlite:///:memory:")


def test_safety_layer_init(repo):
    layer = EnhancedSafetyLayer(repo, confidence_threshold=0.8, max_indexes_per_table=3)
    
    assert layer.confidence_threshold == 0.8
    assert layer.max_indexes_per_table == 3


def test_safety_layer_low_confidence_rejection(repo):
    layer = EnhancedSafetyLayer(repo, confidence_threshold=0.8)
    
    decision = {"action": "CREATE_INDEX", "confidence": 0.5, "table_name": "test_table"}
    
    result = layer.evaluate_decision(decision, "SELECT 1")
    
    assert result == False


def test_safety_layer_high_confidence_approval(repo):
    layer = EnhancedSafetyLayer(repo, confidence_threshold=0.8)
    
    decision = {"action": "CREATE_INDEX", "confidence": 0.95, "table_name": "test_table"}
    
    result = layer.evaluate_decision(decision, "SELECT 1")
    
    assert result == True


def test_safety_layer_keep_action_always_passes(repo):
    layer = EnhancedSafetyLayer(repo)
    
    decision = {"action": "KEEP", "confidence": 0.3}
    
    result = layer.evaluate_decision(decision, "SELECT 1")
    
    assert result == True


def test_safety_layer_rollback_generation(repo):
    layer = EnhancedSafetyLayer(repo)
    
    rollback = layer.generate_rollback_action("CREATE_INDEX", ["col1", "col2"], "my_table")
    
    assert rollback["action"] == "DROP_INDEX"
    assert rollback["table_name"] == "my_table"
    assert rollback["target_columns"] == ["col1", "col2"]


def test_safety_layer_statistics(repo):
    layer = EnhancedSafetyLayer(repo)
    
    decision = {"action": "CREATE_INDEX", "confidence": 0.95, "table_name": "test_table"}
    layer.evaluate_decision(decision, "SELECT 1")
    
    stats = layer.get_statistics()
    
    assert "event_counts" in stats
    assert "active_operations" in stats


def test_safety_layer_audit_log(repo):
    layer = EnhancedSafetyLayer(repo)
    
    decision = {"action": "CREATE_INDEX", "confidence": 0.3, "table_name": "test_table"}
    layer.evaluate_decision(decision, "SELECT 1")
    
    audit = layer.get_audit_log()
    
    assert len(audit) > 0
    assert audit[-1]["event_type"] == "rejected"


def test_safety_layer_execute_with_rollback(repo):
    layer = EnhancedSafetyLayer(repo, rollback_on_regression=True, regression_threshold_ms=10.0)
    
    executed = False
    
    def executor(d):
        nonlocal executed
        executed = True
        return True
    
    # Positive improvement - no rollback
    result = layer.execute_with_rollback(executor, {"action": "CREATE_INDEX", "table_name": "test", "target_columns": ["col"]}, latency_before=100, latency_after=80)
    
    assert result == True
    assert executed == True