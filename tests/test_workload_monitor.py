import pytest
from cldb.metadata_repo import MetadataRepo, Database, QueryLog
from cldb.workload_monitor import WorkloadMonitor

@pytest.fixture
def repo():
    import os
    if "POSTGRES_HOST" in os.environ:
        del os.environ["POSTGRES_HOST"]
    return MetadataRepo(db_url="sqlite:///:memory:")

def test_workload_stats(repo):
    session = repo.get_session()
    db = Database(db_name="test_db")
    session.add(db)
    session.commit()
    
    # 2 SELECTs, 1 UPDATE
    logs = [
        QueryLog(db_id=db.db_id, query_text="SELECT * FROM A;", execution_time_ms=10.0),
        QueryLog(db_id=db.db_id, query_text="select id from B;", execution_time_ms=20.0),
        QueryLog(db_id=db.db_id, query_text="UPDATE A SET val=1;", execution_time_ms=30.0),
    ]
    session.add_all(logs)
    session.commit()
    
    monitor = WorkloadMonitor(repo)
    stats = monitor.get_workload_stats(db_id=db.db_id, window_minutes=10)
    
    assert stats["total_queries"] == 3
    assert stats["avg_latency_ms"] == 20.0
    assert stats["read_queries"] == 2
    assert stats["write_queries"] == 1
    assert stats["rw_ratio"] == 2.0
    
    session.close()
