import pytest
from cldb.metadata_repo import MetadataRepo, Database, DBTable, DBColumn, User, QueryLog

@pytest.fixture
def repo():
    # Uses in-memory sqlite by default when db_url is None and no POSTGRES env vars are set
    # Ensure no POSTGRES env vars are set for this test
    import os
    if "POSTGRES_HOST" in os.environ:
        del os.environ["POSTGRES_HOST"]
        
    r = MetadataRepo(db_url="sqlite:///:memory:")
    return r

def test_metadata_repo_init(repo):
    assert repo.engine is not None

def test_add_database_and_table(repo):
    session = repo.get_session()
    
    db = Database(db_name="test_db")
    session.add(db)
    session.commit()
    
    table = DBTable(db_id=db.db_id, schema_name="public", table_name="test_table")
    session.add(table)
    session.commit()
    
    col = DBColumn(table_id=table.table_id, column_name="id", data_type="INTEGER")
    session.add(col)
    session.commit()
    
    # Query back
    fetched_db = session.query(Database).filter_by(db_name="test_db").first()
    assert fetched_db is not None
    assert fetched_db.db_name == "test_db"
    
    fetched_table = session.query(DBTable).filter_by(table_name="test_table").first()
    assert fetched_table is not None
    assert len(fetched_table.columns) == 1
    assert fetched_table.columns[0].column_name == "id"
    
    session.close()

def test_query_log(repo):
    session = repo.get_session()
    
    db = Database(db_name="test_db2")
    user = User(username="admin")
    session.add_all([db, user])
    session.commit()
    
    log = QueryLog(
        db_id=db.db_id, 
        user_id=user.user_id, 
        query_text="SELECT * FROM table;",
        execution_time_ms=42.0
    )
    session.add(log)
    session.commit()
    
    fetched_log = session.query(QueryLog).first()
    assert fetched_log.query_text == "SELECT * FROM table;"
    assert fetched_log.execution_time_ms == 42.0
    
    session.close()
