from sqlalchemy import text
from cldb.metadata_repo import MetadataRepo, DBIndex, DBColumn, IndexColumn


class Executor:
    def __init__(self, repo: MetadataRepo):
        self.repo = repo

    def _index_exists(self, table_name: str, columns: list) -> bool:
        with self.repo.get_session() as session:
            existing = session.query(DBIndex).join(
                IndexColumn, IndexColumn.index_id == DBIndex.index_id
            ).join(
                DBColumn, DBColumn.column_id == IndexColumn.column_id
            ).filter(
                DBIndex.is_active == True,
                DBColumn.column_name.in_(columns),
            ).all()

            for idx in existing:
                idx_cols = session.query(IndexColumn).filter(IndexColumn.index_id == idx.index_id).all()
                if len(idx_cols) == len(columns):
                    return True
        return False

    def execute_decision(self, decision: dict) -> bool:
        action = decision.get("action")
        if action == "KEEP" or not action:
            return True

        table = decision.get("table_name")
        columns = decision.get("target_columns", [])

        if not table or not columns:
            return False

        index_name = f"idx_cldb_{table}_{'_'.join(columns)}"

        try:
            with self.repo.engine.connect() as conn:
                if action == "CREATE_INDEX":
                    if self._index_exists(table, columns):
                        return True

                    cols_str = ", ".join(columns)
                    conn.execute(text(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} ({cols_str})"))
                elif action == "DROP_INDEX":
                    conn.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
                conn.commit()
            return True
        except Exception as e:
            print(f"Executor Error: {e}")
            return False

    def list_cldb_indexes(self):
        with self.repo.get_session() as session:
            return session.query(DBIndex).filter(
                DBIndex.index_name.like("idx_cldb_%"),
                DBIndex.is_active == True,
            ).all()