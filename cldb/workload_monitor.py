from datetime import datetime, timedelta
from sqlalchemy import func
from cldb.metadata_repo import MetadataRepo, QueryLog


class WorkloadMonitor:
    def __init__(self, repo: MetadataRepo):
        self.repo = repo

    def get_workload_stats(self, db_id: int, window_minutes: int = 60):
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)

        with self.repo.get_session() as session:
            stats = session.query(
                func.count(QueryLog.query_id).label("total_queries"),
                func.avg(QueryLog.execution_time_ms).label("avg_latency_ms"),
            ).filter(
                QueryLog.db_id == db_id,
                QueryLog.executed_at >= cutoff_time,
            ).first()

            total = stats.total_queries or 0
            avg_latency = stats.avg_latency_ms or 0.0

            read_queries = session.query(func.count(QueryLog.query_id)).filter(
                QueryLog.db_id == db_id,
                QueryLog.executed_at >= cutoff_time,
                QueryLog.query_text.ilike("SELECT%"),
            ).scalar() or 0

        write_queries = total - read_queries
        rw_ratio = read_queries / write_queries if write_queries > 0 else float("inf")

        return {
            "total_queries": total,
            "avg_latency_ms": avg_latency,
            "read_queries": read_queries,
            "write_queries": write_queries,
            "rw_ratio": rw_ratio,
        }

    def detect_workload_shift(self, db_id: int, window_minutes: int = 60) -> bool:
        with self.repo.get_session() as session:
            recent = session.query(
                func.count(QueryLog.query_id).label("count"),
                func.avg(QueryLog.execution_time_ms).label("avg_time"),
            ).filter(
                QueryLog.db_id == db_id,
                QueryLog.executed_at >= datetime.utcnow() - timedelta(minutes=window_minutes // 2),
            ).first()

            older = session.query(
                func.count(QueryLog.query_id).label("count"),
                func.avg(QueryLog.execution_time_ms).label("avg_time"),
            ).filter(
                QueryLog.db_id == db_id,
                QueryLog.executed_at < datetime.utcnow() - timedelta(minutes=window_minutes // 2),
                QueryLog.executed_at >= datetime.utcnow() - timedelta(minutes=window_minutes),
            ).first()

        if not recent or not older:
            return False

        latency_ratio = (recent.avg_time or 0) / max(older.avg_time or 1, 1)
        query_ratio = (recent.count or 0) / max(older.count or 1, 1)

        return latency_ratio > 1.5 or query_ratio > 1.5 or query_ratio < 0.5