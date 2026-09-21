from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, BigInteger, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.pool import StaticPool
from datetime import datetime
import os

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Database(Base):
    __tablename__ = "databases"
    db_id = Column(Integer, primary_key=True)
    db_name = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class DBTable(Base):
    __tablename__ = "tables"
    table_id = Column(Integer, primary_key=True)
    db_id = Column(Integer, ForeignKey("databases.db_id", ondelete="CASCADE"))
    schema_name = Column(String(50), nullable=False)
    table_name = Column(String(100), nullable=False)
    row_count_estimate = Column(BigInteger)

    columns = relationship("DBColumn", back_populates="table")
    indexes = relationship("DBIndex", back_populates="table")


class DBColumn(Base):
    __tablename__ = "columns"
    column_id = Column(Integer, primary_key=True)
    table_id = Column(Integer, ForeignKey("tables.table_id", ondelete="CASCADE"))
    column_name = Column(String(100), nullable=False)
    data_type = Column(String(50))

    table = relationship("DBTable", back_populates="columns")


class DBIndex(Base):
    __tablename__ = "indexes"
    index_id = Column(Integer, primary_key=True)
    table_id = Column(Integer, ForeignKey("tables.table_id", ondelete="CASCADE"))
    index_name = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    dropped_at = Column(DateTime)

    table = relationship("DBTable", back_populates="indexes")
    index_columns = relationship("IndexColumn", back_populates="index_ref")


class IndexColumn(Base):
    __tablename__ = "index_columns"
    index_id = Column(Integer, ForeignKey("indexes.index_id", ondelete="CASCADE"), primary_key=True)
    column_id = Column(Integer, ForeignKey("columns.column_id", ondelete="CASCADE"), primary_key=True)

    index_ref = relationship("DBIndex", back_populates="index_columns")


class QueryLog(Base):
    __tablename__ = "query_logs"
    query_id = Column(Integer, primary_key=True)
    db_id = Column(Integer, ForeignKey("databases.db_id", ondelete="CASCADE"))
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="SET NULL"))
    query_text = Column(String, nullable=False)
    query_hash = Column(String(64))
    execution_time_ms = Column(Float)
    executed_at = Column(DateTime, default=datetime.utcnow)


class ExecutionPlan(Base):
    __tablename__ = "execution_plans"
    plan_id = Column(Integer, primary_key=True)
    query_id = Column(Integer, ForeignKey("query_logs.query_id", ondelete="CASCADE"))
    plan_json = Column(JSON, nullable=False)
    estimated_cost = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"
    metric_id = Column(Integer, primary_key=True)
    table_id = Column(Integer, ForeignKey("tables.table_id", ondelete="CASCADE"))
    metric_name = Column(String(50))
    metric_value = Column(Float)
    recorded_at = Column(DateTime, default=datetime.utcnow)


class OptimizationDecision(Base):
    __tablename__ = "optimization_decisions"
    decision_id = Column(Integer, primary_key=True)
    table_id = Column(Integer, ForeignKey("tables.table_id", ondelete="CASCADE"))
    action = Column(String(20), nullable=False)
    confidence_score = Column(Float)
    estimated_cost_saving = Column(Float)
    status = Column(String(20), default="PENDING")
    created_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime)


class OptimizationDecisionColumn(Base):
    __tablename__ = "optimization_decision_columns"
    decision_id = Column(Integer, ForeignKey("optimization_decisions.decision_id", ondelete="CASCADE"), primary_key=True)
    column_id = Column(Integer, ForeignKey("columns.column_id", ondelete="CASCADE"), primary_key=True)


class OptimizationHistory(Base):
    __tablename__ = "optimization_history"
    history_id = Column(Integer, primary_key=True)
    decision_id = Column(Integer, ForeignKey("optimization_decisions.decision_id", ondelete="CASCADE"))
    table_id = Column(Integer, ForeignKey("tables.table_id", ondelete="CASCADE"))
    action = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow)


class ReplayBufferMetadata(Base):
    __tablename__ = "replay_buffer_metadata"
    buffer_id = Column(Integer, primary_key=True)
    query_id = Column(Integer, ForeignKey("query_logs.query_id", ondelete="CASCADE"))
    importance_score = Column(Float)
    is_active = Column(Boolean, default=True)
    added_at = Column(DateTime, default=datetime.utcnow)


class MetadataRepo:
    def __init__(self, db_url: str = None):
        if db_url is None:
            host = os.environ.get("POSTGRES_HOST")
            if host:
                user = os.environ.get("POSTGRES_USER", "postgres")
                password = os.environ.get("POSTGRES_PASSWORD", "postgres")
                db = os.environ.get("POSTGRES_DB", "cldb")
                port = os.environ.get("POSTGRES_PORT", "5432")
                db_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
            else:
                db_url = "sqlite:///:memory:"

        if db_url == "sqlite:///:memory:":
            self.engine = create_engine(db_url, connect_args={"check_same_thread": False}, poolclass=StaticPool)
        else:
            self.engine = create_engine(db_url)
        if db_url.startswith("sqlite"):
            Base.metadata.create_all(self.engine)

        self.Session = sessionmaker(bind=self.engine)

    def get_session(self):
        return self.Session()

    @property
    def session_factory(self):
        return self.Session