-- 1. Tables for Metadata Repository (BCNF-normalized)

CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE databases (
    db_id SERIAL PRIMARY KEY,
    db_name VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE tables (
    table_id SERIAL PRIMARY KEY,
    db_id INT REFERENCES databases(db_id) ON DELETE CASCADE,
    schema_name VARCHAR(50) NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    row_count_estimate BIGINT,
    UNIQUE(db_id, schema_name, table_name)
);

CREATE TABLE columns (
    column_id SERIAL PRIMARY KEY,
    table_id INT REFERENCES tables(table_id) ON DELETE CASCADE,
    column_name VARCHAR(100) NOT NULL,
    data_type VARCHAR(50),
    UNIQUE(table_id, column_name)
);

CREATE TABLE indexes (
    index_id SERIAL PRIMARY KEY,
    table_id INT REFERENCES tables(table_id) ON DELETE CASCADE,
    index_name VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dropped_at TIMESTAMP,
    UNIQUE(table_id, index_name)
);

CREATE TABLE index_columns (
    index_id INT REFERENCES indexes(index_id) ON DELETE CASCADE,
    column_id INT REFERENCES columns(column_id) ON DELETE CASCADE,
    PRIMARY KEY (index_id, column_id)
);

CREATE TABLE query_logs (
    query_id SERIAL PRIMARY KEY,
    db_id INT REFERENCES databases(db_id) ON DELETE CASCADE,
    user_id INT REFERENCES users(user_id) ON DELETE SET NULL,
    query_text TEXT NOT NULL,
    query_hash VARCHAR(64),
    execution_time_ms FLOAT,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE execution_plans (
    plan_id SERIAL PRIMARY KEY,
    query_id INT REFERENCES query_logs(query_id) ON DELETE CASCADE,
    plan_json JSONB NOT NULL,
    estimated_cost FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE performance_metrics (
    metric_id SERIAL PRIMARY KEY,
    table_id INT REFERENCES tables(table_id) ON DELETE CASCADE,
    metric_name VARCHAR(50),
    metric_value FLOAT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE optimization_decisions (
    decision_id SERIAL PRIMARY KEY,
    table_id INT REFERENCES tables(table_id) ON DELETE CASCADE,
    action VARCHAR(20) NOT NULL, -- e.g., 'CREATE_INDEX', 'DROP_INDEX', 'KEEP'
    confidence_score FLOAT,
    estimated_cost_saving FLOAT,
    status VARCHAR(20) DEFAULT 'PENDING', -- 'PENDING', 'EXECUTED', 'REJECTED', 'ROLLED_BACK'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    executed_at TIMESTAMP
);

CREATE TABLE optimization_decision_columns (
    decision_id INT REFERENCES optimization_decisions(decision_id) ON DELETE CASCADE,
    column_id INT REFERENCES columns(column_id) ON DELETE CASCADE,
    PRIMARY KEY (decision_id, column_id)
);

CREATE TABLE optimization_history (
    history_id SERIAL PRIMARY KEY,
    decision_id INT REFERENCES optimization_decisions(decision_id) ON DELETE CASCADE,
    table_id INT REFERENCES tables(table_id) ON DELETE CASCADE,
    action VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE replay_buffer_metadata (
    buffer_id SERIAL PRIMARY KEY,
    query_id INT REFERENCES query_logs(query_id) ON DELETE CASCADE,
    importance_score FLOAT,
    is_active BOOLEAN DEFAULT TRUE,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Views
CREATE OR REPLACE VIEW column_index_status AS
SELECT 
    c.column_id,
    c.table_id,
    c.column_name,
    CASE WHEN COUNT(ic.index_id) > 0 THEN TRUE ELSE FALSE END as is_indexed
FROM columns c
LEFT JOIN index_columns ic ON c.column_id = ic.column_id
LEFT JOIN indexes i ON ic.index_id = i.index_id AND i.is_active = TRUE
GROUP BY c.column_id, c.table_id, c.column_name;

CREATE OR REPLACE VIEW index_effectiveness_summary AS
SELECT 
    t.db_id,
    t.schema_name,
    t.table_name,
    COUNT(CASE WHEN od.action = 'CREATE_INDEX' AND od.status = 'EXECUTED' THEN 1 END) as total_indexes_created,
    COUNT(CASE WHEN od.action = 'DROP_INDEX' AND od.status = 'EXECUTED' THEN 1 END) as total_indexes_dropped,
    COUNT(CASE WHEN od.status = 'ROLLED_BACK' THEN 1 END) as total_rollbacks,
    AVG(CASE WHEN od.status = 'EXECUTED' THEN od.confidence_score END) as avg_execution_confidence
FROM tables t
LEFT JOIN optimization_decisions od ON t.table_id = od.table_id
GROUP BY t.db_id, t.schema_name, t.table_name;

-- 3. PL/pgSQL: Trigger for optimization_history
CREATE OR REPLACE FUNCTION log_optimization_decision()
RETURNS TRIGGER AS $$
BEGIN
    -- Log into history when a decision reaches a terminal or executed state
    IF NEW.status IN ('EXECUTED', 'ROLLED_BACK', 'REJECTED') 
       AND (OLD.status IS NULL OR OLD.status != NEW.status) THEN
        
        INSERT INTO optimization_history (decision_id, table_id, action, status)
        VALUES (NEW.decision_id, NEW.table_id, NEW.action, NEW.status);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER optimization_decision_trigger
AFTER UPDATE ON optimization_decisions
FOR EACH ROW
EXECUTE FUNCTION log_optimization_decision();

-- 4. Stored Procedure to prune old query_logs
CREATE OR REPLACE PROCEDURE prune_old_query_logs(days_to_keep INT)
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM query_logs
    WHERE executed_at < CURRENT_TIMESTAMP - (days_to_keep || ' days')::interval;
END;
$$;
