import pytest
from cldb.parser import SQLParser

@pytest.fixture
def parser():
    return SQLParser()

def test_parse_simple_select(parser):
    query = "SELECT * FROM users;"
    features = parser.parse(query)
    assert features.tables == ["users"]
    assert features.join_count == 0
    assert len(features.predicate_columns) == 0
    assert not features.has_group_by
    assert not features.has_order_by

def test_parse_where_clause(parser):
    query = "SELECT id, name FROM products WHERE category = 'Electronics' AND price > 100;"
    features = parser.parse(query)
    assert "products" in features.tables
    assert "category" in features.predicate_columns
    assert "price" in features.predicate_columns
    assert features.join_count == 0

def test_parse_join(parser):
    query = "SELECT o.id, c.name FROM orders o JOIN customers c ON o.customer_id = c.id;"
    features = parser.parse(query)
    assert "orders" in features.tables
    assert "customers" in features.tables
    assert features.join_count == 1
    assert "customer_id" in features.predicate_columns
    assert "id" in features.predicate_columns

def test_parse_group_by_order_by(parser):
    query = "SELECT region, COUNT(*) FROM customers GROUP BY region ORDER BY COUNT(*) DESC;"
    features = parser.parse(query)
    assert features.has_group_by
    assert features.has_order_by

def test_parse_complex_query(parser):
    query = """
    SELECT c.region, SUM(oi.unit_price * oi.quantity) as total_sales
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    INNER JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.status = 'DELIVERED'
    GROUP BY c.region
    ORDER BY total_sales DESC;
    """
    features = parser.parse(query)
    assert set(features.tables) == {"customers", "orders", "order_items"}
    assert features.join_count == 2
    assert "LEFT" in features.join_types or "INNER" in features.join_types
    assert "customer_id" in features.predicate_columns
    assert "order_id" in features.predicate_columns
    assert "status" in features.predicate_columns
    assert features.has_group_by
    assert features.has_order_by

def test_parse_invalid_sql(parser):
    query = "SELECT FROM WHERE"
    features = parser.parse(query)
    assert len(features.tables) == 0
    assert features.join_count == 0
