-- E-commerce Schema Tables (Seed Data for CLDB Optimization Target)

CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    region VARCHAR(50),
    registration_date TIMESTAMP
);

CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10, 2),
    stock INT
);

CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    order_date TIMESTAMP,
    status VARCHAR(20),
    total_amount DECIMAL(10, 2)
);

CREATE TABLE order_items (
    item_id SERIAL PRIMARY KEY,
    order_id INT REFERENCES orders(order_id),
    product_id INT REFERENCES products(product_id),
    quantity INT,
    unit_price DECIMAL(10, 2)
);

-- Insert dummy data (Customers, Products, Orders, Items)
INSERT INTO customers (name, email, region, registration_date)
SELECT 
    'Customer ' || i,
    'customer' || i || '@example.com',
    (ARRAY['North', 'South', 'East', 'West'])[floor(random() * 4 + 1)],
    CURRENT_TIMESTAMP - (random() * 365 || ' days')::interval
FROM generate_series(1, 100) s(i);

INSERT INTO products (name, category, price, stock)
SELECT
    'Product ' || i,
    (ARRAY['Electronics', 'Books', 'Clothing', 'Home'])[floor(random() * 4 + 1)],
    random() * 500 + 10,
    floor(random() * 1000)
FROM generate_series(1, 50) s(i);

INSERT INTO orders (customer_id, order_date, status, total_amount)
SELECT
    floor(random() * 100 + 1),
    CURRENT_TIMESTAMP - (random() * 365 || ' days')::interval,
    (ARRAY['PENDING', 'SHIPPED', 'DELIVERED', 'CANCELLED'])[floor(random() * 4 + 1)],
    random() * 1000
FROM generate_series(1, 500) s(i);

INSERT INTO order_items (order_id, product_id, quantity, unit_price)
SELECT
    floor(random() * 500 + 1),
    floor(random() * 50 + 1),
    floor(random() * 5 + 1),
    random() * 100
FROM generate_series(1, 1500) s(i);

-- Populate CLDB Metadata Tables with the Schema Info
INSERT INTO databases (db_name) VALUES ('ecommerce_db');

INSERT INTO tables (db_id, schema_name, table_name, row_count_estimate) VALUES 
(1, 'public', 'customers', 100),
(1, 'public', 'products', 50),
(1, 'public', 'orders', 500),
(1, 'public', 'order_items', 1500);

-- Populate columns metadata
INSERT INTO columns (table_id, column_name, data_type) VALUES 
((SELECT table_id FROM tables WHERE table_name = 'customers'), 'customer_id', 'INTEGER'),
((SELECT table_id FROM tables WHERE table_name = 'customers'), 'region', 'VARCHAR'),
((SELECT table_id FROM tables WHERE table_name = 'products'), 'product_id', 'INTEGER'),
((SELECT table_id FROM tables WHERE table_name = 'products'), 'category', 'VARCHAR'),
((SELECT table_id FROM tables WHERE table_name = 'orders'), 'order_id', 'INTEGER'),
((SELECT table_id FROM tables WHERE table_name = 'orders'), 'customer_id', 'INTEGER'),
((SELECT table_id FROM tables WHERE table_name = 'orders'), 'order_date', 'TIMESTAMP'),
((SELECT table_id FROM tables WHERE table_name = 'orders'), 'status', 'VARCHAR'),
((SELECT table_id FROM tables WHERE table_name = 'order_items'), 'item_id', 'INTEGER'),
((SELECT table_id FROM tables WHERE table_name = 'order_items'), 'order_id', 'INTEGER'),
((SELECT table_id FROM tables WHERE table_name = 'order_items'), 'product_id', 'INTEGER');

-- Generate Synthetic Query Logs (~200 queries spanning 3 workload phases)
-- Phase 1: Point lookups (OLTP) - 80 queries
INSERT INTO query_logs (db_id, query_text, execution_time_ms)
SELECT 
    1,
    'SELECT * FROM orders WHERE customer_id = ' || floor(random() * 100 + 1) || ';',
    random() * 50
FROM generate_series(1, 40) s(i);

INSERT INTO query_logs (db_id, query_text, execution_time_ms)
SELECT 
    1,
    'SELECT * FROM order_items oi JOIN products p ON oi.product_id = p.product_id WHERE oi.order_id = ' || floor(random() * 500 + 1) || ';',
    random() * 100
FROM generate_series(1, 40) s(i);

-- Phase 2: Read-Heavy Lookups (Black Friday) - 80 queries
INSERT INTO query_logs (db_id, query_text, execution_time_ms)
SELECT 
    1,
    'SELECT * FROM products WHERE category = ''' || (ARRAY['Electronics', 'Books', 'Clothing', 'Home'])[floor(random() * 4 + 1)] || ''';',
    random() * 200
FROM generate_series(1, 40) s(i);

INSERT INTO query_logs (db_id, query_text, execution_time_ms)
SELECT 
    1,
    'SELECT order_id, status FROM orders WHERE status = ''' || (ARRAY['PENDING', 'SHIPPED', 'DELIVERED', 'CANCELLED'])[floor(random() * 4 + 1)] || ''';',
    random() * 150
FROM generate_series(1, 40) s(i);

-- Phase 3: Analytical (Month-end) - 40 queries
INSERT INTO query_logs (db_id, query_text, execution_time_ms)
SELECT 
    1,
    'SELECT c.region, COUNT(o.order_id), SUM(o.total_amount) FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.region;',
    random() * 1000 + 500
FROM generate_series(1, 20) s(i);

INSERT INTO query_logs (db_id, query_text, execution_time_ms)
SELECT 
    1,
    'SELECT DATE_TRUNC(''month'', order_date) as month, SUM(total_amount) FROM orders GROUP BY DATE_TRUNC(''month'', order_date);',
    random() * 800 + 400
FROM generate_series(1, 20) s(i);
