
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;



CREATE TABLE IF NOT EXISTS bronze.load_tracking (
    table_name TEXT PRIMARY KEY,
    last_write_date TIMESTAMP
);

CREATE TABLE IF NOT EXISTS silver.load_tracking (
    table_name TEXT PRIMARY KEY,
    last_write_date TIMESTAMP
);