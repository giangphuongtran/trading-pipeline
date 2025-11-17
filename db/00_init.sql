-- db/01-init-databases.sql
-- Create databases for trading data and Airflow metadata

CREATE DATABASE airflow;

-- Create dedicated airflow user
CREATE USER airflow WITH PASSWORD 'airflow_password';
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;

-- Grant access to trading_data as well (if needed)
GRANT ALL PRIVILEGES ON DATABASE trading_data TO airflow;

-- db/02-grant-schema-permissions.sql
-- Grant schema-level permissions (PostgreSQL 15+ requirement)

-- Connect to airflow database and grant schema permissions
\c airflow

GRANT ALL ON SCHEMA public TO airflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO airflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO airflow;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO airflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO airflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO airflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO airflow;

-- Connect to trading_data database and grant schema permissions
\c trading_data

GRANT ALL ON SCHEMA public TO airflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO airflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO airflow;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO airflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO airflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO airflow;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO airflow;

-- Also grant to the main user for the trading_data schema
GRANT ALL ON SCHEMA public TO phuonggiang_pgt;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO phuonggiang_pgt;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO phuonggiang_pgt;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO phuonggiang_pgt;