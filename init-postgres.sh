#!/bin/bash
set -e
 
# Create postgres role if it doesn't exist
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE ROLE postgres WITH LOGIN SUPERUSER PASSWORD 'postgres';
    GRANT ALL PRIVILEGES ON DATABASE basketball_analysis TO postgres;
EOSQL 