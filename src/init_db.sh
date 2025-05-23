#!/bin/bash

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
while ! pg_isready -h db -p 5432 -U postgres; do
    sleep 1
done

# Create database and tables
psql -h db -U postgres -d basketball_analysis -f /app/sql/schema.sql

echo "Database initialization complete!" 