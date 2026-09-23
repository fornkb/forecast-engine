-- Read-only role for the engine.
--
-- The engine must not be able to write to the ERP. SELECT on one view is the
-- entire grant: not the base table, not the schema at large. If a future change
-- needs more, it should be visible as a new grant rather than inherited.
--
-- The password is passed in rather than written here, so this file stays
-- committable. Do not commit the password; put the resulting DSN in the
-- ENGINE_PG_DSN environment variable instead.
--
--   psql -U postgres -d forecast_erp \
--        -v reader_password="$ENGINE_READER_PASSWORD" \
--        -f scripts/sql/002_engine_reader.sql

\if :{?reader_password}
\else
\echo 'ERROR: pass the password with  psql -v reader_password=...'
\quit 1
\endif

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'engine_reader') THEN
        CREATE ROLE engine_reader LOGIN;
    END IF;
END
$$;

ALTER ROLE engine_reader WITH PASSWORD :'reader_password';

GRANT CONNECT ON DATABASE forecast_erp TO engine_reader;
GRANT USAGE   ON SCHEMA erp            TO engine_reader;
GRANT SELECT  ON erp.v_demand_history  TO engine_reader;

-- Explicitly not granted: INSERT/UPDATE/DELETE anywhere, and SELECT on
-- erp.sales_history. The engine sees the contract, not the base data.
REVOKE ALL ON erp.sales_history FROM engine_reader;
