-- Write-only role for publishing forecasts.
--
-- Separate from engine_reader on purpose. The reader can see the ERP's sales
-- history and nothing else; the writer can see the forecast output and nothing
-- else. Neither can do the other's job, so a mistake in the publish path cannot
-- touch the ERP's own data.
--
--   psql -U postgres -d forecast_erp \
--        -v writer_password="$ENGINE_WRITER_PASSWORD" \
--        -f scripts/sql/004_engine_writer.sql

\if :{?writer_password}
\else
\echo 'ERROR: pass the password with  psql -v writer_password=...'
\quit 1
\endif

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'engine_writer') THEN
        CREATE ROLE engine_writer LOGIN;
    END IF;
END
$$;

ALTER ROLE engine_writer WITH PASSWORD :'writer_password';

GRANT CONNECT ON DATABASE forecast_erp        TO engine_writer;
GRANT USAGE   ON SCHEMA forecast_output       TO engine_writer;
GRANT SELECT, INSERT, DELETE
    ON ALL TABLES IN SCHEMA forecast_output   TO engine_writer;

-- DELETE is needed because publishing a run twice replaces it rather than
-- duplicating it. UPDATE is not: a published row is never edited in place.

-- Explicitly not granted: anything at all in the erp schema. The publish path
-- has no route to the ERP's sales history.
REVOKE ALL ON SCHEMA erp FROM engine_writer;
