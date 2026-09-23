-- Where the engine publishes its forecasts.
--
-- Deliberately a separate schema from erp: the engine reads one and writes the
-- other, under different roles, and neither grant overlaps. The ERP reads the
-- views at the bottom, not the tables.
--
--   psql -U postgres -d forecast_erp -f scripts/sql/003_forecast_output.sql

CREATE SCHEMA IF NOT EXISTS forecast_output;

-- One row per publish. Lets the ERP tell one forecast cycle from the next, and
-- makes "which forecast is live" a question with an answer.
CREATE TABLE IF NOT EXISTS forecast_output.forecast_run (
    run_id        BIGINT      NOT NULL,
    client_id     TEXT        NOT NULL,
    cutoff_period DATE        NOT NULL,
    horizon       INTEGER     NOT NULL,
    series_count  INTEGER     NOT NULL,
    row_count     INTEGER     NOT NULL,
    published_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (client_id, run_id)
);

CREATE TABLE IF NOT EXISTS forecast_output.demand_forecast (
    run_id        BIGINT NOT NULL,
    client_id     TEXT   NOT NULL,
    series_id     TEXT   NOT NULL,
    cutoff_period DATE   NOT NULL,
    period        DATE   NOT NULL,
    horizon       INTEGER NOT NULL,
    model         TEXT   NOT NULL,
    forecast      DOUBLE PRECISION,
    PRIMARY KEY (client_id, run_id, series_id, period)
);

-- Long, not one column per quantile. The set of levels is client configuration
-- (probabilistic.levels), so a wide table would need a migration every time a
-- client wanted a different service level.
CREATE TABLE IF NOT EXISTS forecast_output.demand_forecast_quantile (
    run_id    BIGINT NOT NULL,
    client_id TEXT   NOT NULL,
    series_id TEXT   NOT NULL,
    period    DATE   NOT NULL,
    horizon   INTEGER NOT NULL,
    model     TEXT   NOT NULL,
    level     DOUBLE PRECISION NOT NULL,
    value     DOUBLE PRECISION,
    PRIMARY KEY (client_id, run_id, series_id, period, level)
);

-- Unlike the source side, these are read by item and by period, so the indexes
-- earn their place.
CREATE INDEX IF NOT EXISTS demand_forecast_series_idx
    ON forecast_output.demand_forecast (client_id, series_id, period);

-- "Current" is the most recently published run, not the highest run id:
-- republishing an older run is a deliberate act and should take effect.
DROP VIEW IF EXISTS forecast_output.v_current_forecast;
CREATE VIEW forecast_output.v_current_forecast AS
SELECT f.*
FROM forecast_output.demand_forecast f
JOIN (
    SELECT DISTINCT ON (client_id) client_id, run_id
    FROM forecast_output.forecast_run
    ORDER BY client_id, published_at DESC, run_id DESC
) current_run USING (client_id, run_id);

DROP VIEW IF EXISTS forecast_output.v_current_forecast_quantile;
CREATE VIEW forecast_output.v_current_forecast_quantile AS
SELECT q.*
FROM forecast_output.demand_forecast_quantile q
JOIN (
    SELECT DISTINCT ON (client_id) client_id, run_id
    FROM forecast_output.forecast_run
    ORDER BY client_id, published_at DESC, run_id DESC
) current_run USING (client_id, run_id);

COMMENT ON VIEW forecast_output.v_current_forecast IS
    'The live forecast. An ERP should read this rather than demand_forecast, '
    'which holds every published run.';

-- Reinstate the writer's grants if the role already exists, so re-running this
-- script cannot quietly lock the engine out of its own output.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'engine_writer') THEN
        GRANT USAGE ON SCHEMA forecast_output TO engine_writer;
        GRANT SELECT, INSERT, DELETE ON ALL TABLES IN SCHEMA forecast_output TO engine_writer;
    END IF;
END
$$;
