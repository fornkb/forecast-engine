-- Stock levels the ERP can act on.
--
-- Mirrors the forecast side deliberately: one row per publish in policy_run, the
-- rows themselves in stock_policy, and a view naming the live one. An ERP reads
-- the view.
--
--   psql -U postgres -d forecast_erp -f scripts/sql/005_stock_policy.sql

CREATE TABLE IF NOT EXISTS forecast_output.policy_run (
    run_id          BIGINT      NOT NULL,
    client_id       TEXT        NOT NULL,
    -- The forecast these levels were derived from. Without it nobody can tell
    -- whether a published policy belongs to the published forecast.
    forecast_run_id BIGINT      NOT NULL,
    service_level   DOUBLE PRECISION NOT NULL,
    window_periods  INTEGER     NOT NULL,
    series_count    INTEGER     NOT NULL,
    published_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (client_id, run_id)
);

CREATE TABLE IF NOT EXISTS forecast_output.stock_policy (
    run_id          BIGINT NOT NULL,
    client_id       TEXT   NOT NULL,
    series_id       TEXT   NOT NULL,
    model           TEXT   NOT NULL,
    service_level   DOUBLE PRECISION NOT NULL,
    window_periods  INTEGER NOT NULL,
    expected_demand DOUBLE PRECISION,
    target_stock    DOUBLE PRECISION,
    safety_stock    DOUBLE PRECISION,
    reorder_point   DOUBLE PRECISION,
    PRIMARY KEY (client_id, run_id, series_id)
);

CREATE INDEX IF NOT EXISTS stock_policy_series_idx
    ON forecast_output.stock_policy (client_id, series_id);

DROP VIEW IF EXISTS forecast_output.v_current_stock_policy;
CREATE VIEW forecast_output.v_current_stock_policy AS
SELECT p.*
FROM forecast_output.stock_policy p
JOIN (
    SELECT DISTINCT ON (client_id) client_id, run_id
    FROM forecast_output.policy_run
    ORDER BY client_id, published_at DESC, run_id DESC
) current_run USING (client_id, run_id);

COMMENT ON VIEW forecast_output.v_current_stock_policy IS
    'What to reorder, per item. target_stock is the order-up-to level for a '
    'periodic review; reorder_point is the trigger for a continuous one. An ERP '
    'should read this rather than stock_policy, which holds every published run.';

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'engine_writer') THEN
        GRANT SELECT, INSERT, DELETE ON ALL TABLES IN SCHEMA forecast_output TO engine_writer;
    END IF;
END
$$;
