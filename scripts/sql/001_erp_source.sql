-- ERP-shaped source schema for the Teknic demand history.
--
-- Two layers on purpose. `erp.sales_history` is the base table, shaped the way
-- a transactional system holds it. `erp.v_demand_history` is the view the
-- engine reads, and it is the integration contract: it does the renaming and
-- the period aggregation, so the same engine config survives a source that
-- posts daily lines instead of monthly totals.
--
-- Run against the forecast_erp database:
--   psql -U postgres -d forecast_erp -f scripts/sql/001_erp_source.sql

CREATE SCHEMA IF NOT EXISTS erp;

CREATE TABLE IF NOT EXISTS erp.sales_history (
    sales_doc_id  BIGSERIAL PRIMARY KEY,
    series_id     TEXT          NOT NULL,
    customer_code TEXT          NOT NULL,
    item_code     TEXT          NOT NULL,
    item_desc     TEXT,
    item_group    TEXT,
    source_sheet  TEXT,
    uom           TEXT,
    posting_date  DATE          NOT NULL,
    quantity      NUMERIC(18,3) NOT NULL,
    -- Synthetic-data label. A real client's ERP has no such column; it is kept
    -- separate from the demand figures and never reaches a model feature.
    demand_pattern TEXT
);

-- No index on posting_date: the engine reads the whole relation, so an index
-- only slows the load. It earns its place when incremental extraction arrives.

-- Dropped and recreated rather than CREATE OR REPLACE, which cannot change a
-- column's type. The grant goes with it, so it is reinstated at the end.
DROP VIEW IF EXISTS erp.v_demand_history;

CREATE VIEW erp.v_demand_history AS
SELECT series_id,
       MIN(customer_code)  AS customer,
       MIN(item_code)      AS item_code,
       MIN(item_desc)      AS description,
       MIN(item_group)     AS item_group,
       MIN(source_sheet)   AS source_sheet,
       MIN(uom)            AS uom,
       MIN(demand_pattern) AS demand_pattern,
       date_trunc('month', posting_date)::date AS period,
       -- double precision, not numeric. The base table stays exact, but the
       -- ADBC driver maps every PostgreSQL numeric to an opaque type that
       -- arrives as a string, so an uncast column would reach the engine as
       -- text and be parsed back into a float further downstream. Converting
       -- here does it once, in the database, to the type the canonical model
       -- uses anyway. Safe for quantities: float64 is exact well past the
       -- magnitudes demand reaches.
       SUM(quantity)::double precision AS quantity
FROM erp.sales_history
GROUP BY series_id, date_trunc('month', posting_date);

COMMENT ON VIEW erp.v_demand_history IS
    'Integration contract for the forecast engine. Columns here are referenced '
    'by source.column_map in the client profile; changing one is a breaking change.';

-- Reinstate the engine's grant if the role already exists, so re-running this
-- script does not quietly leave the engine locked out of its own source.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'engine_reader') THEN
        GRANT SELECT ON erp.v_demand_history TO engine_reader;
    END IF;
END
$$;
