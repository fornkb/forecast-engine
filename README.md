# Prediction Engine — Phase 0

ERP-native demand forecasting engine. Phase 0 covers the data layer, a baseline
model portfolio, and the evaluation scoreboard. No foundation models, no agents,
no UI — those arrive in later phases, and this phase exists to set the bar they
must beat.

## Setup

```bash
uv venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
```

## Running

```bash
engine ingest   --client teknic_mumbai --file data/raw/Demand_History_3_Years.xlsx
engine features --client teknic_mumbai
engine backtest --client teknic_mumbai
engine report   --client teknic_mumbai
engine forecast --client teknic_mumbai --horizon 3
```

`ingest` is idempotent per source file. `features` profiles every series and
classifies its demand pattern. `backtest` runs the rolling-origin evaluation and
persists a scoreboard. `report` re-renders any stored run. `forecast` projects
forward using the per-series model chosen by the latest backtest.

## Architecture

| Module | Responsibility |
| --- | --- |
| `config/` | Typed client profiles (Pydantic, `extra="forbid"` — typos fail at load) |
| `ingest/` | Source readers mapping any file into the canonical model |
| `store/` | Per-client DuckDB database plus Parquet snapshots |
| `features/` | Calendar features, supervised panel construction, demand-pattern profiling |
| `forecast/` | Model portfolio behind one `Forecaster` interface, plus arbitration |
| `evaluate/` | Metrics, rolling-origin backtest, scoreboard rendering and persistence |

Two contracts hold the pipeline together. Ingest maps every source into
`dim_series` + `fact_demand`, and nothing downstream reads a source file
directly. Every model implements `Forecaster.fit_predict` and returns
`(series_id, period, horizon, forecast)`, with the base class enforcing dtypes
and non-negativity — so adding XGBoost, or a foundation model in Phase 1, is a
new file plus a registry entry.

## Model portfolio

| Model | Role |
| --- | --- |
| `seasonal_naive` | Baseline. Every other model is scored against it (Forecast Value Add) |
| `naive` | Last observed value |
| `auto_ets` | Automatic exponential smoothing, per series |
| `croston` | Intermittent demand |
| `lightgbm_global` | One pooled model across all series, horizon as a feature |
| `auto_arima` | Registered, off by default |

Arbitration picks a model per series from backtest results, shrunk toward the
best overall model: a challenger must beat the champion by
`models.arbitration.min_improvement_pct` on that series before it is selected.
Without that margin, per-series selection fits noise and scores *worse*
out-of-sample than simply using the champion everywhere.

## Client configuration

One YAML file per client in `config/clients/`. It holds what the engine cannot
infer — hierarchy keys, calendar granularity, currency, evaluation windows,
model portfolio. It deliberately does **not** hold a demand-pattern setting:
patterns vary series by series within a single client, so the engine classifies
each series from its own history (ADI/CV² for intermittency, STL for trend and
seasonality strength).

`exogenous_signals` and `cost_model` are present but empty, so Phase 2's context
layer and the decision layer extend this file rather than replace it.

## Evaluation

Rolling-origin backtest with an expanding training window; every model at every
origin sees only data up to that cutoff. Metrics are reported two ways, because
they answer different questions:

- **MAE / WAPE** — volume-weighted. Reflects business impact, and on this panel
  a handful of very large series dominate it.
- **Median series WAPE** — every series weighted equally. Reflects per-series
  model quality.

The final origin is excluded from model arbitration, so the arbitrated portfolio
is scored on an origin it was not selected on.

## Testing

```bash
pytest
ruff check .
```
