# Forecast Engine — Phase 0

ERP-native demand forecasting engine. Phase 0 delivers the data layer, a baseline
model portfolio, and the evaluation scoreboard — terminal-driven, no UI.

There are deliberately **no LLMs, agents, or foundation models** in this phase.
Its job is to establish the bar that later phases must beat: every model is
scored against a seasonal-naive baseline on a rolling-origin backtest, so the
value added by anything more sophisticated becomes provable rather than assumed.

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick start](#2-quick-start)
3. [Detailed setup](#3-detailed-setup)
4. [Running the engine](#4-running-the-engine)
5. [Running the tests](#5-running-the-tests)
6. [Verifying it actually works](#6-verifying-it-actually-works)
7. [Command reference](#7-command-reference)
8. [Configuration](#8-configuration)
9. [Extending the engine](#9-extending-the-engine)
10. [Project structure](#10-project-structure)
11. [Troubleshooting](#11-troubleshooting)
12. [Scope and limits](#12-scope-and-limits)

---

## 1. Prerequisites

| Requirement | Notes |
| --- | --- |
| Python 3.11 or newer | Developed and tested on 3.13.2 |
| Git | To clone the repository |
| ~1 GB free disk | Mostly the virtual environment (LightGBM, statsforecast, Polars) |
| CPU only | No GPU required. No network access needed after install |

Check your Python version:

```bash
python --version
```

If it reports anything below 3.11, install a newer Python before continuing.

---

## 2. Quick start

For someone who just wants it running. Each step is explained in detail further
down.

**Windows (PowerShell):**

```powershell
git clone https://github.com/plutoTaco/forecast-engine.git
cd forecast-engine
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\engine.exe ingest   --client teknic_mumbai --file data\raw\Demand_History_3_Years.xlsx
.venv\Scripts\engine.exe features --client teknic_mumbai
.venv\Scripts\engine.exe backtest --client teknic_mumbai
```

**macOS / Linux:**

```bash
git clone https://github.com/plutoTaco/forecast-engine.git
cd forecast-engine
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/engine ingest   --client teknic_mumbai --file data/raw/Demand_History_3_Years.xlsx
.venv/bin/engine features --client teknic_mumbai
.venv/bin/engine backtest --client teknic_mumbai
```

Total runtime after install is about 45 seconds. The last command prints the
full scoreboard.

---

## 3. Detailed setup

### 3.1 Clone the repository

```bash
git clone https://github.com/plutoTaco/forecast-engine.git
cd forecast-engine
```

The sample dataset (`data/raw/Demand_History_3_Years.xlsx`) is included in the
repository, so there is nothing else to download.

### 3.2 Create a virtual environment

Two options. Both produce the same result — pick one.

**Option A — standard library `venv`** (simplest, works everywhere):

```bash
python -m venv .venv
```

**Option B — `uv`** (much faster; this is what the project uses day to day):

```bash
pip install uv
uv venv
```

### 3.3 Install the package and its dependencies

Installing in editable mode (`-e`) means changes to the source take effect
without reinstalling.

**Windows:**

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

**macOS / Linux:**

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

**With uv** (note the explicit `--python` flag — without it, uv may target your
system Python instead of the virtual environment):

```bash
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"   # Windows
uv pip install --python .venv/bin/python -e ".[dev]"           # macOS / Linux
```

`[dev]` adds `pytest` and `ruff`. Omit it for a runtime-only install.

### 3.4 Activating the environment (optional but convenient)

Activating puts `engine`, `pytest`, and `python` on your PATH so you can type
short commands instead of full paths.

```powershell
.venv\Scripts\activate.ps1     # Windows PowerShell
.venv\Scripts\activate.bat     # Windows cmd.exe
source .venv/bin/activate      # macOS / Linux
```

If PowerShell refuses with an execution-policy error, either allow scripts for
the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

…or skip activation entirely and call `.venv\Scripts\engine.exe` directly. Both
work identically. **The rest of this README assumes the environment is
activated** — if it isn't, prefix commands with `.venv\Scripts\` (Windows) or
`.venv/bin/` (macOS/Linux).

### 3.5 Confirm the install

```bash
engine --help
```

You should see the list of commands: `backtest`, `clients`, `features`,
`forecast`, `ingest`, `report`.

---

## 4. Running the engine

The pipeline is four steps. Run them in order the first time; afterwards any
step can be re-run on its own.

### Step 1 — Ingest

Loads the source workbook into the canonical store.

```bash
engine ingest --client teknic_mumbai --file data/raw/Demand_History_3_Years.xlsx
```

Expected output:

```
-> Ingesting Demand_History_3_Years.xlsx for Teknic — Mumbai Operations
   done in 0.20s
   reconciled 10,692 rows (297 series, 2023-09-01 to 2026-08-01); quantity sum 37,372,528
   store: .../data/processed/teknic_mumbai/engine.duckdb
```

The reconciliation line is the point of this step: row count and quantity sum
are compared before and after the canonical mapping, and any mismatch aborts
the run. Re-running this command is safe — it replaces the file's rows rather
than appending them.

### Step 2 — Profile the series

Classifies each series' demand pattern from its own history.

```bash
engine features --client teknic_mumbai
```

Expected output:

```
-> Profiling 297 series
   detected patterns: Seasonal 52, Growing 49, Declining 49, Stable 49, Intermittent 49, Volatile 49

Pattern classifier — agreement with source labels: 99.3% (gate: 75%)
```

…followed by a confusion matrix. Because the sample file carries ground-truth
pattern labels, the classifier can be scored rather than trusted. Real client
data will not have those labels, and the agreement line is simply skipped.

### Step 3 — Backtest

Runs the rolling-origin evaluation across all enabled models and persists the
scoreboard. This is the slow step, roughly 31 seconds.

```bash
engine backtest --client teknic_mumbai
```

It prints five tables:

| Table | What it tells you |
| --- | --- |
| Accuracy across all origins | Every model's error, and its Forecast Value Add against the baseline |
| FVA by demand pattern (volume-weighted) | Where each model wins or loses, weighted by volume |
| FVA by demand pattern (median series WAPE) | The same, with every series weighted equally |
| Final held-out origin | The honest portfolio number — scored on an origin it was not selected on |
| Arbitration | Which model won how many series |

The headline result: **portfolio +45.9% FVA** over seasonal-naive on the
held-out origin.

### Step 4 — Forecast

Projects forward from the end of history, using the model arbitration chose per
series.

```bash
engine forecast --client teknic_mumbai --horizon 3
```

Produces 891 forecasts (297 series × 3 months) and stores them.

### Re-reading results without recomputing

```bash
engine report --client teknic_mumbai            # latest backtest run
engine report --client teknic_mumbai --run-id 2 # a specific earlier run
```

---

## 5. Running the tests

```bash
pytest
```

Expected: **42 passed** in roughly 2 seconds.

For the full list of what each test checks:

```bash
pytest -v
```

Nothing should be skipped. A skip means
`data/raw/Demand_History_3_Years.xlsx` is missing — two integration tests
depend on it.

Lint and formatting:

```bash
ruff check .
ruff format --check .
```

### What the tests cover

| File | Protects |
| --- | --- |
| `test_config.py` | Typos and misconfiguration fail at load time |
| `test_ingest.py` | Duplicates, negatives, nulls, orphan rows rejected; re-ingest is idempotent |
| `test_forecast.py` | Output contract holds for every model; arbitration respects its margin |
| `test_pattern.py` | Classification precedence — intermittency checked before trend/seasonality |
| `test_metrics.py` | MAE, WAPE, and FVA computed as defined |
| `test_backtest.py` | **No leakage** — forecast periods always sit strictly after the training cutoff |

---

## 6. Verifying it actually works

Passing tests prove the code does what it was written to do. These checks prove
the *claims* — run them if you want to falsify rather than confirm.

### 6.1 Ingest really is idempotent

Run ingest twice and confirm the row count does not change:

```bash
engine ingest --client teknic_mumbai --file data/raw/Demand_History_3_Years.xlsx
engine ingest --client teknic_mumbai --file data/raw/Demand_History_3_Years.xlsx
```

Both runs must report 10,692 rows. Run it five more times — still 10,692.

### 6.2 Configuration fails loudly, not silently

Open `config/clients/teknic_mumbai.yaml`, rename `seasonal_strength` to
`seasonal_strengths`, and run any command:

```bash
engine features --client teknic_mumbai
```

You should get a validation error naming the unexpected key — not a silent
fallback to a default. Undo the edit afterwards.

### 6.3 Results are stored, not just printed

Start Python (`python`) and run:

```python
import duckdb
con = duckdb.connect("data/processed/teknic_mumbai/engine.duckdb", read_only=True)
rows = con.execute("select scope_value, round(value,1) from backtest_metrics "
                   "where scope_type='series' and metric='mae' "
                   "and model='lightgbm_global' order by value desc limit 10").fetchall()
for series, mae in rows: print(series, mae)
```

Expect ten series ids, worst first, with `S0258` around 54,580 at the top —
those are the large moulding-component series that dominate the volume-weighted
metrics.

### 6.4 Accuracy degrades at longer horizons

If it doesn't, something is wrong:

```bash
engine backtest --client teknic_mumbai --horizon 6 --origins 2
```

### 6.5 Rebuild everything from scratch

Deleting `data/processed` removes only derived data; the source workbook is
untouched.

```powershell
Remove-Item -Recurse -Force data\processed    # Windows
rm -rf data/processed                          # macOS / Linux
```

Then re-run steps 1–3. Total: about 42 seconds.

---

## 7. Command reference

| Command | Purpose | Key options |
| --- | --- | --- |
| `engine clients` | List configured client profiles | — |
| `engine ingest` | Load a source file into the canonical store | `--client`, `--file` |
| `engine features` | Profile series and classify demand patterns | `--client` |
| `engine backtest` | Rolling-origin evaluation, persists scoreboard | `--client`, `--horizon`, `--origins` |
| `engine report` | Re-render a stored run | `--client`, `--run-id` |
| `engine forecast` | Forecast forward from end of history | `--client`, `--horizon`, `--model` |

`--horizon` and `--origins` on `backtest` override the client profile for that
run only; they do not change the file.

`--model` on `forecast` forces one model instead of per-series arbitration —
useful for comparing a single model's forward projection.

---

## 8. Configuration

One YAML file per client in `config/clients/`. It is validated on load with
`extra="forbid"`, so a misspelled key is an error rather than a silent default.

The file holds what the engine **cannot infer**:

| Section | Controls |
| --- | --- |
| `source` | Which reader to use and how source columns map to canonical fields |
| `calendar` | Granularity (monthly/weekly/daily) and fiscal year start |
| `hierarchy` | Which columns identify a series, and which group it |
| `evaluation` | Horizon, origin count, minimum training window |
| `patterns` | Thresholds for the demand-pattern classifier |
| `models` | Which forecasters run, which is the baseline, arbitration margin |

It deliberately does **not** hold a demand-pattern setting. Patterns vary series
by series *within* a single client — this dataset has one customer with series
in all six pattern classes at once — so the engine classifies each series from
its own history instead.

`exogenous_signals` and `cost_model` exist but are empty, so Phase 2's context
layer and the decision layer extend this file rather than replace it.

### Onboarding a new client

1. Copy `config/clients/teknic_mumbai.yaml` to `config/clients/<new_id>.yaml`
2. Set `client_id` to match the filename exactly (validated on load)
3. Adjust `source.column_map` to match the new source file's column headings
4. Run `engine ingest --client <new_id> --file <path>`

Each client gets its own DuckDB database under `data/processed/<client_id>/`,
so tenant isolation is a property of the filesystem layout.

---

## 9. Extending the engine

### Adding a forecasting model

Every model implements one interface, so a new model is a new file plus a
registry entry — nothing else in the pipeline changes.

```python
# src/engine/forecast/my_model.py
from engine.forecast.base import ForecastContext, Forecaster, register_forecaster

@register_forecaster
class MyForecaster(Forecaster):
    name = "my_model"

    def fit_predict(self, history, horizon, ctx: ForecastContext):
        # history: series_id, period, quantity (everything up to the cutoff)
        # return:  series_id, period, horizon, forecast
        ...
```

Import it in `src/engine/forecast/__init__.py`, then add `"my_model"` to
`models.enabled` in a client profile. The base class enforces dtypes and clips
negative forecasts centrally, so implementations do not repeat that.

Swapping LightGBM for XGBoost works exactly this way — the panel construction,
per-series scaling, and rescaling in `lightgbm_model.py` are model-agnostic.

### Adding a data source

Implement `SourceReader` in `src/engine/ingest/`, decorate it with
`@register_reader`, and set `source.reader` in the client profile. The reader's
only job is producing the canonical `dim_series` + `fact_demand` frames;
nothing downstream reads source files directly.

---

## 10. Project structure

```
config/clients/          Client profiles (YAML, validated on load)
data/raw/                Source files
data/processed/          Generated — DuckDB + Parquet, gitignored
src/engine/
  config/                Typed profile schema and loader
  ingest/                Source readers, canonical schema, validation
  store/                 DuckDB persistence (all SQL lives here)
  features/              Calendar features, panel construction, pattern profiling
  forecast/              Model portfolio and per-series arbitration
  evaluate/              Metrics, backtest, scoreboard
  cli.py                 Terminal interface
tests/                   42 tests
```

### The model portfolio

| Model | Role |
| --- | --- |
| `seasonal_naive` | Baseline — every other model is scored against it |
| `naive` | Last observed value |
| `auto_ets` | Automatic exponential smoothing, per series |
| `croston` | Intermittent demand |
| `lightgbm_global` | One pooled model across all series, horizon as a feature |
| `auto_arima` | Registered, disabled by default |

Arbitration picks a model per series, **shrunk toward the best overall model**:
a challenger must beat the champion by `models.arbitration.min_improvement_pct`
on that series before it is selected. Without that margin, per-series selection
fits noise and scores *worse* out-of-sample than using the champion everywhere —
measured at 794 vs 692 MAE before the margin was introduced.

### How accuracy is measured

Rolling-origin backtest with an expanding training window; at every origin each
model sees only data up to that cutoff. Two metric families are reported because
they answer different questions:

- **MAE / WAPE** — volume-weighted. Reflects business impact; on this dataset a
  handful of very large series dominate it.
- **Median series WAPE** — every series weighted equally. Reflects per-series
  model quality.

The final origin is excluded from arbitration, so the portfolio is scored on an
origin it was not selected on.

---

## 11. Troubleshooting

**`ModuleNotFoundError: No module named 'engine'`**
The editable install did not complete, or you are using a different Python than
the one you installed into. Re-run the install command from §3.3 and confirm
with `python -c "import engine; print(engine.__version__)"`.

**`uv` installs into the wrong Python**
Pass `--python .venv/Scripts/python.exe` (Windows) or `--python .venv/bin/python`
(macOS/Linux) explicitly. Without it, uv may target the system interpreter and
fail with a permissions error.

**PowerShell: "running scripts is disabled on this system"**
Either run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or skip
activation and call `.venv\Scripts\engine.exe` directly.

**`UnicodeEncodeError` when printing tables**
The engine reconfigures its own output to UTF-8, so its commands are unaffected.
If you hit this in your own Python session on Windows, set `PYTHONIOENCODING=utf-8`
before starting Python.

**`No data ingested yet. Run engine ingest first.`**
Expected if `data/processed/` was deleted or never created. Re-run step 1.

**Backtest fails with "too few periods"**
The configured horizon and origin count need more history than the data has.
Lower `evaluation.horizon`, `evaluation.origins`, or `evaluation.min_train_periods`
in the client profile.

---

## 12. Scope and limits

Stated plainly, so nothing here is mistaken for more than it is.

- **The sample data is synthetic.** Realistically shaped, but generated. The
  accuracy figures establish that the pipeline and measurement work — not what
  accuracy real demand will achieve.
- **36 months is short.** Three seasonal cycles is the minimum for credible
  yearly seasonality, and it limits how many origins can be backtested.
- **Classifier thresholds were tuned on the labelled sample file** and need
  revalidation against real, unlabelled data.
- **No exogenous signals, no uncertainty intervals, no decisions.** Forecasts are
  point estimates in units. Weather, festival calendars, conformal intervals, and
  ordering recommendations arrive in later phases by design.
- **Intermittent series remain hard.** That slice is where foundation models are
  most likely to add value next.

Phase 1 adds Chronos-2 and TimesFM for cold-start and long-tail series,
arbitrated against these baselines by the same scoreboard — with this run as the
number they have to beat.
