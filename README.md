# Forecast Engine

ERP-native demand forecasting engine. Terminal-driven, no UI.

**Phase 0** delivered the data layer, a baseline model portfolio, and the
evaluation scoreboard — the bar every later phase has to beat.

**Phase 1** adds Chronos-2, a pretrained time-series foundation model, and the
machinery to test where it actually helps: volume segmentation, stratified
arbitration, and a cold-start experiment. On the synthetic panel it won the long
tail and short history exactly as the literature predicts. On **real** Walmart
data it wins neither, and a pooled gradient-boosted model beats it everywhere —
see [What M5 changed](#14-what-m5-changed). The contrast between those two
results is the most useful thing this project has produced so far.

There are still **no LLMs and no agents** here. Chronos-2 is a pretrained
forecasting model, not a language model; the LLM-driven context and explanation
layers belong to later phases.

Every model is scored against a seasonal-naive baseline on a rolling-origin
backtest, so added sophistication has to prove itself rather than be assumed.

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
13. [What Phase 1 measured](#13-what-phase-1-measured)
14. [What M5 changed](#14-what-m5-changed)
15. [Reading from PostgreSQL](#15-reading-from-postgresql)
16. [Publishing forecasts back](#16-publishing-forecasts-back)
17. [The decision layer](#17-the-decision-layer)

---

## 1. Prerequisites

| Requirement | Notes |
| --- | --- |
| Python 3.11 or newer | Developed and tested on 3.13.2 |
| Git | To clone the repository |
| ~4 GB free disk | Virtual environment (~3 GB, mostly PyTorch) plus ~500 MB of model weights |
| CPU only | No GPU required — Chronos-2 forecasts all 297 series in ~1.6s on CPU |
| Network, first run only | Chronos-2 weights download from Hugging Face once, then cache |
| PostgreSQL 14+ | **Optional.** Only to read demand history from a database — see [section 15](#15-reading-from-postgresql) |

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
.venv\Scripts\python.exe -m pip install -e ".[dev,foundation]"
.venv\Scripts\engine.exe ingest   --client teknic_mumbai --file data\raw\Demand_History_3_Years.xlsx
.venv\Scripts\engine.exe features --client teknic_mumbai
.venv\Scripts\engine.exe backtest --client teknic_mumbai
```

**macOS / Linux:**

```bash
git clone https://github.com/plutoTaco/forecast-engine.git
cd forecast-engine
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,foundation]"
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
.venv\Scripts\python.exe -m pip install -e ".[dev,foundation]"
```

**macOS / Linux:**

```bash
.venv/bin/python -m pip install -e ".[dev,foundation]"
```

**With uv** (note the explicit `--python` flag — without it, uv may target your
system Python instead of the virtual environment):

```bash
uv pip install --python .venv/Scripts/python.exe -e ".[dev,foundation]"   # Windows
uv pip install --python .venv/bin/python -e ".[dev,foundation]"           # macOS / Linux
```

`[dev]` adds `pytest` and `ruff`; omit it for a runtime-only install.
`[postgres]` adds the ADBC driver needed to read from a database; omit it
for a file-fed install.
`[foundation]` adds Chronos-2 and PyTorch (~3 GB). To skip it, install plain
`-e "."` and remove `chronos2` from `models.enabled` in the client profile — the
rest of the portfolio runs without it, and the engine will tell you clearly if
the model is enabled but not installed.

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

You should see the list of commands: `backtest`, `clients`, `coldstart`,
`features`, `forecast`, `ingest`, `report`.

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
   volume segments: Head 20 series / 79.6% of volume, Mid 51 series / 15.4%, Tail 226 series / 5.0%

Pattern classifier — agreement with source labels: 99.3% (gate: 75%)
```

…followed by a confusion matrix. Because the sample file carries ground-truth
pattern labels, the classifier can be scored rather than trusted. Real client
data will not have those labels, and the agreement line is simply skipped.

The volume segments are a textbook Pareto curve — 20 series carry 80% of demand
while 226 carry the last 5%. That split matters because models differ sharply
across it.

### Step 3 — Backtest

Runs the rolling-origin evaluation across all enabled models and persists the
scoreboard. This is the slow step, roughly 31 seconds.

```bash
engine backtest --client teknic_mumbai
```

It prints:

| Table | What it tells you |
| --- | --- |
| Accuracy across all origins | Every model's error, and its Forecast Value Add against the baseline |
| FVA by demand pattern (volume-weighted) | Where each model wins or loses, weighted by volume |
| FVA by demand pattern (median series WAPE) | The same, with every series weighted equally |
| FVA by volume segment | Head / Mid / Tail — where foundation models earn their place |
| Final held-out origin | The honest portfolio number — scored on an origin it was not selected on |
| Portfolio by pattern and by segment | The phase-gate view: the portfolio must not lose on any slice |
| Arbitration | Which model won how many series |

The headline result: **portfolio +45.7% FVA** over seasonal-naive on the
held-out origin, with a median series WAPE of 0.120.

It also prints the default model chosen for each volume segment, e.g.
`Head -> lightgbm_global, Mid -> lightgbm_global, Tail -> chronos2`.

### Step 3b — Cold-start experiment

```bash
engine coldstart --client teknic_mumbai
```

Re-runs the backtest with each series truncated to 3, 6, 12, 24, and 36 periods
of history, showing how each model degrades as history runs out. This is the
experiment that justifies including a foundation model at all — see
[What Phase 1 measured](#13-what-phase-1-measured). Takes a few minutes.

`engine backtest --max-history N` runs a single depth if you want just one.

### Step 4 — Forecast

Projects forward from the end of history, using the model arbitration chose per
series.

```bash
engine forecast --client teknic_mumbai --horizon 3
```

Produces 891 forecasts (297 series × 3 months) and stores them.

**The table printed on screen is only a preview of the first 8 rows.** All 891
are saved in the store. To get every row as a file:

```bash
engine forecast --client teknic_mumbai --horizon 3 --export forecasts.csv
```

`.csv` opens directly in Excel; `.parquet` is for other tools. Each row carries
the customer, item code and description alongside the forecast and its range,
so the file reads without looking anything up. An unsupported file name is
refused before the forecast runs.

Some rows have blank `q10`/`q90`: AutoETS and Croston do not produce ranges yet,
so items assigned to them get a point forecast only.

### Re-reading results without recomputing

```bash
engine report --client teknic_mumbai            # latest backtest run
engine report --client teknic_mumbai --run-id 2 # a specific earlier run
```

### Opening the database directly

Each client's results live in one DuckDB file,
`data/processed/<client_id>/engine.duckdb`. To browse it:

```bash
engine ui --client teknic_mumbai
```

Your browser opens DuckDB's viewer — a SQL notebook with all 11 tables listed
under `engine`. It opens the file read-only, so nothing in it can be changed by
accident. Press Enter in the terminal to close it. The viewer itself loads from
`ui.duckdb.org`, so it needs internet access; your data stays on your machine.

If port 4213 is taken — usually by a viewer you left open — the command says so
rather than silently showing you the old one. Close that one, or use
`--port 4214`. `--no-browser` starts it without opening a browser window.

For example, the latest forecast:

```sql
SELECT * FROM forecasts WHERE run_id = (SELECT max(run_id) FROM forecasts);
```

**Close the viewer before running any other `engine` command.** DuckDB lets
only one process write to a file, and even a read-only viewer blocks that: an
engine command started while the viewer is open fails with a lock error naming
the viewer's process ID.

Opening the file yourself with `duckdb.connect(path, read_only=True)` and then
starting the viewer does **not** work: read-only applies to the whole session,
so the viewer cannot create the small database it keeps its notebooks in, and
fails with `Catalog "_duckdb_ui" does not exist`. `engine ui` avoids this by
attaching only the engine's file read-only.

If you prefer a desktop tool, [DBeaver](https://dbeaver.io) opens `.duckdb`
files through its DuckDB driver, and the standalone DuckDB command line installs
with `winget install DuckDB.cli` (then `duckdb -ui <file>`).

---

## 5. Running the tests

```bash
pytest
```

Expected: **116 passed, 9 deselected** in roughly 13 seconds.

For the full list of what each test checks:

```bash
pytest -v
```

The five deselected tests need something the default run cannot assume:

| Marker | Needs | Run with |
| --- | --- | --- |
| `slow` | Chronos-2 weights cached locally | `pytest -m slow` |
| `integration` | A live PostgreSQL, with `ENGINE_PG_DSN` and `ENGINE_PG_WRITE_DSN` set | `pytest -m integration` |

A *skip* is different from a deselect and means
`data/raw/Demand_History_3_Years.xlsx` is missing — several tests depend on it.

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
rows = con.execute(
    "select scope_value, round(value,1) from backtest_metrics "
    "where scope_type='series' and metric='mae' "
    "and model='lightgbm_global' order by value desc limit 10"
).fetchall()
for series, mae in rows:
    print(series, mae)
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
| `engine ingest` | Load a source into the canonical store | `--client`, `--file` (file readers only) |
| `engine features` | Profile series and classify demand patterns | `--client` |
| `engine backtest` | Rolling-origin evaluation, persists scoreboard | `--client`, `--horizon`, `--origins`, `--max-history` |
| `engine coldstart` | Accuracy vs. length of training history | `--client`, `--depths` |
| `engine report` | Re-render a stored run | `--client`, `--run-id` |
| `engine forecast` | Forecast forward from end of history | `--client`, `--horizon`, `--model`, `--export` |
| `engine decide` | Score models on decision cost and set stock levels | `--client`, `--run-id` |
| `engine publish` | Write a stored forecast into the ERP database | `--client`, `--run-id` |
| `engine ui` | Browse the client's database in DuckDB's viewer, read-only | `--client`, `--port`, `--no-browser` |

`--horizon` and `--origins` on `backtest` override the client profile for that
run only; they do not change the file.

`--model` on `forecast` forces one model instead of per-series arbitration —
useful for comparing a single model's forward projection.

`--file` on `ingest` is required for file readers and rejected for database
readers, which are located by the client profile instead. Passing it to a
database reader is an error rather than a silent no-op.

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
| `segments` | Cumulative volume-share boundaries for Head / Mid / Tail |
| `models` | Which forecasters run, the baseline, arbitration, and per-model settings |

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

A reader that is located by its client profile rather than by a path — a
database, an API — sets `requires_path = False`; the CLI then rejects `--file`
for it instead of ignoring it. See
[section 15](#15-reading-from-postgresql) for the PostgreSQL reader.

---

## 10. Project structure

```
config/clients/          Client profiles (YAML, validated on load)
data/raw/                Source files
data/processed/          Generated — DuckDB + Parquet, gitignored
scripts/                 Data preparation and loading utilities
  sql/                   PostgreSQL schema, view and role definitions
src/engine/
  config/                Typed profile schema and loader
  ingest/                Source readers, canonical schema, validation
  store/                 DuckDB persistence, file export, PostgreSQL write-back
  features/              Calendar features, panel construction, pattern profiling
  forecast/              Model portfolio and per-series arbitration
  evaluate/              Metrics, backtest, scoreboard
  decide/                Inventory policy, decision cost, reporting
  cli.py                 Terminal interface
tests/                   125 tests
```

### The model portfolio

| Model | Role |
| --- | --- |
| `seasonal_naive` | Baseline — every other model is scored against it |
| `naive` | Last observed value |
| `auto_ets` | Automatic exponential smoothing, per series |
| `croston` | Intermittent demand |
| `lightgbm_global` | One pooled model across all series, horizon as a feature |
| `chronos2` | Amazon Chronos-2 (120M, Apache-2.0), pretrained and run zero-shot |
| `auto_arima` | Registered, disabled by default |

Arbitration picks a model per series in two stages.

**A default per stratum.** Each volume segment gets its own champion, scored on
median series WAPE — the model that is best for a *typical* series in that
segment. A single global default cannot work once the portfolio holds models with
complementary strengths: measured on this data, defaulting everything to the
narrow global winner cost 660.1 MAE versus 631.5 for per-segment defaults.

**A per-series override with a margin.** A challenger must beat its segment's
champion by `models.arbitration.min_improvement_pct` on that series before it is
selected. Without the margin, per-series selection fits noise and scores *worse*
out-of-sample than using the champion everywhere — 794 vs 692 MAE when measured.

Both stages exist because they were measured, not assumed. Blending was tested
and rejected: an equal-weight mean of the two strongest models scored 684.7 MAE,
worse than selection at 631.5.

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
- **Decision costs are not currency.** The cost model is a ratio, not rupees.
  Teknic's source carries no prices, so the engine reports cost in the units the
  ratio is expressed in and does not pretend otherwise. See
  [section 17](#17-the-decision-layer).
- **13.5% of the portfolio gets no stock policy.** Arbitration selects on
  accuracy, and two of the enabled models produce no quantiles, so series whose
  champion is one of them have no distribution to stock against. The engine
  reports this rather than hiding it; fixing it means arbitration knowing that a
  decision needs a distribution.
- **Intermittent series remain hard.** A median series WAPE near 1.0 on that
  slice reflects the genuine difficulty of mostly-zero demand.
- **Context features are supported but thin.** Exogenous signals are ingested
  and used where a client supplies them; on M5 they added 1.4%. Teknic has none.
- **The database round trip is demand only.** Demand history is read from
  PostgreSQL and forecasts are published back, but context features still come
  from files, and re-ingest replaces everything under a source label rather than
  merging. See sections [15](#15-reading-from-postgresql)
  and [16](#16-publishing-forecasts-back).

---

## 13. What Phase 1 measured

Phase 1's premise was that a pretrained foundation model earns its place on
cold-start and long-tail series while a tuned pooled model keeps winning on
high-volume ones. That is testable, so it was tested rather than assumed.

### Chronos-2 does not win overall

On the held-out origin, Chronos-2 (694.2 MAE) and LightGBM (691.7) are within
0.4% of each other — statistically indistinguishable. Adding a foundation model
did **not** move the headline number: the portfolio went from 629.4 MAE in Phase
0 to 631.5 in Phase 1. What did improve is median series WAPE, 0.127 → 0.120,
because the gain lands on many small series rather than the few large ones that
dominate a volume-weighted total.

### …and on real data it does not win anywhere

Everything below this line was measured on the **synthetic** panel. Re-running
the same portfolio on M5 — 12,196 real Walmart item-store series — reversed the
conclusion. See [What M5 changed](#14-what-m5-changed). The synthetic findings
are kept here because the contrast between them is the most useful result the
project has produced.

### On synthetic data it won exactly where the research predicted

Forecast Value Add by volume segment (median series WAPE, vs seasonal-naive):

| Model | Head | Mid | Tail |
| --- | --- | --- | --- |
| `lightgbm_global` | −15.0% | +38.7% | +30.7% |
| `chronos2` | −29.2% | +36.6% | **+33.8%** |

Chronos-2 wins the long tail and loses the head. That is the split the
arbitration now encodes: `Head -> lightgbm_global`, `Tail -> chronos2`.

### The cold-start crossover is at roughly two years

`engine coldstart` truncates every series' history and re-runs the backtest.
Median series WAPE, lower is better:

| History | 3 periods | 6 | 12 | 24 | 36 |
| --- | --- | --- | --- | --- | --- |
| `chronos2` | **0.188** | **0.199** | **0.183** | 0.147 | 0.133 |
| `lightgbm_global` | 0.222 | 0.215 | 0.200 | **0.144** | 0.137 |
| `auto_ets` | 1.000 | 1.000 | 0.184 | 0.183 | 0.149 |

Below about 24 periods the pretrained model wins; above it the pooled learner
takes over. `auto_ets` cannot fit a seasonal model on 3–6 observations at all.
This is the clearest evidence for keeping a foundation model in the portfolio:
not because it is better in general, but because it degrades gracefully when
there is nothing to learn from.

### Caveat on these numbers

The per-segment champion metric was chosen *after* seeing held-out results,
which makes 631.5 mildly optimistic — the holdout informed a design decision, so
it is no longer a fully untouched test set. The reasoning behind the choice is
independent of the score (a default should represent the typical series in its
segment, not the segment's largest), and the same scale-domination lesson was
already established in Phase 0. But the honest reading is that Phase 1 leaves
overall accuracy flat and improves the long tail — not that it lifted the
headline number.

---

## 14. What M5 changed

The synthetic conclusions did not survive real data. This section supersedes
section 13 wherever the two disagree.

### Setup

M5 is Walmart's public competition dataset — 30,490 item-store series, daily,
with real prices, SNAP benefit days and event calendars. Prepared to weekly
California: **12,196 series, 2.7M rows, 281 weeks**.

```bash
python scripts/prepare_m5.py --states CA
engine ingest   --client m5_walmart --file data/raw/m5_walmart.parquet
engine features --client m5_walmart
engine backtest --client m5_walmart
```

### Real demand looks nothing like the synthetic panel

| | Synthetic | M5 |
| --- | --- | --- |
| Volatile + Intermittent | 33% | **88%** |
| Clean trend / seasonal | 50% | **6%** |
| Series carrying 80% of volume | 7% | 32% |
| Median series WAPE achieved | 0.120 | 0.443 |

The generator produced a tidy uniform mix of six archetypes. Real retail is
overwhelmingly noisy and zero-inflated, and roughly four times harder.

### The portfolio still works

| Model | MAE | WAPE | FVA | Median series WAPE |
| --- | --- | --- | --- | --- |
| **portfolio** | **3.9** | **0.362** | **+42.2%** | **0.443** |
| `lightgbm_global` | 4.2 | 0.385 | +38.5% | 0.460 |
| `chronos2` | 4.2 | 0.391 | +37.5% | 0.481 |
| `croston` | 4.4 | 0.409 | +34.6% | 0.491 |
| `seasonal_naive` | 6.8 | 0.626 | — | 0.676 |

Arbitration beats every individual model, and no pattern or volume slice loses
to the baseline. That part of Phase 1 holds up.

### The foundation model does not

Forecast Value Add by volume segment (median series WAPE):

| Model | Head | Mid | Tail |
| --- | --- | --- | --- |
| `lightgbm_global` | **+39.2%** | **+32.7%** | **+24.0%** |
| `chronos2` | +37.5% | +29.7% | +19.6% |

On synthetic data Chronos-2 won the tail. On real data it loses the tail, and
LightGBM wins every segment. Arbitration picks LightGBM for 66.2% of series and
Chronos-2 for 7.5% — down from 39.4%.

Cold start shows no crossover either. Median series WAPE by training history:

| Model | 8 periods | 26 | 52 | 104 |
| --- | --- | --- | --- | --- |
| `lightgbm_global` | **0.462** | **0.458** | **0.458** | **0.460** |
| `croston` | 0.480 | 0.481 | 0.484 | 0.485 |
| `chronos2` | 0.509 | 0.491 | 0.484 | 0.484 |

LightGBM wins at every depth, including eight weeks of history.

### A hypothesis that was tested and failed

The obvious explanation was panel width: with 12,196 series a pooled model has
~100k training rows even at 8 periods each, while 297 synthetic series give
~900. Holding history fixed at 26 periods and varying the number of series:

| Series in panel | 300 | 1,200 | 5,000 | 12,196 |
| --- | --- | --- | --- | --- |
| `lightgbm_global` | **0.467** | **0.452** | **0.461** | **0.458** |
| `chronos2` | 0.507 | 0.484 | 0.496 | 0.491 |

LightGBM wins at 300 series too, so width is **not** the explanation. The more
likely remaining account is that the synthetic panel's smooth parametric
patterns are exactly what a pretrained model extrapolates well, while real
zero-inflated count demand is better served by a pooled learner on lag features.
That has not been tested and should not be stated as fact.

### What this does not settle

- **Weekly aggregation.** Chronos-2 may do better on M5's native daily data,
  closer to its pretraining mix.
- **Covariates.** Chronos-2 accepts them natively and none are being supplied.
  Prices, SNAP days and events are sitting unused in
  `data/raw/m5_walmart_exogenous.parquet`.

Until those are tested, the honest position is that Chronos-2 does not earn a
3 GB dependency on this evidence, but it has not been given its best shot
either. It stays registered and enabled; `models.enabled` is where to drop it.

Phase 2 adds the exogenous context layer and conformal prediction intervals,
with the M5 portfolio run as the number to beat.

---

## 15. Reading from PostgreSQL

The engine can read demand history from a database instead of a file. This is
how an ERP integration actually works: the ERP publishes a view, the engine
reads the view, and the view is the contract between them.

Nothing downstream of ingest changes. Features, models, backtest and
arbitration all consume the canonical `dim_series` + `fact_demand` frames and
cannot tell which reader produced them.

### 15.1 What runs where

| | Role |
|---|---|
| **PostgreSQL** | System of record — the ERP side. The engine has SELECT on one view and nothing else. |
| **DuckDB** | The engine's analytical working store. Unchanged, still one file per client. |

Two databases is the right answer, not an accident: one is transactional and
owned by the ERP, the other is columnar and owned by the engine.

### 15.2 Install the driver

```bash
pip install -e ".[postgres]"
```

This pulls `adbc-driver-postgresql` and `pyarrow` — Arrow-native bulk transfer,
so a large relation does not round-trip through Python objects. The extra is
separable: a file-fed deployment does not need it.

### 15.3 Create the database

Start the PostgreSQL service if it is not running. On Windows this needs an
**elevated** prompt — "Run as administrator":

```powershell
net start postgresql-x64-18
```

Create the database, then the ERP-shaped schema:

```bash
psql -U postgres -c "CREATE DATABASE forecast_erp"
```

```bash
psql -U postgres -d forecast_erp -f scripts/sql/001_erp_source.sql
```

This creates `erp.sales_history` (the base table, shaped like a transactional
system) and `erp.v_demand_history` (the view the engine reads). The view does
the renaming and the monthly aggregation, which is why a source that posts
daily lines would need no engine change.

**The view must expose quantity as `double precision`.** The base table keeps
`NUMERIC(18,3)` — exact decimal, as a system of record should — but the ADBC
driver maps every PostgreSQL `NUMERIC` to an opaque type that arrives as text.
Text parses into a float correctly further downstream, so an uncast view
reconciles perfectly and looks fine while depending entirely on string
formatting. The reader rejects a non-numeric quantity column outright rather
than let that pass.

The script is re-runnable: it drops and recreates the view (`CREATE OR REPLACE`
cannot change a column's type) and reinstates the engine's grant afterwards.

### 15.4 Create the read-only role

The engine must not be able to write to the ERP. Choose a password and set it
in your shell:

```powershell
$env:ENGINE_READER_PASSWORD = "<choose one>"   # PowerShell
set ENGINE_READER_PASSWORD=<choose one>        # cmd.exe
export ENGINE_READER_PASSWORD='<choose one>'   # bash
```

Then create the role. The password is passed to `psql` rather than written into
the script, which is why the script is safe to commit:

```powershell
psql -U postgres -d forecast_erp -v reader_password="$env:ENGINE_READER_PASSWORD" -f scripts/sql/002_engine_reader.sql
```

The grant is SELECT on one view — not the base table, not the schema at large.

### 15.5 Load the data

```powershell
$env:ENGINE_PG_ADMIN_DSN = "postgresql://postgres:<password>@localhost:5432/forecast_erp"
```

```bash
python scripts/load_to_postgres.py --file data/raw/Demand_History_3_Years.xlsx --truncate
```

The admin DSN is deliberately a different credential from the one the engine
uses: loading needs INSERT, forecasting needs SELECT.

### 15.6 Point the engine at it

```powershell
$env:ENGINE_PG_DSN = "postgresql://engine_reader:<password>@localhost:5432/forecast_erp"
```

```bash
engine ingest --client teknic_mumbai_pg
```

No `--file` — the reader is located by the client profile. Passing one is an
error rather than a silent no-op, so nobody can believe they loaded a workbook
while the engine read the database.

The `teknic_mumbai_pg` profile is the same client as `teknic_mumbai` with a
different source block, kept alongside it so the two stores can be compared.
A database source that changes the numbers is a bug.

### 15.7 Credentials

Passwords never go in a client profile. The profile names an environment
variable:

```yaml
source:
  reader: postgres_panel
  database:
    dsn_env: ENGINE_PG_DSN
    relation: erp.v_demand_history
    where: null          # optional, applied server-side
    source_label: null   # defaults to "postgres:<relation>"
```

Driver errors quote the connection string back at you, so everything that
leaves the reader passes through a scrubber first, and failures are raised
with `from None` — a chained traceback would reprint the driver's unscrubbed
message underneath the clean one. There are tests for both.

### 15.8 Re-ingest is a full refresh

`replace_source` is idempotent per source label, and a database relation has no
filename, so the label defaults to `postgres:erp.v_demand_history`. Re-ingesting
deletes everything previously loaded under that label and replaces it.

**This means narrowing `where` between runs drops history rather than leaving
it in place.** Incremental extraction — pull only what changed, upsert the rest
— is a separate piece of work and is not built.

### 15.9 Testing

Unit tests for the reader run with no database and are always on. The
round-trip test against a live server is deselected by default:

```bash
pytest -m integration
```

It skips unless `ENGINE_PG_DSN` is set. It checks the view's grain and types
against the canonical contract, reconciles the totals against the workbook, and
asserts that the engine's role cannot insert.

### 15.10 What is not built

- **Exogenous features from the database.** `engine exogenous` still reads a
  file. Teknic has no context features, so there was nothing to test a database
  path against, and untested code is worse than absent code.
- **Write-back.** Forecasts stay in DuckDB with CSV export. Publishing them to
  PostgreSQL needs a second role with INSERT rights and a schema the ERP agrees
  to read — a separate piece of work.
- **Incremental extraction.** See 15.8.

---

## 16. Publishing forecasts back

Reading from the ERP is half an integration. A forecast that lives in a DuckDB
file or a CSV is not something an ERP screen can show a planner. `engine publish`
writes a completed forecast run into a schema the ERP reads.

The engine's own store stays the source of truth. Publishing copies one finished
run out; it never reads back.

### 16.1 Two roles, not one

| Role | Sees | Can |
| --- | --- | --- |
| `engine_reader` | `erp.v_demand_history` | SELECT only |
| `engine_writer` | `forecast_output.*` | SELECT, INSERT, DELETE |

Neither can do the other's job. A mistake in the publish path has no route to
the ERP's sales history, and there is an integration test asserting exactly
that. They use different DSNs, in different environment variables, for the same
reason.

### 16.2 Create the output schema and the writer

```bash
psql -U postgres -d forecast_erp -f scripts/sql/003_forecast_output.sql
```

```powershell
$env:ENGINE_WRITER_PASSWORD = "<choose one>"
```

```powershell
psql -U postgres -d forecast_erp -v writer_password="$env:ENGINE_WRITER_PASSWORD" -f scripts/sql/004_engine_writer.sql
```

### 16.3 Publish

```powershell
$env:ENGINE_PG_WRITE_DSN = "postgresql://engine_writer:<password>@localhost:5432/forecast_erp"
```

```bash
engine publish --client teknic_mumbai_pg
```

Without `--run-id` it publishes the latest forecast run. Add the output block to
the client profile first:

```yaml
output:
  postgres:
    dsn_env: ENGINE_PG_WRITE_DSN
    target_schema: forecast_output
```

A client with no `output` block has nowhere to publish to and says so. That is
deliberate — a publish target should never be acquired by accident.

### 16.4 What the ERP reads

| Object | Contents |
| --- | --- |
| `v_current_forecast` | **The live forecast.** What an ERP should read. |
| `v_current_forecast_quantile` | Its quantiles, long form |
| `demand_forecast` | Every published run, for history and accuracy tracking |
| `forecast_run` | One row per publish: cutoff, horizon, row count, timestamp |

"Current" is the most recently *published* run, not the highest run id —
republishing an older run is a deliberate act and takes effect.

Quantiles are stored long (one row per level) rather than as `q10`/`q50`/`q90`
columns, because the set of levels is client configuration. A client that wants
a 95% service level should not need a schema migration. An ERP wanting wide
columns can pivot in its own view.

### 16.5 Republishing is safe

Publishing the same run twice replaces it rather than duplicating it, and the
whole run — summary, forecasts and quantiles — goes in one transaction. A
partial publish would leave the ERP reading a forecast covering some items and
not others, which is worse than no forecast at all.

### 16.6 A note on local authentication

A default PostgreSQL install on Windows ships with `trust` authentication for
connections from `127.0.0.1`:

```
host    all    all    127.0.0.1/32    trust
```

**Any local connection is accepted regardless of password.** Role *privileges*
are still enforced — `engine_writer` genuinely cannot read the `erp` schema —
but the passwords are not checked from localhost, so the separation is by
privilege rather than by credential.

Before this runs anywhere but a development machine, change both `trust` lines
in `pg_hba.conf` to `scram-sha-256` and reload the server. The tests in this
repo point at a closed port rather than localhost for the same reason: a unit
test with a plausible DSN will otherwise reach a live database.

### 16.7 What is still not built

- **Nothing reads the forecast back.** No ERP screen, no reorder suggestion, no
  accuracy tracking against what actually sold. The rows are there; consuming
  them is the ERP side of the work.
- **No decision layer.** The published numbers are quantities and quantiles, not
  reorder points or safety stock. See [section 12](#12-scope-and-limits).

---

## 17. The decision layer

Accuracy metrics treat over- and under-forecasting as equally bad. Inventory does
not. A stockout and a month of holding are different amounts of pain, and their
ratio is what decides how much stock to keep — so a model can be less accurate
and still make better decisions.

`engine decide` scores each model on what its stocking decisions cost, and turns
the winning forecast into stock levels.

### 17.1 The cost model

```yaml
cost_model:
  holding_cost_per_unit: 1.0
  shortage_cost_per_unit: 9.0
  lead_time_periods: 1
  review_period_periods: 1
```

The **ratio** is what matters, not the absolute numbers. 9:1 says a stockout
hurts nine times as much as carrying a unit for a month, and the newsvendor
critical ratio turns that into a service level:

```
service level = Cu / (Cu + Co) = 9 / (9 + 1) = 90%
```

That is the quantile at which one more unit costs exactly as much as one fewer.
Choosing a target service level by hand is an implicit claim about this ratio,
made without stating it; this way round, the claim is written down.

**Units are not currency.** Teknic's source carries no prices, so the engine
reports cost in whatever units the ratio is expressed in. It does not print a ₹
sign it has not earned. Supplying real per-unit costs makes the same numbers
monetary without any code change.

**You can only stock to a quantile you forecast.** A 9:1 ratio needs `0.9` in
`probabilistic.levels`. A 2:1 ratio implies 66.7%, which is not forecast, and the
config is rejected with that spelled out — interpolating between the 50th and
90th percentile would invent a distribution shape rather than report one a model
produced.

### 17.2 Running it

```bash
engine backtest --client teknic_mumbai_pg
```

```bash
engine decide --client teknic_mumbai_pg
```

The backtest must be re-run first if it predates this layer: it always computed
quantiles at each origin but only persisted the point forecast, and a decision is
scored against the distribution available when it was made. The command says so
rather than failing obscurely.

### 17.3 How decision cost is measured

A per-period newsvendor on held-out origins. At each origin the model's quantile
at the critical ratio is the stock level, the actual is what showed up, holding
cost falls on the excess and shortage cost on the shortfall.

**Inventory carried between periods is deliberately not simulated.** Doing it
needs an opening stock, an order calendar and receipt lags — none of which exist
in the data, all of which would be invented. Because the carryover a model would
face is the same whichever model produced the forecast, the comparison between
models stays fair, which is the number worth having.

One consequence is useful: under this reduction a perfect forecast costs exactly
nothing, so **a model's decision cost is its regret** against perfect foresight.

**Decision Value Add** is the share of the baseline's decision cost avoided — the
money-shaped counterpart of FVA. It is computed within each segment, not globally,
so Head items are compared against the baseline's Head cost.

### 17.4 Measured on the Teknic panel

| Model | Decision cost | Held | Short | Fill rate | DVA |
| --- | --- | --- | --- | --- | --- |
| `lightgbm_global` | 1,867,686 | 1,507,051 | 40,071 | 98.7% | **+21.2%** |
| `seasonal_naive` | 2,370,937 | 1,574,977 | 88,440 | 97.0% | baseline |
| `chronos2` | 2,421,259 | 1,820,284 | 66,775 | 97.8% | −2.1% |
| `naive` | 3,197,545 | 2,747,095 | 50,050 | 98.3% | −34.9% |

Two things worth reading off this. LightGBM avoids **21% of the baseline's
decision cost**, consistently across Head, Mid and Tail. And **Chronos-2 costs
more than the baseline** despite winning on accuracy for many individual series —
its intervals are wide, so it over-stocks. Accuracy and decision quality are not
the same ranking.

### 17.5 Calibration is the check that makes the rest trustworthy

A model stocking to its own 90th percentile should serve about 90% of demand. On
the Teknic panel every model serves 97–99%, so every model's intervals are too
wide and every model over-stocks. `engine decide` prints this as a warning.

This is a real finding, not a rounding artefact: it means the p90s are too high,
the portfolio holds more inventory than the configured service level asks for,
and no accuracy metric reports it. Narrowing them is a model-layer fix that now
has a number attached to it.

### 17.6 Stock levels

| Column | Meaning |
| --- | --- |
| `expected_demand` | Sum of point forecasts over the protection window |
| `target_stock` | Order up to this at each review — the quantile over lead time + review period |
| `safety_stock` | `target_stock` − `expected_demand`, floored at zero |
| `reorder_point` | Trigger level for continuous review — the quantile over lead time alone |

Both `target_stock` and `reorder_point` are reported because which policy a
client runs is the client's business. Under periodic review only the first is
used.

**One known conservatism:** `target_stock` sums per-period quantiles, and for a
service level above 50% the sum of quantiles exceeds the quantile of the sum. It
therefore errs toward holding too much. Correcting it needs a joint distribution
across horizons, which the models do not produce.

### 17.7 What this exposed

**40 of 297 series get no stock policy at all.** Arbitration picks a model on
accuracy, and `auto_ets` and `croston` produce no quantiles — so a series whose
champion is one of them has no distribution to stock against and never reaches a
reorder screen. The engine reports the count and names the models.

This is the decision layer earning its place before it has changed anything:
it found a constraint the arbitration does not know about. Fixing it means either
teaching those models to emit quantiles, or making arbitration prefer a model
that can support a decision.

### 17.8 What is not built

- **Arbitration still selects on accuracy**, not on decision cost. Measuring
  first was deliberate — now we know the two rankings disagree, and by how much.
- **Decisions are not published.** `engine publish` writes forecasts, not stock
  levels. Nothing in the ERP reads a reorder point yet.
- **No carryover simulation.** See 17.3.
- **One cost ratio for every item.** Real portfolios have cheap items and
  critical ones; per-item costs would need a cost table as a new input.
