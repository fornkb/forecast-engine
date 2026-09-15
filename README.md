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

---

## 1. Prerequisites

| Requirement | Notes |
| --- | --- |
| Python 3.11 or newer | Developed and tested on 3.13.2 |
| Git | To clone the repository |
| ~4 GB free disk | Virtual environment (~3 GB, mostly PyTorch) plus ~500 MB of model weights |
| CPU only | No GPU required — Chronos-2 forecasts all 297 series in ~1.6s on CPU |
| Network, first run only | Chronos-2 weights download from Hugging Face once, then cache |

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
| `engine ingest` | Load a source file into the canonical store | `--client`, `--file` |
| `engine features` | Profile series and classify demand patterns | `--client` |
| `engine backtest` | Rolling-origin evaluation, persists scoreboard | `--client`, `--horizon`, `--origins`, `--max-history` |
| `engine coldstart` | Accuracy vs. length of training history | `--client`, `--depths` |
| `engine report` | Re-render a stored run | `--client`, `--run-id` |
| `engine forecast` | Forecast forward from end of history | `--client`, `--horizon`, `--model`, `--export` |
| `engine ui` | Browse the client's database in DuckDB's viewer, read-only | `--client`, `--port`, `--no-browser` |

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
- **No exogenous signals, no uncertainty intervals, no decisions.** Forecasts are
  point estimates in units. Weather, festival calendars, conformal intervals, and
  ordering recommendations arrive in later phases by design.
- **Intermittent series remain hard.** A median series WAPE near 1.0 on that
  slice reflects the genuine difficulty of mostly-zero demand.
- **No exogenous signals, no uncertainty intervals, no decisions yet.** Chronos-2
  produces quantiles and accepts covariates natively, but neither is used yet —
  those are Phase 2.

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
