# Setup

Getting the forecast engine running from nothing, on a machine that has never
seen it. Steps 1–10 need no database and take about fifteen minutes, most of
that install time.

Commands are written for **Windows PowerShell**, which is what the project is
developed on. On macOS or Linux, swap `.venv\Scripts\` for `.venv/bin/` and
backslashes for forward slashes; nothing else changes.

Every command calls `.venv\Scripts\python.exe` directly rather than relying on
an activated environment. That is deliberate — it always works, and it avoids
the PowerShell execution-policy error that activation sometimes triggers on a
locked-down machine. If you would rather activate, see
[§11](#11-optional-activating-the-environment).

For what the engine does and why, see [README.md](README.md). This file only
covers getting it to run.

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the repository](#2-clone-the-repository)
3. [Create a virtual environment](#3-create-a-virtual-environment)
4. [Install](#4-install)
5. [Verify the install](#5-verify-the-install)
6. [Load the demand history](#6-load-the-demand-history)
7. [Profile the series](#7-profile-the-series)
8. [Backtest and arbitrate](#8-backtest-and-arbitrate)
9. [Forecast](#9-forecast)
10. [Get the stock levels](#10-get-the-stock-levels)
11. [Optional: activating the environment](#11-optional-activating-the-environment)
12. [Optional: reading from PostgreSQL](#12-optional-reading-from-postgresql)
13. [When something goes wrong](#13-when-something-goes-wrong)

---

## 1. Prerequisites

| Requirement | Notes |
| --- | --- |
| **Python 3.11 or newer** | Developed and tested on 3.13.2 |
| **Git** | To clone the repository |
| **~4 GB free disk** | Virtual environment, mostly PyTorch, plus ~500 MB of model weights |
| **Internet, first run only** | Chronos-2 weights download from Hugging Face once, then cache |
| **CPU only** | No GPU needed. Chronos-2 forecasts all 297 sample series in about 1.6s |
| **Access to the repository** | It is **private** — a clone fails with a 404 unless you have been added as a collaborator |

Check your Python version first:

```bash
python --version
```

If it reports anything below 3.11, install a newer Python before continuing.

---

## 2. Clone the repository

```bash
git clone https://github.com/plutoTaco/forecast-engine.git
```

```bash
cd forecast-engine
```

The sample demand workbook is committed, so there is nothing else to download.
The M5 Walmart dataset is deliberately **not** in the repository — it is large
and its competition licence does not permit redistribution. You do not need it
for any step below.

---

## 3. Create a virtual environment

```bash
python -m venv .venv
```

This creates `.venv/` inside the project. It is gitignored.

---

## 4. Install

```bash
.venv\Scripts\python.exe -m pip install -e ".[dev,foundation]"
```

Expect several minutes — PyTorch is a large download. Installing in editable
mode (`-e`) means source changes take effect without reinstalling.

Three optional extras exist:

| Extra | Adds | When to skip it |
| --- | --- | --- |
| `dev` | `pytest`, `ruff` | Only for a runtime-only deployment |
| `foundation` | Chronos-2 and PyTorch (~3 GB) | See below |
| `postgres` | ADBC driver and PyArrow | Unless you want [§12](#12-optional-reading-from-postgresql) |

**To avoid the 3 GB download**, install `-e ".[dev]"` instead, then remove
`chronos2` from `models.enabled` in `config/clients/teknic_mumbai.yaml`. The
other five models run without it, and the engine gives a clear error if the
model is enabled but not installed.

---

## 5. Verify the install

```bash
.venv\Scripts\python.exe -m pytest -q
```

Expected: **151 passed, 13 deselected** in roughly 15 seconds.

The deselected tests are correct, not a problem. They need something a clean
machine does not have:

| Marker | Needs | Run with |
| --- | --- | --- |
| `slow` | Chronos-2 weights already cached | `pytest -m slow` |
| `integration` | A live PostgreSQL, with `ENGINE_PG_DSN` and `ENGINE_PG_WRITE_DSN` set | `pytest -m integration` |

A **skipped** test is different from a deselected one and means
`data/raw/Demand_History_3_Years.xlsx` is missing from the clone.

Lint and formatting, if you are going to change anything:

```bash
.venv\Scripts\python.exe -m ruff check .
```

---

## 6. Load the demand history

```bash
.venv\Scripts\python.exe -m engine.cli ingest --client teknic_mumbai --file data\raw\Demand_History_3_Years.xlsx
```

Expected:

```
reconciled 10,692 rows (297 series, 2023-09-01 to 2026-08-01); quantity sum 37,372,528
```

That reconciliation line is the point of the step. Row count and quantity sum
are compared before and after mapping into the engine's canonical model, and a
mismatch fails the load rather than producing a quietly wrong panel.

Re-running is safe. Loads are idempotent per source: the same file replaces its
own rows rather than duplicating them.

---

## 7. Profile the series

```bash
.venv\Scripts\python.exe -m engine.cli features --client teknic_mumbai
```

Classifies every series by demand pattern — smooth, seasonal, intermittent,
volatile and so on — and splits the portfolio into Head, Mid and Tail by
cumulative volume share.

Because the sample data carries ground-truth pattern labels, this also prints a
confusion matrix. That is validation of the classifier against known answers;
those labels are never used as a model feature.

---

## 8. Backtest and arbitrate

```bash
.venv\Scripts\python.exe -m engine.cli backtest --client teknic_mumbai
```

**The slow step — expect a few minutes.** Every model is run at every origin.
Chronos-2 dominates the time, and on the first run it downloads its weights.

It prints Forecast Value Add by demand pattern and by volume segment, the
held-out portfolio result, and which model won how many series. Everything is
stored, not just printed.

Expected headline: **portfolio +45.7% FVA** on the held-out origin.

Three flags override the client profile for one run without editing the file:

```bash
.venv\Scripts\python.exe -m engine.cli backtest --client teknic_mumbai --horizon 6 --origins 2
```

`--max-history` additionally truncates each series, which is how the cold-start
experiment works.

---

## 9. Forecast

```bash
.venv\Scripts\python.exe -m engine.cli forecast --client teknic_mumbai --horizon 3 --export forecasts.csv
```

Projects forward from the end of history using the arbitrated model selection,
and writes point forecasts plus calibrated p10/p50/p90 quantiles.

`--export` additionally writes a spreadsheet-ready CSV with customer, item code
and description joined on, because a bare series id means nothing to a planner.
Use `.parquet` instead of `.csv` if you want the typed version.

`--model lightgbm_global` forces one model instead of per-series arbitration,
which is useful for comparing a single model's forward projection.

---

## 10. Get the stock levels

```bash
.venv\Scripts\python.exe -m engine.cli decide --client teknic_mumbai
```

Expected: `policy covers all 297 series`

**This is the output worth showing people.** Everything before it is a forecast.
This is the only view that says what to actually do: target stock, safety stock
and a reorder point for every item, plus what each model's stocking decisions
would have cost on held-out data.

### Looking at the results afterwards

```bash
.venv\Scripts\python.exe -m engine.cli ui --client teknic_mumbai
```

Opens the client's DuckDB database in a read-only browser SQL viewer on port
4213. Every table written by every step above is there.

```bash
.venv\Scripts\python.exe -m engine.cli report --client teknic_mumbai
```

Re-prints a stored scoreboard without recomputing anything.

```bash
.venv\Scripts\python.exe -m engine.cli clients
```

Lists the configured client profiles.

---

## 11. Optional: activating the environment

Activating puts `python`, `pytest` and `engine` on your PATH, so commands get
shorter — `engine ingest --client teknic_mumbai` instead of the full path.

```powershell
.venv\Scripts\activate.ps1     # Windows PowerShell
.venv\Scripts\activate.bat     # Windows cmd.exe
source .venv/bin/activate      # macOS / Linux
```

If PowerShell refuses with an execution-policy error, allow scripts for the
current session only:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Or simply skip activation — the explicit `.venv\Scripts\python.exe` form used
throughout this file works identically and never hits this.

---

## 12. Optional: reading from PostgreSQL

Steps 1–10 are the entire engine running on file input. Replacing the workbook
with a database is how a real ERP integration works, but it changes nothing
about what the forecasts say — it is worth doing to demonstrate the integration,
not to improve a result.

It needs the `postgres` extra, a running PostgreSQL 14+, and the five scripts in
`scripts/sql/`. The full walkthrough, including credential handling and the
read/write role separation, is in **README sections 15 and 16** — more detail
than belongs in a quick start.

The short version:

```bash
.venv\Scripts\python.exe -m pip install -e ".[postgres]"
```

Then follow [README §15](README.md#15-reading-from-postgresql) to create the
database, the ERP-shaped schema and the read-only role, and
[README §16](README.md#16-publishing-forecasts-back) to publish forecasts and
stock levels back.

---

## 13. When something goes wrong

| Message | What it means |
| --- | --- |
| `404` on clone | The repository is private; you have not been added to it |
| `No data ingested yet. Run engine ingest first.` | Steps must run in order. Each command names the step it needs |
| `Backtest run N stored no quantiles at level 0.9` | That run predates the decision layer. Re-run `backtest` |
| `Minimum samples for computing prediction intervals are N` | The horizon is too long for the history available — see [README §18](README.md#18-forecast-horizons) |
| `Earliest origin would train on N periods, below the configured minimum` | Lower `evaluation.origins`, `evaluation.horizon`, or `min_train_periods` |
| A config key is rejected at load | Client profiles forbid unknown keys, so a typo fails immediately rather than silently taking a default nobody intended. The error names the key |
| `Port 4213 is already in use` | A viewer is already running. Pass `--port 4214` |
| `Environment variable ENGINE_PG_DSN is not set` | Only relevant to §12. Passwords live in environment variables, never in a profile |
| PowerShell execution-policy error | Only happens when activating the environment. See [§11](#11-optional-activating-the-environment) |
| `ModuleNotFoundError: chronos` | `foundation` extra not installed, but `chronos2` is still in `models.enabled` |

If a step fails in a way that is not listed here, the stored run is still in the
client's DuckDB file — `engine ui` will show you how far it got.
