# Reasoning layer: external factors and a ± band around the forecast

The forecasting engine predicts demand from a company's own sales history. It cannot see
anything happening outside the company. The reasoning layer answers one question each month:

> **What changed that could affect this company's forecast, and what evidence do we have?**

It watches the outside factors that matter to the company, runs a broad news search to catch
what no watched factor covers, and shows a **what-if band around the saved forecast with
plain-language reasoning and sources**. The stored forecast is never changed.

Every review ends in one of three plain answers:
- **Changes found**, each with its evidence and effect.
- **Nothing important changed**, with the sources that were checked. This is a normal result.
- **Incomplete**: a source failed (for example, news was rate-limited). The review names it and
  never claims "nothing changed" for what that source covers.

Everything company-specific is configuration: a short `reasoning:` section in the client
profile and a company profile YAML. The same code serves a water bottler and an electrical
components maker.

## How it works

```
ONCE PER CLIENT: onboarding                                        1–2 AI calls
  ERP product list + sales summary + admin's description
     → AI drafts a company profile: product families, inputs, regions, the DRIVERS to watch
       and 2–4 broad SCAN topics
     → the admin answers the AI's questions in one batch → config/profiles/<client>.yaml

EVERY MONTH, after `engine forecast`                               typically 0–2 AI calls
  1. forecast.py   pin the saved forecast; roll SKUs up into families           no AI
  2. signals.py    each driver's sources, plus the open news scan                 no AI
  3.               change filter: drivers with nothing new keep last month's finding
  4. triage.py     screen articles with a small model, only if there are many    0–1 call
  5. reasoner.py   one call: changed drivers × families, plus OTHER NEWS          1 call
  6. adjust.py     Python checks citations, quotes and caps, then computes bands  no AI
  7. report.py     HTML for managers, Markdown/JSON, planner overrides            no AI
```

**Drivers** are the outside factors a company watches, such as `import_duty`, `copper` or
`winter_weather`. The onboarding AI chooses them from the company's data and description. Its
prompt names no industries or topics; it asks what a model trained only on past sales cannot see,
including indirect chains. Each driver uses a **data source** (template) from
`src/engine/reason/catalog/`. The catalogue lists tools, not topics:

| Template | Evidence it gives |
| --- | --- |
| `topic_news` | news on any topic |
| `commodity_news`, `policy_news`, `market_news`, `geopolitics_news` | news on an input, a rule, a market or a route |
| `seasonal_weather` | monthly temperature and rain vs the 10-year normal for a region |
| `enso` | NOAA's current El Niño / La Niña status, plus news on its effects |
| `festival` | holiday dates, plus the effect measured in the company's own history |

`monsoon` and `heatwave` still work in existing profiles but are no longer offered to onboarding
(`listed: false`).

**The open news scan.** Each review also runs 2–4 broad searches from the profile's `scan` topics
(or one per family when none are set). The results go to the model as OTHER NEWS. If an item
reports a real development that no driver covers, the model may raise it as a **new factor**.
Most months nothing qualifies, and that is the expected answer. New factors are held to stricter
checks (below). The planner can accept one and click **Add to watch list**, which adds it as a
`topic_news` driver so later reviews track it. Turn the scan off with `sources.scan: false`.

**Connectors** fetch the data:

| Connector | Source | Key needed |
| --- | --- | --- |
| `news` | GDELT (headlines only) or Brave (with snippets) | GDELT: none; Brave: paid key |
| `weather` | Open-Meteo, ECMWF SEAS5 ensemble mean plus archive | Free for non-commercial use; companies need the commercial key |
| `enso` | NOAA CPC discussion page | None |
| `calendar` | `holidays` package or dates given in the profile | None, works offline |

Queries contain only public words from the profile, such as material names, regions and topics.
No customer names or ERP numbers leave the system, and article URLs are citations that are
never fetched.

## How numbers stay honest

The model proposes a percentage low/high per family and month and labels its evidence.
Python enforces each label:

| Evidence tier | Meaning | Python's check |
| --- | --- | --- |
| measured | A number from the company's own sales history (e.g. "flavoured water ran +11.9% in the 2024 spring after the last El Niño") | Must cite that measure for that family. The size is capped at 1.5× the measured value. |
| sourced | A sentence quoted from a stored article | The quote must really appear in the cited source, otherwise it becomes judgment. |
| judgment | The model's own estimate | Capped at `max_judgment_pct` (default ±10%) and labelled as judgment. |
| none / supply_risk / cost | A direction or an alert only | No quantity change. The oil-supply example lands here: cost and supply risk, not demand, unless prices pass through. |

**New factors** from the open scan are stricter, because no watched driver vouches for them:
- The development must be stated in an article the model cites, and its `quote` must appear in
  that article. Otherwise the new factor is dropped and the drop is listed under gaps. The model
  may explain the *mechanism* from its own knowledge; the *fact* has to come from a source.
- They cannot use the measured tier.
- A judgment size is capped at `max_new_factor_pct` (default ±5%).

More rules:
- **Double counting:** the model sees "same month last year" for each family and must mark
  effects that normal seasonality already explains. The report warns when 3 or more drivers
  stack on one family-month.
- **Bands:** baseline × (1 + Σ low%) to baseline × (1 + Σ high%) per SKU-month, never below
  zero.
- **Kept separate from model uncertainty:** the band is a what-if range, not a probability. The
  report shows it next to the model's own p10–p90.
- **Bad answers are corrected, never trusted:** unknown citations and months outside the
  forecast are removed, every correction is listed as a flag on the finding, and the model
  never computes a number itself.
- **Measured effects:** `analog_periods` on any driver (e.g. the months of the last El Niño
  spring) makes Python measure how far those months ran above the same months in other years,
  with growth removed. The `festival` template does this automatically for festival months.

## Cost: why a month costs cents

| Lever | Effect |
| --- | --- |
| Reason per product family × driver, not per SKU × month | 297 Teknic SKUs become 5 families. |
| Change filter | An unchanged driver keeps its finding at zero cost. A month with no news makes 0 calls. |
| No tool loops | Python collects everything first, so one request carries the whole packet. |
| Free article filters first | Dedupe, newest first, a cap per driver. The triage call runs only above `triage_threshold`. |
| Compact packet | Short ids (A1, S2, M3), trimmed snippets, family totals. |
| Fixed instructions first | Provider prompt caching applies (OpenAI automatic, Anthropic `cache_control`, Gemini implicit). |
| Same inputs return the saved run | Re-running costs nothing. |
| Hard caps | `max_calls_per_run` (default 3) and `max_input_tokens` (default 20k). `--dry-run` estimates before spending. |

Measured on the AquaPure demo: 6 drivers and 3 families fit in **1 call of about 1,700 input
tokens**. Every call is logged with tokens and cost (`engine reason-cost`). Cost is computed
from the rates in the client profile; without rates it shows as unknown.

## Model providers

Set per client in `reasoning.llm`. Each provider uses its official SDK and native JSON-schema
output.

```yaml
reasoning:
  llm:
    provider: openai        # openai | anthropic | gemini
    model: gpt-6.2-sol      # name as the provider spells it
    triage_model: gpt-6-luna
    effort: low             # minimal | low | medium | high | xhigh | max (mapped per provider)
    api_key_env: OPENAI_API_KEY   # default per provider: OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY
    rates: {gpt-6.2-sol: {input_per_m: 2.0, output_per_m: 10.0}}
```

Install only the SDK a client uses: `pip install -e ".[openai]"`, `".[anthropic]"` or
`".[gemini]"`. `--llm record` saves every answer under `recordings_dir`. `--llm replay` serves
them offline at zero cost, which is how a demo is shown without a network or a key.

## Customizing for a company

| Need | Change |
| --- | --- |
| New company | `engine reason-onboard --client C --about about.txt`. Answer the printed questions in a text file, then run `--answers answers.txt --confirm`. Or write `config/profiles/<client>.yaml` by hand, following `config/profiles/demo_aquapure.yaml`. |
| New product family | Add it under `families:` with a match rule (`item_group`, `item_prefix` or `series`). |
| New driver of a known kind | Add it under `drivers:` with a template and params. |
| New kind of driver | Add one YAML file to `config/drivers/` (company-specific) or `src/engine/reason/catalog/`. |
| New data source | One module in `src/engine/reason/connectors/` plus one line in `CONNECTORS`. |
| Different model or provider | `reasoning.llm` in the client profile. |
| Stricter or looser limits | `reasoning.budget`, e.g. `max_judgment_pct`. |

## Runbook

```powershell
# AquaPure demo (synthetic packaged water, El Niño story)
python scripts/make_demo_water.py
engine ingest   --client demo_aquapure --file data/demo/aquapure_sales.csv
engine features --client demo_aquapure
engine backtest --client demo_aquapure
engine forecast --client demo_aquapure --horizon 6

$env:OPENAI_API_KEY = "<your key>"
engine reason --client demo_aquapure --scenario data/demo/scenarios/aquapure_el_nino.json --dry-run
engine reason --client demo_aquapure --scenario data/demo/scenarios/aquapure_el_nino.json --llm record --html reports/aquapure.html

# Later, offline and free: the same answers from the recording
engine reason --client demo_aquapure --scenario data/demo/scenarios/aquapure_el_nino.json --llm replay --full --html reports/aquapure.html

# Live sources instead of the scenario (GDELT, Open-Meteo, NOAA)
engine reason --client demo_aquapure --html reports/aquapure_live.html

# Planner decisions and cost (no AI calls)
engine reason-adjust --client demo_aquapure --reason-id R20261126-xxxxxx --finding F2 --action edit --low 2 --high 6 --by "planner name"
engine reason-report --client demo_aquapure --html reports/aquapure.html
engine reason-cost   --client demo_aquapure
```

Scenario packs (`data/demo/scenarios/*.json`) replace only the network sources, so a demo is
repeatable and does not depend on today's news. The festival calendar and measured effects
still run. Scenario articles are marked `[Scenario]` and the report carries a banner saying the
signals are illustrative.

## Web app: Forecast Engine v2 console

`engine web` starts a local console for the ERP provider's team on `http://127.0.0.1:8000`
and opens the browser. It covers the forecast and the external-factor reviews for every client.
It calls the same functions as the CLI (`engine forecast`, `engine backtest`,
`engine reason*`), so both always agree, and the CLI keeps working unchanged.

```powershell
pip install -e ".[web,openai]"
$env:OPENAI_API_KEY = "<your key>"     # set before starting; Settings shows whether it is set
engine web                             # --port 8080, --no-open
```

Without a terminal: double-click **`web.bat`** in the project folder (or run `python web.py`).

**Preview mode, no API key needed:** double-click **`web_preview.bat`** (or `engine web --preview`).
A script writes sample answers in place of the AI, so every page, check, band, decision and
download can be tried. It works on a fresh copy of each store under
`data/processed/_preview/`, labels every page and run as a preview, keeps profiles read-only
and turns off forecast refresh. Use it to explore the interface, never as a demo of AI quality.

| Page | What it shows and does | CLI equivalent |
| --- | --- | --- |
| **Portfolio** | Every client company: KPIs (active clients, need attention, open decisions, reviewed this cycle, AI spend this month), a "Needs attention" list (stale data, unreviewed forecasts, draft profiles, missing keys, open decisions), and a card per client with its headline, forecast, accuracy and open decisions | – |
| **AI usage** | Spend and calls by client and by month, recent calls across clients | `reason-cost` per client |
| Client · **Summary** | One page for a manager: next months vs last year, accuracy vs a simple baseline, the what-if range from the latest review, open decisions, a demand chart, the latest review's headline and findings, a table by product family | – |
| Client · **Forecast** | Total demand chart with the model's p10–p90, month table, accuracy from the backtest (error vs baseline, by volume segment, range coverage, bias), families, largest SKUs, models used. **Refresh forecast**, **Run backtest**, **Export CSV** | `forecast`, `backtest`, `--export` |
| Client · **External factors** | Start a review (live or scenario, live/record/replay, date, reassess all), **Estimate cost** first, list of reviews with headline, range and decision progress | `reason`, `--dry-run` |
| Review | Impact by family, a chart per family with the what-if band, findings with evidence and Python's checks. **Accept, dismiss or edit** (bands recalculate, no AI call). Report, bands CSV for the ERP, Markdown, JSON | `reason-report`, `reason-adjust` |
| Client · **Company profile** | Families with SKU counts, drivers with setup checks, AI questions with an answer box, YAML editor that validates before saving, **draft a profile with AI**, templates, version history | `reason-onboard` |
| Client · **Settings** | Setup checklist, AI model and rates, cost limits, data sources | config YAML |
| Client · **AI cost** | Every call with tokens, cost and outcome | `reason-cost` |

Accuracy is shown as 100% − WAPE from the backtest's held-out months, next to the same
measure for the seasonal-naive baseline. Totals add units across SKUs, so they are only
meaningful where a client's units are comparable; the per-family tables are the safer read.

Reviews, onboarding, forecast refresh and backtests run in the background and the page
follows their progress. One job per client runs at a time, so a double click never pays
twice. The "Signed in as" box in the sidebar is written to the audit trail for decisions and
profile saves.

The app has no login. It listens on 127.0.0.1 by default and refuses form posts that come
from other sites. Do not expose it on a network without putting authentication in front of
it. For an ERP portal, the same functions (`run_review`, `apply_override`,
`run_onboarding`) can sit behind the portal's own authenticated API.

## Storage

These tables are additive and live in the client's existing DuckDB store. Forecast tables are
never written.

| Table | Holds |
| --- | --- |
| `reason_runs` | The full run document: signals, findings, references, bands |
| `reason_bands` | Flat per-SKU low/high for the ERP portal |
| `reason_llm_calls` | Tokens, cost and outcome per call |
| `reason_overrides` | Planner accept, edit and dismiss decisions |
| `reason_signal_cache` | Fetched signals, reused within `cache_hours` |
| `reason_profile_versions` | Every saved company profile, with who saved it |

## Limits

- **Monthly forecasts only.**
- **Weather is regional model output,** not bias-corrected, compared with a reanalysis normal.
  Treat it as indicative.
- **GDELT returns headlines only.** A "sourced" quote therefore has to come from a headline.
  Brave gives snippets.
- **GDELT limits each network to about one request every 5 seconds.** The engine spaces
  requests 6 seconds apart, retries twice, and then pauses news for 10 minutes with a clear
  gap on each driver. Shared connections can be refused even at that pace; Brave avoids it.
- **The band is advisory.** Stock policy and ERP publishing keep using the original forecast.
- **The value is not proven yet.** Whether bands beat the plain forecast needs months of
  outcomes. Keep the run documents and compare once actuals arrive.
