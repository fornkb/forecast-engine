# Reasoning layer: external factors and a ± band around the forecast

The forecasting engine predicts demand from a company's own sales history. It cannot know that
a strong El Niño is building, that copper just hit a record, or that Diwali falls in a
different month this year. The reasoning layer adds that context. It finds the external factors
that matter to a company, reads what changed this month, and shows a **what-if band around
the saved forecast with plain-language reasoning and sources**. The stored forecast is never
changed.

Everything company-specific is configuration: a short `reasoning:` section in the client
profile and a company profile YAML. The same code serves a water bottler and an electrical
components maker.

## How it works

```
ONCE PER CLIENT: onboarding                                        1–2 AI calls
  ERP product list + sales summary + admin's description
     → AI drafts a company profile: product families, inputs, regions, and the DRIVERS to watch
     → the admin answers the AI's questions in one batch → config/profiles/<client>.yaml

EVERY MONTH, after `engine forecast`                               typically 0–2 AI calls
  1. forecast.py   pin the saved forecast; roll SKUs up into families           no AI
  2. signals.py    news, seasonal weather, ENSO, festival dates per driver        no AI
  3.               change filter: drivers with nothing new keep last month's finding
  4. triage.py     screen articles with a small model, only if there are many    0–1 call
  5. reasoner.py   one call reasons over all changed drivers × families          1 call
  6. adjust.py     Python checks citations, quotes and caps, then computes bands  no AI
  7. report.py     HTML for managers, Markdown/JSON, planner overrides            no AI
```

**Drivers** are external factors such as `el_nino`, `pune_weather`, `copper` or `diwali`. Each
one uses a **template** from `src/engine/reason/catalog/`:

| Template | Watches |
| --- | --- |
| `enso` | NOAA ENSO status and regional news |
| `seasonal_weather` | monthly temperature and rain vs the 10-year normal |
| `monsoon`, `heatwave` | the same weather data, plus targeted news |
| `commodity_news` | prices and supply of an input |
| `festival` | holiday dates, plus the uplift measured in the company's own history |
| `policy_news`, `geopolitics_news`, `market_news` | regulation, conflict and route disruption, the company's markets |

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
- **The band is advisory.** Stock policy and ERP publishing keep using the original forecast.
- **The value is not proven yet.** Whether bands beat the plain forecast needs months of
  outcomes. Keep the run documents and compare once actuals arrive.
