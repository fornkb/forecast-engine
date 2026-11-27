# Mock run: Haveli Harvest Oils (edible oils, North India)

A complete practice run of Forecast Engine v2 on a fictional company. You add an API key, play
the company's planning manager on the web site, and see whether the AI finds anything useful in
**live** news, weather outlooks and NOAA bulletins. No scenario file feeds it the answer.

| File | What it is |
| --- | --- |
| [company-brief.md](company-brief.md) | The company: products, customers, raw materials, seasons, past events, what management needs |
| [manager-answers.md](manager-answers.md) | What the manager types on the site: the company description and answers to the AI's questions |
| [reference_profile.yaml](reference_profile.yaml) | The answer key: the watch list an expert would set up. Compare the AI's draft with it afterwards |
| `data/demo/haveliharvest/` | ERP data: `sales.csv` (36 months, 59 series), `item_master.csv`, `customer_master.csv`, and the real Delhi weather used to build the history |
| `scripts/make_demo_oil.py` | Rebuilds the data (same result every time) |
| `config/clients/demo_haveliharvest.yaml` | The client's engine settings: OpenAI, live sources, no scenario |

## Already done

The data is generated, ingested and backtested, and a 6-month forecast is saved (Nov 2026 – Apr
2027; backtest error 8.1%, 18% better than a seasonal-naive baseline). The company profile is
**deliberately missing**: the AI drafts it in step 3.

To rebuild from scratch:

```powershell
python scripts/make_demo_oil.py
engine ingest   --client demo_haveliharvest --file data/demo/haveliharvest/sales.csv
engine features --client demo_haveliharvest
engine backtest --client demo_haveliharvest
engine forecast --client demo_haveliharvest --horizon 6
```

## The run

**1. Key and model names.** In PowerShell:

```powershell
cd C:\dev\forecast-engine
pip install -e ".[web,openai]"
$env:OPENAI_API_KEY = "<your key>"
```

Check that `gpt-6.2-sol` and `gpt-6-luna` exist on your OpenAI account. If not, change
`reasoning.llm.model`, `triage_model` and `rates` in `config/clients/demo_haveliharvest.yaml`.

**2. Start the site** from the same window, so it sees the key: `python web.py`. Pick **Haveli
Harvest Oils** in the sidebar and type your name under "Signed in as". The summary says the
company profile is missing; that's expected.

**3. Let the AI set up the company** (1 AI call, about $0.01):
1. Open *Company profile* → *Draft a profile with AI*.
2. Paste section 1 of [manager-answers.md](manager-answers.md).
3. Choose **Record** and click *Draft profile*.
4. Read the draft: which product families and outside factors did it pick, which broad scan
   topics did it set, and what does it ask?

**4. Answer as the manager** (1 AI call):
1. Under *Questions from the AI*, paste section 3 of manager-answers.md, or answer each
   question from the answer bank.
2. Choose **Record** and click *Apply answers*.
3. Check that every driver shows **OK**, then click *Mark confirmed*. Section 4 of the answers
   file covers small fixes.

**5. Look before spending** (no AI call). Open *External factors*:
1. Choose *Live sources*, then *Estimate cost*.
2. The table shows what each source found today: news headlines for each driver and for the
   open news scan, plus whatever weather, climate, calendar and measured sources the AI chose.
3. A source that failed says so here. **If most news rows show an error, wait 5–10 minutes and
   estimate again.** News is the main source for this company, so don't run the review
   without it.

**6. Run the review** (usually 1–2 AI calls, a few cents):
1. Back on the form, choose *Live sources* and **Record**.
2. Click *Run review*.

**7. Judge it as the manager.** On the review page, Accept, Edit or Dismiss each finding.
Then look at *Summary* and *Portfolio*. The **Report** button saves a page to share.

**8. Send me the result.** Download **JSON** from the review page (or send screenshots). We use
it to tune the prompt, the drivers and the checks.

## How to judge "useful"

The AI wasn't told what to look for, and neither should you tell it. Afterwards, ask:

| Question | Good sign | Bad sign |
| --- | --- | --- |
| Did it find factors that matter to *this* company? | Edible-oil prices, import policy, oilseed crops, supply routes, festivals, winter | Generic macro news unrelated to cooking oil |
| Did it find a real change that wasn't on its watch list? | A **New factor** card citing a dated article that no driver covers, sized modestly | A new factor built on general knowledge, or one that repeats a watched driver |
| Is every claim sourced? | Each finding cites dated headlines, a weather outlook, NOAA, or a measured past effect | "No valid citation", or quotes downgraded by Python's checks |
| Does it separate cost from demand? | Price and supply news flagged as *cost* or *supply risk*, with demand changed only when it explains why | Every price story turned into a demand change |
| Are the sizes sensible? | Small ranges with stated conditions, sizes tied to measured history (e.g. the −8% after the 2024 duty hike) | Large numbers on AI judgment alone (Python caps these at ±10%) |
| Does it avoid double counting? | Notes that Holi and the wedding season are already in the forecast | Adds a festival lift the forecast already has |
| Would you act on it? | Something that changes a booking, a price list or stock decision | Nothing a planner can use |

**"Nothing important changed" is a valid answer.** It's better than an invented effect. But
check the banner at the top of the review: if it says **Incomplete**, a source failed and that
part was not checked at all. Run the review again later rather than reading it as "nothing".

## Cost

About $0.01 per onboarding call and $0.01–0.05 per review at the configured rates. *AI usage*
on the site shows the real figures. A second review with no new news costs nothing.

## If something goes wrong

| Symptom | Fix |
| --- | --- |
| "model not found" or similar from OpenAI | Change the model names in `config/clients/demo_haveliharvest.yaml`, then restart the site |
| `OPENAI_API_KEY is not set` | Set it in the same PowerShell window, then start `python web.py` from there |
| News rows say "GDELT news is limiting requests from this network" | GDELT, the free news source, allows about one request every 5 seconds, and the engine spaces its requests 6 seconds apart (news for 8 drivers takes about a minute). If your connection is shared or was blocked, GDELT refuses it for a while and the engine pauses news for 10 minutes. Wait and run *Estimate cost* again. If it keeps happening, switch to Brave news: set `news: {provider: brave, lookback_days: 30, max_articles_per_driver: 6}` in `config/clients/demo_haveliharvest.yaml` and `$env:BRAVE_SEARCH_API_KEY = "<key>"` (Brave Search API account), then restart the site |
| You want to redo onboarding | Delete `config/profiles/demo_haveliharvest.yaml` and start again at step 3 |
| You want a fresh review with nothing carried over | Tick *Reassess every driver* on the review form |

## What is real

Real: New Delhi temperatures (Nov 2023 – Oct 2026), festival dates, the import duty hike of
14 Sep 2024 and the cut of 31 May 2025, and everything the live sources return today.

Invented: the company, brands, customers, volumes, prices and how strongly sales react. See the
[company brief](company-brief.md#what-is-real-and-what-is-invented).
