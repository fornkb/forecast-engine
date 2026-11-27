"""Reasoning layer: external factors, AI reasoning and a ± band around saved forecasts.

Flow for one monthly run (see docs/reasoning.md):
  forecast.py   pin a saved forecast and roll it up into product families
  signals.py    collect news / weather / ENSO / calendar signals per driver, detect changes
  triage.py     keep only relevant articles (0 or 1 cheap model call)
  reasoner.py   one model call that reasons over all changed drivers
  adjust.py     Python checks the answer and computes per-SKU bands
  run.py        orchestration, budgets, caching; report.py renders the result
"""

# Client config embeds ReasoningConfig, so it must finish loading before any reason
# module: importing engine.reason.* first would otherwise hit a circular import.
import engine.config  # noqa: E402, F401
