from engine.forecast.arbitration import apply_selection, select_per_series, selection_summary
from engine.forecast.base import (
    ForecastContext,
    Forecaster,
    available_forecasters,
    get_forecaster,
    register_forecaster,
)

# Importing the implementations populates the registry.
from engine.forecast.baselines import NaiveForecaster, SeasonalNaiveForecaster  # noqa: E402
from engine.forecast.classical import (  # noqa: E402
    AutoARIMAForecaster,
    AutoETSForecaster,
    CrostonForecaster,
)
from engine.forecast.lightgbm_model import GlobalLightGBMForecaster  # noqa: E402

__all__ = [
    "AutoARIMAForecaster",
    "AutoETSForecaster",
    "CrostonForecaster",
    "ForecastContext",
    "Forecaster",
    "GlobalLightGBMForecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "apply_selection",
    "available_forecasters",
    "get_forecaster",
    "register_forecaster",
    "select_per_series",
    "selection_summary",
]
