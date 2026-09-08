from engine.forecast.arbitration import (
    GLOBAL_STRATUM,
    apply_selection,
    pick_champions,
    select_per_series,
    selection_summary,
)
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
from engine.forecast.foundation import Chronos2Forecaster  # noqa: E402
from engine.forecast.lightgbm_model import GlobalLightGBMForecaster  # noqa: E402

__all__ = [
    "GLOBAL_STRATUM",
    "AutoARIMAForecaster",
    "AutoETSForecaster",
    "Chronos2Forecaster",
    "CrostonForecaster",
    "ForecastContext",
    "Forecaster",
    "GlobalLightGBMForecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "apply_selection",
    "available_forecasters",
    "get_forecaster",
    "pick_champions",
    "register_forecaster",
    "select_per_series",
    "selection_summary",
]
