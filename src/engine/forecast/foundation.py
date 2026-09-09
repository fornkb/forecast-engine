"""Zero-shot time-series foundation models.

Chronos-2 is pretrained and never fitted on client data — `fit_predict` only
supplies context and reads a forecast back. That is the whole reason it is
expected to help on series with too little history to fit anything: there is no
per-series estimation step to be starved of data.

The model is imported lazily so the rest of the engine runs without torch
installed, and the loaded pipeline is cached because loading costs seconds while
inference costs milliseconds.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from engine.config.schema import ChronosParams, quantile_column
from engine.forecast.base import ForecastContext, Forecaster, register_forecaster

_PIPELINE_CACHE: dict[tuple[str, str], Any] = {}


def load_pipeline(params: ChronosParams) -> Any:
    """Load and cache a Chronos pipeline, keyed by model and device."""
    key = (params.model_id, params.device)
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]

    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "chronos2 is enabled for this client but chronos-forecasting is not "
            'installed. Run: pip install "chronos-forecasting>=2.0" — or remove '
            "chronos2 from models.enabled in the client profile."
        ) from exc

    pipeline = Chronos2Pipeline.from_pretrained(params.model_id, device_map=params.device)
    _PIPELINE_CACHE[key] = pipeline
    return pipeline


@register_forecaster
class Chronos2Forecaster(Forecaster):
    """Amazon Chronos-2 (120M, Apache-2.0), run zero-shot.

    Quantiles are produced directly by the model rather than sampled, so repeated
    runs on the same history return identical forecasts.
    """

    name = "chronos2"

    def fit_predict(
        self, history: pl.DataFrame, horizon: int, ctx: ForecastContext
    ) -> pl.DataFrame:
        params = ctx.client.models.chronos
        levels = ctx.quantile_levels
        pipeline = load_pipeline(params)

        context = (
            history.sort("series_id", "period")
            .select(
                pl.col("series_id").alias("id"),
                pl.col("period").alias("timestamp"),
                pl.col("quantity").alias("target"),
            )
            .to_pandas()
        )

        # Ask for the median alongside the configured levels so the point
        # forecast and the interval come from one pass over the model.
        requested = sorted({0.5, *levels})

        predictions = pipeline.predict_df(
            context,
            prediction_length=horizon,
            quantile_levels=requested,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

        # The median minimises absolute error, which is the metric the scoreboard
        # leads on; the mean would be the right choice for a squared-error metric.
        median_column = _median_column(requested)
        frame = pl.from_pandas(predictions)

        return frame.select(
            pl.col("id").cast(pl.Utf8).alias("series_id"),
            pl.col("timestamp").cast(pl.Date).alias("period"),
            pl.col("timestamp").rank("ordinal").over("id").cast(pl.Int32).alias("horizon"),
            pl.col(median_column).cast(pl.Float64).alias("forecast"),
            *[
                pl.col(_level_column(level, frame.columns))
                .cast(pl.Float64)
                .alias(quantile_column(level))
                for level in levels
            ],
        )


def _level_column(level: float, columns: list[str]) -> str:
    """Chronos names quantile columns by their level, e.g. '0.1'."""
    for candidate in (f"{level:g}", str(level)):
        if candidate in columns:
            return candidate
    raise KeyError(f"Chronos returned no column for quantile {level}; got {columns}")


def _median_column(quantile_levels: list[float]) -> str:
    """Name of the column holding the median forecast.

    Chronos names quantile columns by their level. If 0.5 was not requested the
    model still returns a `predictions` column, which is its point forecast.
    """
    return "0.5" if 0.5 in quantile_levels else "predictions"
