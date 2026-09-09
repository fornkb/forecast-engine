"""Pooled gradient-boosted forecaster.

One model is trained across every series at once, with the horizon as a feature
(direct multi-horizon), so a single booster serves all steps without recursive
error accumulation. Targets and level features are scaled per series; predictions
are rescaled back to native units.

Swapping this for XGBoost means a sibling module implementing `Forecaster` — the
panel construction, scaling, and rescaling here are model-agnostic.
"""

from __future__ import annotations

import gc

import lightgbm as lgb
import numpy as np
import polars as pl

from engine.features.panel import (
    DEFAULT_LAGS,
    DEFAULT_WINDOWS,
    add_history_features,
    build_inference_frame,
    build_training_frame,
    encode_static_features,
    feature_columns,
)
from engine.forecast.base import ForecastContext, Forecaster, register_forecaster


def _feature_matrix(frame: pl.DataFrame, features: list[str]) -> np.ndarray:
    """Feature matrix at float32 — the precision LightGBM actually uses."""
    return frame.select([pl.col(name).cast(pl.Float32) for name in features]).to_numpy()


@register_forecaster
class GlobalLightGBMForecaster(Forecaster):
    name = "lightgbm_global"

    def fit_predict(
        self, history: pl.DataFrame, horizon: int, ctx: ForecastContext
    ) -> pl.DataFrame:
        static_features = tuple(ctx.client.models.static_features)
        available = tuple(f for f in static_features if f in ctx.series_meta.columns)
        features = feature_columns(DEFAULT_LAGS, DEFAULT_WINDOWS, available)
        categorical = [f"{column}_code" for column in available]

        base = add_history_features(history, DEFAULT_LAGS, DEFAULT_WINDOWS)
        static = encode_static_features(ctx.series_meta, available)

        train = build_training_frame(base, horizon, ctx.granularity).join(
            static, on="series_id", how="left"
        )

        window = ctx.client.models.lightgbm.max_training_periods
        if window is not None:
            cutoff = base["period"].max()
            earliest = (
                base.filter(pl.col("period") <= cutoff)
                .select(pl.col("period").unique().sort().tail(window).first())
                .item()
            )
            train = train.filter(pl.col("period") >= earliest)
        if train.is_empty():
            raise ValueError(
                "No training rows available — history is shorter than the forecast horizon"
            )

        params = ctx.client.models.lightgbm.model_dump()
        num_boost_round = params.pop("num_boost_round")
        params.pop("max_training_periods", None)

        # LightGBM bins features to float32 internally, so building the matrix at
        # float64 doubles peak memory for nothing. On a panel of millions of rows
        # that difference decides whether the run completes.
        matrix = _feature_matrix(train, features)
        labels = train["target_scaled"].to_numpy()

        # The training frame carries every lag, rolling and scale intermediate —
        # far more columns than the feature list. Release it before LightGBM
        # allocates its own copy.
        del train
        gc.collect()

        dataset = lgb.Dataset(
            matrix,
            label=labels,
            feature_name=features,
            categorical_feature=categorical,
        )
        booster = lgb.train(params, dataset, num_boost_round=num_boost_round)
        del dataset, matrix, labels
        gc.collect()

        inference = build_inference_frame(base, horizon, ctx.granularity).join(
            static, on="series_id", how="left"
        )
        scaled_predictions = booster.predict(_feature_matrix(inference, features))

        return inference.select(
            "series_id",
            pl.col("target_period").alias("period"),
            "horizon",
            (pl.Series("scaled", np.asarray(scaled_predictions)) * pl.col("scale")).alias(
                "forecast"
            ),
        )
