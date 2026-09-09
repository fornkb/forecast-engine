"""Pooled gradient-boosted forecaster.

One model is trained across every series at once, with the horizon as a feature
(direct multi-horizon), so a single booster serves all steps without recursive
error accumulation. Targets and level features are scaled per series; predictions
are rescaled back to native units.

Swapping this for XGBoost means a sibling module implementing `Forecaster` — the
panel construction, scaling, and rescaling here are model-agnostic.
"""

from __future__ import annotations

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

        dataset = lgb.Dataset(
            train.select(features).to_numpy(),
            label=train["target_scaled"].to_numpy(),
            feature_name=features,
            categorical_feature=categorical,
            free_raw_data=False,
        )
        booster = lgb.train(params, dataset, num_boost_round=num_boost_round)

        inference = build_inference_frame(base, horizon, ctx.granularity).join(
            static, on="series_id", how="left"
        )
        scaled_predictions = booster.predict(inference.select(features).to_numpy())

        return inference.select(
            "series_id",
            pl.col("target_period").alias("period"),
            "horizon",
            (pl.Series("scaled", np.asarray(scaled_predictions)) * pl.col("scale")).alias(
                "forecast"
            ),
        )
