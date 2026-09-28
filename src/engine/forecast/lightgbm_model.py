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

from engine.config.schema import quantile_column
from engine.features.panel import (
    add_history_features,
    attach_exogenous,
    build_inference_frame,
    build_training_frame,
    encode_static_features,
    feature_columns,
    lags_for,
    windows_for,
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
        exog_config = ctx.client.exogenous
        # Lags counted in the client's own periods, so a weekly panel reaches
        # back a year rather than a quarter.
        lags = lags_for(ctx.granularity)
        windows = windows_for(ctx.granularity)
        features = feature_columns(lags, windows, available, exog_config)
        categorical = [f"{column}_code" for column in available]

        base = add_history_features(history, lags, windows)
        static = encode_static_features(ctx.series_meta, available)

        settings = ctx.client.models.lightgbm
        train = attach_exogenous(
            build_training_frame(base, horizon, ctx.granularity, settings.target_cap_quantile),
            ctx.exogenous,
            exog_config,
        ).join(static, on="series_id", how="left")

        window = settings.max_training_periods
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

        params = settings.model_dump()
        num_boost_round = params.pop("num_boost_round")
        params.pop("max_training_periods", None)
        params.pop("target_cap_quantile", None)

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

        # Quantile regression fits one booster per level. The L1 objective
        # already estimates the median, so the 0.5 level reuses the point
        # forecast rather than paying for a near-identical fit.
        quantile_boosters = {
            level: lgb.train(
                {**params, "objective": "quantile", "alpha": level},
                dataset,
                num_boost_round=num_boost_round,
            )
            for level in ctx.quantile_levels
            if level != 0.5
        }

        del dataset, matrix, labels
        gc.collect()

        inference = attach_exogenous(
            build_inference_frame(base, horizon, ctx.granularity), ctx.exogenous, exog_config
        ).join(static, on="series_id", how="left")
        inference_matrix = _feature_matrix(inference, features)

        point = np.asarray(booster.predict(inference_matrix))
        outputs = [(pl.Series("scaled", point) * pl.col("scale")).alias("forecast")]

        for level in ctx.quantile_levels:
            scaled = (
                point
                if level == 0.5
                else np.asarray(quantile_boosters[level].predict(inference_matrix))
            )
            outputs.append(
                (pl.Series(f"scaled_{level}", scaled) * pl.col("scale")).alias(
                    quantile_column(level)
                )
            )

        return inference.select(
            "series_id",
            pl.col("target_period").alias("period"),
            "horizon",
            *outputs,
        )
