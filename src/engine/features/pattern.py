"""Per-series demand-pattern profiling.

The pattern of a series is inferred from its own history rather than configured
per client: a single client routinely has stable, seasonal, trending, and
intermittent series side by side, so one client-level setting would be wrong for
most of them.

Classification is deliberately explainable — every series carries the statistics
that produced its label, so a disagreement can be inspected rather than argued
with.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from statsmodels.tsa.seasonal import STL

from engine.config.schema import PatternThresholds

PATTERNS = ("Stable", "Seasonal", "Growing", "Declining", "Intermittent", "Volatile")

PROFILE_SCHEMA = {
    "series_id": pl.Utf8,
    "n_periods": pl.Int64,
    "n_zeros": pl.Int64,
    "zero_fraction": pl.Float64,
    "mean_demand": pl.Float64,
    "total_volume": pl.Float64,
    "cv": pl.Float64,
    "adi": pl.Float64,
    "cv2": pl.Float64,
    "trend_strength": pl.Float64,
    "seasonal_strength": pl.Float64,
    "residual_cv": pl.Float64,
    "slope": pl.Float64,
    "pattern": pl.Utf8,
}


def _decompose(values: np.ndarray, seasonal_period: int) -> tuple[float, float, float]:
    """Return (trend_strength, seasonal_strength, residual_cv) via STL.

    Strengths follow the standard Wang–Smith–Hyndman definition: the share of
    variance the component explains over and above the remainder.
    """
    n = len(values)
    mean = float(np.mean(values))
    if n < 2 * seasonal_period or np.allclose(values, values[0]):
        return 0.0, 0.0, 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = STL(values, period=seasonal_period, robust=True).fit()
        except Exception:  # noqa: BLE001 - degenerate series should not abort profiling
            return 0.0, 0.0, 0.0

    resid_var = float(np.var(fit.resid))
    trend_var = float(np.var(fit.trend + fit.resid))
    seasonal_var = float(np.var(fit.seasonal + fit.resid))

    trend_strength = 0.0 if trend_var <= 0 else max(0.0, 1.0 - resid_var / trend_var)
    seasonal_strength = 0.0 if seasonal_var <= 0 else max(0.0, 1.0 - resid_var / seasonal_var)
    residual_cv = 0.0 if mean <= 0 else float(np.std(fit.resid)) / mean
    return trend_strength, seasonal_strength, residual_cv


def _relative_slope(values: np.ndarray) -> float:
    """Least-squares slope per period, expressed as a fraction of mean demand."""
    n = len(values)
    mean = float(np.mean(values))
    if n < 3 or mean <= 0:
        return 0.0
    slope = float(np.polyfit(np.arange(n, dtype=float), values, 1)[0])
    return slope / mean


def classify(stats: dict[str, float], thresholds: PatternThresholds) -> str:
    """Assign one of the six demand patterns from a series' statistics.

    Order matters: intermittency is checked first because a mostly-zero series
    produces meaningless trend and seasonality scores.
    """
    if (
        stats["zero_fraction"] >= thresholds.zero_fraction_intermittent
        or stats["adi"] >= thresholds.adi_intermittent
    ):
        return "Intermittent"

    seasonal = stats["seasonal_strength"]
    trend = stats["trend_strength"]

    if seasonal >= thresholds.seasonal_strength and seasonal >= trend:
        return "Seasonal"

    if trend >= thresholds.trend_strength:
        return "Growing" if stats["slope"] > 0 else "Declining"

    if stats["residual_cv"] >= thresholds.volatility_cv:
        return "Volatile"

    return "Stable"


def profile_series(
    history: pl.DataFrame, thresholds: PatternThresholds, seasonal_period: int
) -> pl.DataFrame:
    """Compute a statistical profile and pattern label for every series."""
    records: list[dict[str, object]] = []

    for (series_id,), group in history.sort("series_id", "period").group_by(
        ["series_id"], maintain_order=True
    ):
        values = group["quantity"].to_numpy().astype(float)
        n = len(values)
        nonzero = values[values > 0]
        n_zeros = int(n - len(nonzero))
        mean = float(np.mean(values)) if n else 0.0

        cv = float(np.std(values) / mean) if mean > 0 else 0.0
        adi = float(n / len(nonzero)) if len(nonzero) else float(n)
        nonzero_mean = float(np.mean(nonzero)) if len(nonzero) else 0.0
        cv2 = float((np.std(nonzero) / nonzero_mean) ** 2) if nonzero_mean > 0 else 0.0

        zero_fraction = n_zeros / n if n else 1.0
        is_intermittent = (
            zero_fraction >= thresholds.zero_fraction_intermittent
            or adi >= thresholds.adi_intermittent
        )

        # Decomposition is skipped for intermittent series: their trend and
        # seasonality scores are meaningless (which is why classification checks
        # intermittency first), and STL is by far the most expensive step here.
        if is_intermittent:
            trend_strength = seasonal_strength = residual_cv = 0.0
        else:
            trend_strength, seasonal_strength, residual_cv = _decompose(values, seasonal_period)

        stats = {
            "zero_fraction": zero_fraction,
            "adi": adi,
            "cv2": cv2,
            "trend_strength": trend_strength,
            "seasonal_strength": seasonal_strength,
            "residual_cv": residual_cv,
            "slope": _relative_slope(values),
        }

        records.append(
            {
                "series_id": series_id,
                "n_periods": n,
                "n_zeros": n_zeros,
                "zero_fraction": stats["zero_fraction"],
                "mean_demand": mean,
                "total_volume": float(np.sum(values)),
                "cv": cv,
                "adi": adi,
                "cv2": cv2,
                "trend_strength": trend_strength,
                "seasonal_strength": seasonal_strength,
                "residual_cv": residual_cv,
                "slope": stats["slope"],
                "pattern": classify(stats, thresholds),
            }
        )

    return pl.DataFrame(records, schema=PROFILE_SCHEMA)


def validate_against_labels(profiles: pl.DataFrame, dim: pl.DataFrame) -> pl.DataFrame | None:
    """Confusion matrix of predicted pattern vs. the source file's labels.

    Returns None when the source carries no labels, which is the expected case
    for real client data.
    """
    if "ground_truth_pattern" not in dim.columns:
        return None
    labelled = dim.filter(pl.col("ground_truth_pattern").is_not_null())
    if labelled.is_empty():
        return None

    joined = profiles.join(
        labelled.select("series_id", "ground_truth_pattern"), on="series_id", how="inner"
    )
    return (
        joined.group_by("ground_truth_pattern", "pattern")
        .len()
        .rename({"ground_truth_pattern": "ground_truth", "pattern": "predicted", "len": "n"})
        .sort("ground_truth", "predicted")
    )


def agreement_rate(matrix: pl.DataFrame) -> float:
    total = int(matrix["n"].sum())
    if total == 0:
        return 0.0
    correct = int(matrix.filter(pl.col("ground_truth") == pl.col("predicted"))["n"].sum())
    return correct / total
