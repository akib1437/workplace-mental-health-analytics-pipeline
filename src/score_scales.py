"""Configuration-driven scoring, severity classification, and reliability estimation."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def cronbach_alpha(items: pd.DataFrame) -> float:
    complete = items.dropna()
    number_of_items = complete.shape[1]
    if number_of_items < 2 or len(complete) < 2:
        return float("nan")
    item_variances = complete.var(axis=0, ddof=1).sum()
    total_variance = complete.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return float("nan")
    return float(number_of_items / (number_of_items - 1) * (1 - item_variances / total_variance))


def corrected_item_total_correlations(items: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for column in items.columns:
        other_columns = [candidate for candidate in items.columns if candidate != column]
        result[column] = float(items[column].corr(items[other_columns].sum(axis=1)))
    return result


def classify_severity(score: float, thresholds: list[dict[str, Any]]) -> str:
    for threshold in thresholds:
        if score <= threshold["upper_inclusive"]:
            return str(threshold["label"])
    return str(thresholds[-1]["label"])


def score_scales(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored = df.copy()
    reliability_rows: list[dict[str, Any]] = []
    severity_rows: list[dict[str, Any]] = []

    for scale_key, scale in config["scales"].items():
        item_columns = scale["item_columns"]
        score_column = scale["score_column"]
        severity_column = scale["severity_column"]
        binary_column = scale["moderate_or_higher_column"]

        scored[score_column] = scored[item_columns].sum(axis=1, min_count=len(item_columns))
        scored[severity_column] = scored[score_column].map(
            lambda value: classify_severity(value, scale["thresholds"]) if pd.notna(value) else np.nan
        )
        scored[binary_column] = np.where(
            scored[score_column].ge(scale["moderate_or_higher_score"]),
            "Moderate or higher",
            "Below moderate",
        )
        scored.loc[scored[score_column].isna(), binary_column] = np.nan

        items = scored[item_columns]
        alpha = cronbach_alpha(items)
        correlations = corrected_item_total_correlations(items)
        reliability_rows.append(
            {
                "scale": scale_key,
                "display_name": scale["display_name"],
                "n_complete": int(items.dropna().shape[0]),
                "cronbach_alpha": alpha,
                "minimum_corrected_item_total_correlation": min(correlations.values()),
                "maximum_corrected_item_total_correlation": max(correlations.values()),
                "item_total_correlations": "; ".join(f"{key}={value:.3f}" for key, value in correlations.items()),
            }
        )

        distribution = scored[severity_column].value_counts(dropna=False)
        severity_order = [threshold["label"] for threshold in scale["thresholds"]]
        for label in severity_order:
            count = int(distribution.get(label, 0))
            severity_rows.append(
                {
                    "scale": scale_key,
                    "display_name": scale["display_name"],
                    "severity": label,
                    "count": count,
                    "percentage": float(count / len(scored) * 100),
                }
            )
        moderate_count = int(scored[score_column].ge(scale["moderate_or_higher_score"]).sum())
        severity_rows.append(
            {
                "scale": scale_key,
                "display_name": scale["display_name"],
                "severity": "Moderate or higher",
                "count": moderate_count,
                "percentage": float(moderate_count / len(scored) * 100),
            }
        )

    return scored, pd.DataFrame(reliability_rows), pd.DataFrame(severity_rows)
