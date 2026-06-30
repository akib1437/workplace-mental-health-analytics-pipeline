"""Schema validation and response normalization for ordinal psychometric survey inputs."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def parse_ordinal_response(value: Any) -> float:
    """Return the first integer in an ordinal response (e.g., '1 — Several days')."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    match = re.search(r"(?<!\d)([0-3])(?!\d)", str(value))
    return float(match.group(1)) if match else np.nan


def normalize_item_columns(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    normalized = df.copy()
    for scale in config["scales"].values():
        for column in scale["item_columns"]:
            normalized[column] = normalized[column].map(parse_ordinal_response)
    return normalized


def validate_schema(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Return a complete audit log without exposing respondent-level records."""
    all_item_columns = [
        column
        for scale in config["scales"].values()
        for column in scale["item_columns"]
    ]
    required_columns = set(all_item_columns + config["analysis"]["subgroup_columns"])
    missing_columns = sorted(required_columns.difference(df.columns))
    response_values = set(config["validation"]["item_response_values"])

    invalid_values: dict[str, int] = {}
    missing_by_item: dict[str, int] = {}
    for column in all_item_columns:
        missing_by_item[column] = int(df[column].isna().sum())
        invalid_values[column] = int((~df[column].isin(response_values) & df[column].notna()).sum())

    duplicate_rule = config["validation"].get("duplicate_rule", "full_row")
    if duplicate_rule == "participant_id" and config["dataset"].get("join", {}).get("participant_id"):
        id_column = config["dataset"]["join"]["participant_id"]
        duplicate_count = int(df.duplicated(subset=[id_column]).sum())
    else:
        duplicate_count = int(df.duplicated().sum())

    expected_rows = config["dataset"].get("expected_rows")
    row_check = expected_rows is None or int(len(df)) == int(expected_rows)

    return {
        "dataset": config["dataset"]["name"],
        "n_rows": int(len(df)),
        "expected_rows": expected_rows,
        "expected_row_count_passed": bool(row_check),
        "missing_required_columns": missing_columns,
        "missing_by_item": missing_by_item,
        "invalid_values_by_item": invalid_values,
        "duplicate_count": duplicate_count,
        "all_required_columns_present": not missing_columns,
        "all_item_values_valid": sum(invalid_values.values()) == 0,
        "all_items_complete": sum(missing_by_item.values()) == 0,
        "passed": bool(
            row_check
            and not missing_columns
            and sum(invalid_values.values()) == 0
            and (
                not config["validation"].get("require_complete_item_responses", True)
                or sum(missing_by_item.values()) == 0
            )
        ),
    }
