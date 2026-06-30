"""Schema validation and response normalization for ordinal psychometric survey inputs."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def parse_ordinal_response(value: Any) -> float:
    """Return the first valid ordinal integer from a response value."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value,
        bool,
    ):
        return float(value)

    match = re.search(r"(?<!\d)([0-3])(?!\d)", str(value))
    return float(match.group(1)) if match else np.nan


def normalize_item_columns(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Normalize present item columns without hiding missing-column errors."""
    normalized = df.copy()

    for scale in config["scales"].values():
        for column in scale["item_columns"]:
            if column in normalized.columns:
                normalized[column] = normalized[column].map(
                    parse_ordinal_response
                )

    return normalized


def validate_schema(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return an aggregate validation audit without exposing respondent rows."""

    all_item_columns = [
        column
        for scale in config["scales"].values()
        for column in scale["item_columns"]
    ]

    required_columns = set(
        all_item_columns + config["analysis"]["subgroup_columns"]
    )
    missing_columns = sorted(required_columns.difference(df.columns))

    response_values = set(config["validation"]["item_response_values"])
    available_item_columns = [
        column for column in all_item_columns if column in df.columns
    ]

    invalid_values_by_item: dict[str, int] = {}
    missing_by_item: dict[str, int] = {}

    for column in available_item_columns:
        missing_by_item[column] = int(df[column].isna().sum())
        invalid_values_by_item[column] = int(
            (~df[column].isin(response_values) & df[column].notna()).sum()
        )

    missing_item_columns = sorted(
        set(all_item_columns).difference(available_item_columns)
    )

    all_required_columns_present = not missing_columns
    all_item_values_valid = (
        all_required_columns_present
        and sum(invalid_values_by_item.values()) == 0
    )
    all_items_complete = (
        all_required_columns_present
        and sum(missing_by_item.values()) == 0
    )

    duplicate_rule = config["validation"].get("duplicate_rule", "full_row")
    maximum_duplicate_count = int(
        config["validation"].get("maximum_duplicate_count", 0)
    )

    duplicate_check_possible = True
    duplicate_basis = "full_row"

    if duplicate_rule == "participant_id":
        participant_id = (
            config["dataset"].get("join", {}).get("participant_id")
            or config["dataset"].get("id_column")
        )

        duplicate_basis = str(participant_id)

        if participant_id and participant_id in df.columns:
            duplicate_count = int(
                df.duplicated(subset=[participant_id]).sum()
            )
        else:
            duplicate_count = None
            duplicate_check_possible = False
    else:
        duplicate_count = int(df.duplicated().sum())

    duplicate_count_within_limit = bool(
        duplicate_check_possible
        and duplicate_count is not None
        and duplicate_count <= maximum_duplicate_count
    )

    expected_rows = config["dataset"].get("expected_rows")
    expected_row_count_passed = (
        expected_rows is None or int(len(df)) == int(expected_rows)
    )

    require_complete_item_responses = bool(
        config["validation"].get("require_complete_item_responses", True)
    )

    passed = bool(
        expected_row_count_passed
        and all_required_columns_present
        and all_item_values_valid
        and (
            not require_complete_item_responses
            or all_items_complete
        )
        and duplicate_count_within_limit
    )

    return {
        "dataset": config["dataset"]["name"],
        "n_rows": int(len(df)),
        "expected_rows": expected_rows,
        "expected_row_count_passed": bool(expected_row_count_passed),
        "missing_required_columns": missing_columns,
        "missing_item_columns": missing_item_columns,
        "missing_by_item": missing_by_item,
        "invalid_values_by_item": invalid_values_by_item,
        "all_required_columns_present": all_required_columns_present,
        "all_item_values_valid": all_item_values_valid,
        "all_items_complete": all_items_complete,
        "duplicate_rule": duplicate_rule,
        "duplicate_basis": duplicate_basis,
        "duplicate_check_possible": duplicate_check_possible,
        "duplicate_count": duplicate_count,
        "maximum_duplicate_count": maximum_duplicate_count,
        "duplicate_count_within_limit": duplicate_count_within_limit,
        "passed": passed,
    }