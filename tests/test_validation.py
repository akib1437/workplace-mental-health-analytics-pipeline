import numpy as np
import pandas as pd

from src.validate_schema import validate_schema


def make_config(
    duplicate_rule: str = "full_row",
) -> dict:
    return {
        "dataset": {
            "name": "test dataset",
            "expected_rows": None,
            "id_column": "participant_id",
        },
        "validation": {
            "item_response_values": [0, 1, 2, 3],
            "duplicate_rule": duplicate_rule,
            "maximum_duplicate_count": 0,
            "require_complete_item_responses": True,
        },
        "scales": {
            "phq9": {"item_columns": ["phq1"]},
            "gad7": {"item_columns": ["gad1"]},
        },
        "analysis": {
            "subgroup_columns": ["group"],
        },
    }


def valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["p1", "p2"],
            "phq1": [0, 1],
            "gad1": [1, 2],
            "group": ["A", "B"],
        }
    )


def test_missing_required_item_column_fails() -> None:
    df = valid_dataframe().drop(columns=["gad1"])

    result = validate_schema(df, make_config())

    assert result["passed"] is False
    assert "gad1" in result["missing_required_columns"]


def test_invalid_ordinal_value_fails() -> None:
    df = valid_dataframe()
    df.loc[0, "phq1"] = 4

    result = validate_schema(df, make_config())

    assert result["passed"] is False
    assert result["invalid_values_by_item"]["phq1"] == 1


def test_missing_item_response_fails() -> None:
    df = valid_dataframe()
    df.loc[0, "gad1"] = np.nan

    result = validate_schema(df, make_config())

    assert result["passed"] is False
    assert result["missing_by_item"]["gad1"] == 1


def test_duplicate_full_row_fails() -> None:
    df = pd.concat(
        [valid_dataframe(), valid_dataframe().iloc[[0]]],
        ignore_index=True,
    )

    result = validate_schema(df, make_config("full_row"))

    assert result["passed"] is False
    assert result["duplicate_count"] == 1
    assert result["duplicate_count_within_limit"] is False


def test_duplicate_participant_id_fails() -> None:
    df = valid_dataframe()
    df.loc[1, "participant_id"] = "p1"

    result = validate_schema(df, make_config("participant_id"))

    assert result["passed"] is False
    assert result["duplicate_basis"] == "participant_id"
    assert result["duplicate_count"] == 1
    assert result["duplicate_count_within_limit"] is False