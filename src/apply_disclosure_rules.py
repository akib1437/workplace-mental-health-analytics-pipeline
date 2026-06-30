"""Minimum-cell disclosure controls for aggregate tables."""
from __future__ import annotations

from typing import Any

import pandas as pd


def disclose_crosstab(
    table: pd.DataFrame,
    config: dict[str, Any],
    subgroup: str,
    outcome: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rules = config["disclosure_control"]
    minimum = int(rules["minimum_cell_count"])
    suppressed_text = rules["text_for_suppressed_cell"]
    displayed = table.astype(object).copy()
    log: list[dict[str, Any]] = []

    for row_label in table.index:
        for column_label in table.columns:
            value = int(table.loc[row_label, column_label])
            if value < minimum:
                displayed.loc[row_label, column_label] = suppressed_text
                log.append(
                    {
                        "subgroup": subgroup,
                        "outcome": outcome,
                        "row_category": str(row_label),
                        "column_category": str(column_label),
                        "rule": f"Cell count < {minimum}",
                        "action": "suppressed",
                    }
                )
    displayed.index.name = subgroup
    return displayed, log
