"""Exploratory subgroup analysis with chi-square diagnostics, FDR correction, and Cramer's V."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

from src.apply_disclosure_rules import disclose_crosstab


def cramers_v(chi_square: float, n: int, rows: int, columns: int) -> float:
    denominator = n * min(rows - 1, columns - 1)
    if denominator <= 0:
        return float("nan")
    return float(np.sqrt(chi_square / denominator))


def analyse_subgroups(
    scored: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis = config["analysis"]
    outcomes: list[tuple[str, str]] = []
    for scale in config["scales"].values():
        outcomes.append((scale["severity_column"], "severity"))
        outcomes.append((scale["moderate_or_higher_column"], "binary"))

    results: list[dict[str, Any]] = []
    disclosed_tables: list[pd.DataFrame] = []
    disclosure_log: list[dict[str, Any]] = []

    for outcome, outcome_type in outcomes:
        for subgroup in analysis["subgroup_columns"]:
            analysis_frame = scored[[subgroup, outcome]].dropna()
            table = pd.crosstab(analysis_frame[subgroup], analysis_frame[outcome])
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue

            chi_square, p_value, degrees_of_freedom, expected = chi2_contingency(table)
            expected_low = int((expected < analysis["chi_square"]["min_expected_cell_count"]).sum())
            expected_low_fraction = float(expected_low / expected.size)
            reportable = expected_low_fraction <= float(
                analysis["chi_square"]["max_fraction_expected_below_min"]
            )
            v_value = cramers_v(chi_square, int(table.to_numpy().sum()), *table.shape)
            flag_threshold = float(analysis["effect_size"]["illustrative_reporting_signal_threshold"])

            result = {
                "outcome": outcome,
                "outcome_type": outcome_type,
                "subgroup": subgroup,
                "n": int(table.to_numpy().sum()),
                "chi_square": float(chi_square),
                "degrees_of_freedom": int(degrees_of_freedom),
                "p_value_raw": float(p_value),
                "p_value_fdr": np.nan,
                "cramers_v": v_value,
                "expected_cells_below_minimum": expected_low,
                "fraction_expected_cells_below_minimum": expected_low_fraction,
                "chi_square_reportable": bool(reportable),
                "illustrative_reporting_signal": bool(reportable and v_value >= flag_threshold),
                "interpretation": (
                    "Exploratory aggregate association; not diagnostic, causal, or a management decision rule."
                    if reportable
                    else "Do not interpret inferential result: chi-square expected-cell criterion not met."
                ),
            }
            results.append(result)

            displayed, entries = disclose_crosstab(table, config, subgroup, outcome)
            long_form = displayed.reset_index().melt(
                id_vars=[subgroup], var_name="outcome_category", value_name="released_cell"
            )
            long_form.insert(0, "outcome", outcome)
            long_form.insert(1, "outcome_type", outcome_type)
            disclosed_tables.append(long_form)
            disclosure_log.extend(entries)

    result_frame = pd.DataFrame(results)
    if not result_frame.empty:
        for outcome, indexes in result_frame.groupby("outcome").groups.items():
            eligible = result_frame.loc[indexes, "chi_square_reportable"]
            eligible_indexes = eligible[eligible].index
            if len(eligible_indexes):
                _, adjusted, _, _ = multipletests(
                    result_frame.loc[eligible_indexes, "p_value_raw"],
                    alpha=float(analysis["fdr_alpha"]),
                    method=analysis["fdr_method"],
                )
                result_frame.loc[eligible_indexes, "p_value_fdr"] = adjusted

    crosstab_frame = pd.concat(disclosed_tables, ignore_index=True) if disclosed_tables else pd.DataFrame()
    disclosure_frame = pd.DataFrame(disclosure_log)
    return result_frame, crosstab_frame, disclosure_frame
