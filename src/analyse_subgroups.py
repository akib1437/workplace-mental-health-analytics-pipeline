"""Exploratory subgroup analysis with chi-square diagnostics, FDR correction, and Cramer's V."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

from src.apply_disclosure_rules import disclose_crosstab


def cramers_v(chi_square: float, n: int, rows: int, columns: int) -> float:
    """Calculate Cramer's V for a contingency table."""
    denominator = n * min(rows - 1, columns - 1)

    if denominator <= 0:
        return float("nan")

    return float(np.sqrt(chi_square / denominator))


def _interpret_result(row: pd.Series) -> str:
    """Return a bounded, aggregate-only interpretation for one subgroup result."""
    if not bool(row["chi_square_reportable"]):
        return (
            "Do not interpret inferential result: chi-square expected-cell "
            "eligibility criteria were not met."
        )

    if bool(row["illustrative_reporting_signal"]):
        return (
            "Eligible aggregate association meets the combined FDR and "
            "effect-size illustrative reporting-signal rule; it is not "
            "diagnostic, causal, or an employment decision rule."
        )

    return (
        "Eligible exploratory aggregate association did not meet the combined "
        "FDR and effect-size illustrative reporting-signal rule."
    )


def analyse_subgroups(
    scored: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run disclosure-controlled exploratory subgroup analyses."""

    analysis = config["analysis"]
    chi_square_config = analysis["chi_square"]

    minimum_expected_cell_count = float(
        chi_square_config["min_expected_cell_count"]
    )
    maximum_fraction_expected_below_min = float(
        chi_square_config["max_fraction_expected_below_min"]
    )
    minimum_expected_cell_allowed = float(
        chi_square_config["minimum_expected_cell_allowed"]
    )

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
            table = pd.crosstab(
                analysis_frame[subgroup],
                analysis_frame[outcome],
            )

            if table.shape[0] < 2 or table.shape[1] < 2:
                continue

            chi_square, p_value, degrees_of_freedom, expected = chi2_contingency(
                table
            )

            expected_cells_below_minimum = int(
                (expected < minimum_expected_cell_count).sum()
            )
            expected_cells_below_absolute_floor = int(
                (expected < minimum_expected_cell_allowed).sum()
            )
            fraction_expected_cells_below_minimum = float(
                expected_cells_below_minimum / expected.size
            )

            no_expected_cell_below_absolute_floor = (
                expected_cells_below_absolute_floor == 0
            )
            fraction_expected_cells_acceptable = (
                fraction_expected_cells_below_minimum
                <= maximum_fraction_expected_below_min
            )

            reportable = bool(
                no_expected_cell_below_absolute_floor
                and fraction_expected_cells_acceptable
            )

            v_value = cramers_v(
                chi_square,
                int(table.to_numpy().sum()),
                *table.shape,
            )

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
                "expected_cells_below_minimum": expected_cells_below_minimum,
                "expected_cells_below_absolute_floor": (
                    expected_cells_below_absolute_floor
                ),
                "fraction_expected_cells_below_minimum": (
                    fraction_expected_cells_below_minimum
                ),
                "no_expected_cell_below_absolute_floor": (
                    no_expected_cell_below_absolute_floor
                ),
                "fraction_expected_cells_acceptable": (
                    fraction_expected_cells_acceptable
                ),
                "chi_square_reportable": reportable,
                "illustrative_reporting_signal": False,
                "interpretation": "",
            }

            results.append(result)

            displayed, entries = disclose_crosstab(
                table,
                config,
                subgroup,
                outcome,
            )

            long_form = displayed.reset_index().melt(
                id_vars=[subgroup],
                var_name="outcome_category",
                value_name="released_cell",
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

        illustrative_threshold = float(
            analysis["effect_size"]["illustrative_reporting_signal_threshold"]
        )

        result_frame["illustrative_reporting_signal"] = (
            result_frame["chi_square_reportable"].astype(bool)
            & result_frame["p_value_fdr"].le(float(analysis["fdr_alpha"]))
            & result_frame["cramers_v"].ge(illustrative_threshold)
        )

        result_frame["interpretation"] = result_frame.apply(
            _interpret_result,
            axis=1,
        )

    crosstab_frame = (
        pd.concat(disclosed_tables, ignore_index=True)
        if disclosed_tables
        else pd.DataFrame()
    )

    disclosure_frame = pd.DataFrame(disclosure_log)

    return result_frame, crosstab_frame, disclosure_frame