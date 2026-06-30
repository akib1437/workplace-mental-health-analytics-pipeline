"""Configuration-driven sensitivity analysis for workplace reporting rules."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.adapters import load_dataset
from src.analyse_subgroups import analyse_subgroups
from src.score_scales import score_scales
from src.validate_schema import normalize_item_columns, validate_schema


def signal_summary(results: pd.DataFrame) -> str:
    """Return a compact list of outcome/subgroup signal combinations."""
    signals = results.loc[
        results["illustrative_reporting_signal"].astype(bool),
        ["outcome", "subgroup"],
    ].sort_values(["outcome", "subgroup"])

    if signals.empty:
        return "None"

    return "; ".join(
        f"{row.outcome} | {row.subgroup}"
        for row in signals.itertuples(index=False)
    )


def summary_row(
    results: pd.DataFrame,
    disclosure_log: pd.DataFrame,
    setting_name: str,
    setting_value: float | int,
) -> dict[str, Any]:
    """Create one compact reporting-rule sensitivity summary row."""
    eligible = results.loc[results["chi_square_reportable"].astype(bool)]

    return {
        "setting_name": setting_name,
        "setting_value": setting_value,
        "subgroup_tests_evaluated": int(len(results)),
        "reportable_subgroup_tests": int(len(eligible)),
        "suppressed_cells": int(len(disclosure_log)),
        "illustrative_signal_count": int(
            results["illustrative_reporting_signal"].astype(bool).sum()
        ),
        "illustrative_signal_summary": signal_summary(results),
    }


def outcome_rankings(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank reportable subgroup associations by Cramer's V per outcome."""
    eligible = results.loc[
        results["chi_square_reportable"].astype(bool)
    ].copy()

    ranking_rows: list[dict[str, Any]] = []
    leader_rows: list[dict[str, Any]] = []

    for (outcome, outcome_type), group in eligible.groupby(
        ["outcome", "outcome_type"],
        sort=True,
    ):
        ordered = group.sort_values(
            ["cramers_v", "subgroup"],
            ascending=[False, True],
        ).reset_index(drop=True)

        for rank, row in enumerate(ordered.itertuples(index=False), start=1):
            ranking_rows.append(
                {
                    "outcome": outcome,
                    "outcome_type": outcome_type,
                    "rank_by_cramers_v": rank,
                    "subgroup": row.subgroup,
                    "cramers_v": row.cramers_v,
                    "p_value_fdr": row.p_value_fdr,
                    "illustrative_reporting_signal": (
                        row.illustrative_reporting_signal
                    ),
                }
            )

        if not ordered.empty:
            leader = ordered.iloc[0]
            leader_rows.append(
                {
                    "outcome": outcome,
                    "outcome_type": outcome_type,
                    "leading_subgroup": leader["subgroup"],
                    "leading_cramers_v": leader["cramers_v"],
                    "leading_p_value_fdr": leader["p_value_fdr"],
                    "leading_is_illustrative_signal": (
                        leader["illustrative_reporting_signal"]
                    ),
                }
            )

    return pd.DataFrame(ranking_rows), pd.DataFrame(leader_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the workplace YAML configuration file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    raw = load_dataset(config)
    normalized = normalize_item_columns(raw, config)
    schema_log = validate_schema(normalized, config)

    if not schema_log["passed"]:
        raise RuntimeError(
            "Schema validation failed. Fix the input before sensitivity analysis."
        )

    scored, _, _ = score_scales(normalized, config)

    sensitivity = config["analysis"]["sensitivity"]
    baseline_k = int(config["disclosure_control"]["minimum_cell_count"])
    baseline_v = float(
        config["analysis"]["effect_size"][
            "illustrative_reporting_signal_threshold"
        ]
    )

    output_dir = Path(config["outputs"]["directory"]) / "sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    disclosure_rows: list[dict[str, Any]] = []

    for minimum_cell_count in sensitivity["disclosure_minimum_cell_counts"]:
        temporary_config = copy.deepcopy(config)
        temporary_config["disclosure_control"]["minimum_cell_count"] = int(
            minimum_cell_count
        )

        results, _, disclosure_log = analyse_subgroups(scored, temporary_config)

        disclosure_rows.append(
            summary_row(
                results,
                disclosure_log,
                "minimum_cell_count",
                int(minimum_cell_count),
            )
        )

    disclosure_frame = pd.DataFrame(disclosure_rows)
    disclosure_frame.to_csv(
        output_dir / "sensitivity_disclosure_thresholds.csv",
        index=False,
    )

    effect_rows: list[dict[str, Any]] = []

    for threshold in sensitivity[
        "illustrative_reporting_signal_thresholds"
    ]:
        temporary_config = copy.deepcopy(config)
        temporary_config["analysis"]["effect_size"][
            "illustrative_reporting_signal_threshold"
        ] = float(threshold)

        results, _, disclosure_log = analyse_subgroups(scored, temporary_config)

        effect_rows.append(
            summary_row(
                results,
                disclosure_log,
                "illustrative_reporting_signal_threshold",
                float(threshold),
            )
        )

    effect_frame = pd.DataFrame(effect_rows)
    effect_frame.to_csv(
        output_dir / "sensitivity_effect_size_thresholds.csv",
        index=False,
    )

    baseline_results, _, baseline_disclosure_log = analyse_subgroups(
        scored,
        config,
    )

    rankings, leaders = outcome_rankings(baseline_results)

    rankings.to_csv(
        output_dir / "sensitivity_outcome_rankings.csv",
        index=False,
    )

    leaders.to_csv(
        output_dir / "sensitivity_outcome_leaders.csv",
        index=False,
    )

    summary = [
        "# Workplace reporting-rule sensitivity analysis",
        "",
        f"- Baseline disclosure threshold: k = {baseline_k}",
        f"- Baseline illustrative Cramer's V threshold: {baseline_v:.2f}",
        (
            "- FDR rule: Benjamini-Hochberg adjustment applied only to "
            "chi-square-reportable subgroup tests."
        ),
        (
            "- Sensitivity analysis evaluates operational stability of "
            "reporting rules; it does not establish causal robustness."
        ),
        "",
        "## Generated files",
        "",
        "- `sensitivity_disclosure_thresholds.csv`",
        "- `sensitivity_effect_size_thresholds.csv`",
        "- `sensitivity_outcome_rankings.csv`",
        "- `sensitivity_outcome_leaders.csv`",
        "",
        (
            f"Baseline run suppressed {len(baseline_disclosure_log)} "
            "small cells under the configured disclosure rule."
        ),
    ]

    (output_dir / "sensitivity_summary.md").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print("Sensitivity analysis completed.")
    print(f"Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()