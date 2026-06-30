"""Aggregate-only result exports and concise report-ready outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8")


def generate_severity_figure(severity: pd.DataFrame, output_path: Path) -> None:
    selected = severity[severity["severity"] == "Moderate or higher"].copy()
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(selected["display_name"], selected["percentage"])
    axis.set_ylabel("Respondents (%)")
    axis.set_title("Moderate-or-higher screening-category symptom burden")
    for position, value in enumerate(selected["percentage"]):
        axis.text(position, value, f"{value:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_outputs(
    output_dir: Path,
    schema_log: dict[str, Any],
    reliability: pd.DataFrame,
    severity: pd.DataFrame,
    subgroup_results: pd.DataFrame,
    disclosed_crosstabs: pd.DataFrame,
    disclosure_log: pd.DataFrame,
    verification: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "schema_validation.json").write_text(json.dumps(schema_log, indent=2), encoding="utf-8")
    write_csv(reliability, output_dir / "reliability.csv")
    write_csv(severity, output_dir / "severity_distribution.csv")
    write_csv(subgroup_results, output_dir / "subgroup_results.csv")
    write_csv(disclosed_crosstabs, output_dir / "subgroup_crosstabs_disclosed.csv")
    write_csv(disclosure_log, output_dir / "disclosure_log.csv")
    write_csv(verification, output_dir / "verification_log.csv")
    generate_severity_figure(severity, output_dir / "severity_distribution.png")

    passed_checks = int(verification["passed"].sum()) if not verification.empty else 0
    summary = f"""# Pipeline run summary

- Schema validation passed: **{schema_log['passed']}**
- Input records processed: **{schema_log['n_rows']}**
- Reliability/scoring checks passed: **{passed_checks}/{len(verification)}**
- Disclosure rule: cells below the configured minimum are replaced in released crosstabs and logged in `disclosure_log.csv`.
- Reporting boundary: all generated outputs are aggregate-only. No participant-level rows are exported.

Use `reliability.csv`, `severity_distribution.csv`, `subgroup_results.csv`, and the disclosed crosstabs as source files for manuscript tables. Do not report subgroup p-values where `chi_square_reportable` is false.
"""
    (output_dir / "run_summary.md").write_text(summary, encoding="utf-8")
