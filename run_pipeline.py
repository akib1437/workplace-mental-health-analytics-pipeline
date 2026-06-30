"""Run the complete configuration-driven, disclosure-controlled analysis pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.adapters import load_dataset
from src.analyse_subgroups import analyse_subgroups
from src.generate_report import generate_outputs
from src.score_scales import score_scales
from src.validate_schema import normalize_item_columns, validate_schema


def normalize_label(value: object) -> str:
    return str(value).strip().replace("None-minimal", "Minimal").casefold()


def verification_log(reliability: pd.DataFrame, severity: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for scale_key, scale in config["scales"].items():
        observed_alpha = float(reliability.loc[reliability["scale"] == scale_key, "cronbach_alpha"].iloc[0])
        observed_pct = float(
            severity.loc[
                (severity["scale"] == scale_key)
                & (severity["severity"] == "Moderate or higher"),
                "percentage",
            ].iloc[0]
        )
        expected_alpha = scale.get("expected_alpha")
        expected_pct = scale.get("expected_moderate_or_higher_pct")
        rows.extend(
            [
                {
                    "scale": scale_key,
                    "check": "Cronbach alpha",
                    "expected": expected_alpha,
                    "observed": observed_alpha,
                    "tolerance": 0.000001,
                    "passed": abs(observed_alpha - expected_alpha) <= 0.000001 if expected_alpha is not None else True,
                },
                {
                    "scale": scale_key,
                    "check": "Moderate-or-higher percentage",
                    "expected": expected_pct,
                    "observed": observed_pct,
                    "tolerance": 0.000001,
                    "passed": abs(observed_pct - expected_pct) <= 0.000001 if expected_pct is not None else True,
                },
            ]
        )
    return pd.DataFrame(rows)


def source_severity_reconciliation(scored: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows: list[dict] = []
    source_columns = config["dataset"].get("source_severity_columns", {})
    for scale_key, source_column in source_columns.items():
        if source_column not in scored.columns:
            continue
        calculated = config["scales"][scale_key]["severity_column"]
        matches = scored[source_column].map(normalize_label).eq(scored[calculated].map(normalize_label))
        rows.append(
            {
                "scale": scale_key,
                "source_severity_column": source_column,
                "calculated_severity_column": calculated,
                "n_compared": int(matches.notna().sum()),
                "mismatch_count_after_normalization": int((~matches).sum()),
                "status": "matched" if bool(matches.all()) else "review_required",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    raw = load_dataset(config)
    normalized = normalize_item_columns(raw, config)
    schema_log = validate_schema(normalized, config)
    if not schema_log["passed"]:
        raise RuntimeError(f"Schema validation failed. Review outputs/schema_validation.json after fixing: {schema_log}")

    scored, reliability, severity = score_scales(normalized, config)
    subgroup_results, disclosed_crosstabs, disclosure_log = analyse_subgroups(scored, config)
    verification = verification_log(reliability, severity, config)
    reconciliation = source_severity_reconciliation(scored, config)
    if not reconciliation.empty:
        verification = pd.concat(
            [
                verification,
                reconciliation.assign(
                    check="Source severity reconciliation",
                    expected="0 mismatches after normalization",
                    observed=reconciliation["mismatch_count_after_normalization"],
                    tolerance=0,
                    passed=reconciliation["mismatch_count_after_normalization"].eq(0),
                )[["scale", "check", "expected", "observed", "tolerance", "passed"]],
            ],
            ignore_index=True,
        )

    output_dir = Path(config["outputs"]["directory"])
    generate_outputs(
        output_dir,
        schema_log,
        reliability,
        severity,
        subgroup_results,
        disclosed_crosstabs,
        disclosure_log,
        verification,
    )
    print(f"Pipeline completed. Aggregate outputs written to: {output_dir.resolve()}")
    print(verification.to_string(index=False))


if __name__ == "__main__":
    main()
