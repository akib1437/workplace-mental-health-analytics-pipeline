"""Configuration-driven input adapters for single Excel/CSV and multi-CSV datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _read_table(path: Path, file_format: str, sheet_name: str | None = None) -> pd.DataFrame:
    if file_format == "xlsx":
        return pd.read_excel(path, sheet_name=sheet_name)
    if file_format == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {file_format}")


def _prefix_non_key_columns(table: pd.DataFrame, table_name: str, key: str) -> pd.DataFrame:
    """Prevent collisions such as `question1` in PHQ-9 and GAD-7 source files.

    For multi-file data, the shared ID remains unchanged and every other source column receives
    a deterministic prefix such as `phq9__question1`. This allows the YAML mapping to state
    exactly which source table supplied each generic analytical field.
    """
    rename_map = {
        column: f"{table_name}__{column}"
        for column in table.columns
        if column != key
    }
    return table.rename(columns=rename_map)


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Load and optionally merge configured source files, then apply a generic field map."""
    dataset = config["dataset"]
    file_format = dataset["format"]

    if file_format in {"xlsx", "csv"}:
        path = Path(dataset["path"])
        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}. See data/private/README.md for the required placement."
            )
        df = _read_table(path, file_format, dataset.get("sheet_name"))

    elif file_format == "multi_csv":
        tables: dict[str, pd.DataFrame] = {}
        for name, info in dataset["files"].items():
            path = Path(info["path"])
            if not path.exists():
                raise FileNotFoundError(f"Required external file not found: {path}")
            tables[name] = _read_table(path, info["format"])

        join = dataset["join"]
        key = join["participant_id"]
        if str(key).startswith("REPLACE_"):
            raise ValueError(
                "The external mapping is incomplete. Replace dataset.join.participant_id in config/zenodo_config.yaml."
            )
        base_name = join["base_table"]
        if key not in tables[base_name].columns:
            raise KeyError(f"Join key '{key}' was not found in the configured base table '{base_name}'.")

        # Preserve base-table headers (e.g., gender, age). Prefix every non-ID field from
        # each additional table so PHQ-9 and GAD-7 question columns cannot overwrite one another.
        df = tables[base_name]
        for table_name, table in tables.items():
            if table_name == base_name:
                continue
            if key not in table.columns:
                raise KeyError(f"Join key '{key}' was not found in {table_name}.csv")
            prepared = _prefix_non_key_columns(table, table_name, key)
            df = df.merge(prepared, on=key, how=join.get("how", "inner"), validate="one_to_one")

    else:
        raise ValueError(f"Unsupported dataset format: {file_format}")

    field_mapping = dataset.get("field_mapping", {})
    if field_mapping:
        inverse_map = {source: generic for generic, source in field_mapping.items()}
        missing_sources = [source for source in inverse_map if source not in df.columns]
        if missing_sources:
            raise KeyError(f"Configured source columns not found: {missing_sources}")
        df = df.rename(columns=inverse_map)

    return df
