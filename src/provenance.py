"""Run provenance manifests for aggregate-only pipeline executions."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PACKAGE_NAMES = (
    "pandas",
    "numpy",
    "scipy",
    "statsmodels",
    "PyYAML",
    "matplotlib",
    "openpyxl",
    "pytest",
)


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 hash of one file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    """Return a Git value without failing a pipeline run outside Git."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip() or "unavailable"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unavailable"


def package_versions() -> dict[str, str]:
    """Collect installed versions of the packages used by this project."""
    versions: dict[str, str] = {}

    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = "not_installed"

    return versions


def input_source_description(config: dict[str, Any]) -> dict[str, Any]:
    """Describe configured input sources without exposing respondent rows."""
    dataset = config["dataset"]

    if dataset["format"] == "multi_csv":
        source_files = {
            name: str(info["path"]).replace("\\", "/")
            for name, info in dataset["files"].items()
        }
    else:
        source_files = {
            "primary_input": str(dataset["path"]).replace("\\", "/")
        }

    return {
        "dataset_name": dataset["name"],
        "format": dataset["format"],
        "configured_source_files": source_files,
    }


def write_run_manifest(
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    schema_log: dict[str, Any],
) -> Path:
    """Write an aggregate-only provenance manifest after output generation."""
    manifest_path = output_dir / "run_manifest.json"

    output_hashes = {
        file_path.name: sha256_file(file_path)
        for file_path in sorted(output_dir.iterdir())
        if file_path.is_file() and file_path.name != manifest_path.name
    }

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "pipeline_version": git_output("describe", "--tags", "--always", "--dirty"),
        "git_commit_hash": git_output("rev-parse", "HEAD"),
        "configuration": {
            "filename": config_path.name,
            "relative_path": str(config_path).replace("\\", "/"),
            "sha256": sha256_file(config_path),
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "python_executable": Path(sys.executable).name,
            "package_versions": package_versions(),
        },
        "input": {
            **input_source_description(config),
            "record_count": int(schema_log["n_rows"]),
            "contains_row_level_data": False,
        },
        "reporting_boundary": config.get("project", {}).get(
            "report_boundary",
            "Aggregate-only reporting.",
        ),
        "output_file_sha256": output_hashes,
        "manifest_note": (
            "This manifest records configuration, environment, input metadata, "
            "and aggregate output hashes. It does not contain respondent rows."
        ),
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return manifest_path