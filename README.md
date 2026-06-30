# Configurable, Disclosure-Controlled Workplace Survey Pipeline

This repository implements a reproducible, configuration-driven analysis pipeline for aggregate PHQ-9/GAD-7 workplace-survey reporting. It integrates schema validation, scoring, severity classification, Cronbach's alpha, corrected item-total correlations, exploratory subgroup analysis, Benjamini-Hochberg false-discovery-rate adjustment, Cramer's V, and minimum-cell disclosure control.

## Reporting boundary

This artefact produces aggregate outputs only. It is not a diagnostic system, a clinical decision tool, a formal privacy-preserving mechanism, or a validated organisational intervention engine. Minimum-cell suppression is a disclosure-control measure, not a full privacy guarantee.

## What is intentionally excluded

The prior ML models, saved model files, prediction accuracy, confusion matrices, and model-training notebooks are **not part of this JSCDM artefact**. This paper should not claim disease prediction or ML novelty.

## 1. Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Confidential original dataset

Place the supplied Excel workbook in `data/private/` and rename it exactly:

```text
data/private/workplace_survey.xlsx
```

It is ignored by Git. Do not publish it.

## 3. Reproduce the original analysis

```powershell
python run_pipeline.py --config config/workplace_config.yaml
```

Expected verification values:

- PHQ-9 Cronbach's alpha: 0.9192933860
- GAD-7 Cronbach's alpha: 0.9230573096
- PHQ-9 moderate-or-higher: 23.6453201970%
- GAD-7 moderate-or-higher: 19.7044334975%
- Source severity labels match score-derived severity after whitespace/case normalization.

## 4. Independent technical portability demonstration

Download `demographic.csv`, `phq9.csv`, and `gad7.csv` from Zenodo record `10.5281/zenodo.10423537` and place them unchanged in `data/external_raw/`. Inspect the headers, complete `config/zenodo_config_TEMPLATE.yaml`, then make a working copy named `config/zenodo_config.yaml`.

Only dataset paths, the join key, original-to-generic field mapping, and eligible subgroup fields may change. The core modules in `src/` must remain unchanged.

The external student dataset demonstrates technical portability of the pipeline. It does **not** replicate or validate workplace associations, management categories, or generalisability.

## Output files

All generated files are aggregate-only and saved in `outputs/`:

- `schema_validation.json`
- `reliability.csv`
- `severity_distribution.csv`
- `subgroup_results.csv`
- `subgroup_crosstabs_disclosed.csv`
- `disclosure_log.csv`
- `verification_log.csv`
- `severity_distribution.png`
- `run_summary.md`

Never report a subgroup p-value where `chi_square_reportable` is `False`.

## Public repository checklist

Commit: source code, YAML configuration, README, requirements, notebooks, generated aggregate outputs, and documentation.

Do not commit: original workplace data, any respondent-level export, `.venv/`, or local caches.


## Automated safeguard tests

Run the automated failure-mode tests before creating a release:

```powershell
python -m pytest -q


