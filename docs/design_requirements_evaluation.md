# Design Requirements Evaluation

| Design requirement | Implemented mechanism | Observable evidence |
|---|---|---|
| Reproducibility | YAML configuration files, pinned dependencies, executable run script, and notebooks | Original 406-record results reproduced through six verification checks |
| Input integrity | Schema validation checks required columns, valid ordinal values, completeness, expected record count, and duplicate limits | `schema_validation.json`; automated failure-mode tests |
| Statistical discipline | Expected-cell eligibility gate, Pearson chi-square only for reportable tables, Benjamini–Hochberg adjustment only for eligible tests | `subgroup_results.csv`; automated expected-count failure test |
| Disclosure control | Minimum-cell suppression before release plus disclosure log | `disclosure_log.csv`; suppression test; sensitivity analysis at k = 3, 5, and 10 |
| Aggregate-only boundary | No participant-level output is written to generated output folders | Repository structure, `.gitignore`, README, and aggregate-only manifests |
| Technical portability | Dataset-specific field mapping in YAML with unchanged core modules | Zenodo PHQ-9/GAD-7 run on 24,292 student records |
| Traceability | Run manifest records Git commit, configuration hash, environment, record count, and output hashes | `run_manifest.json` for workplace and Zenodo runs |
| Reporting-rule robustness | Configuration-driven sensitivity analysis for disclosure and Cramér’s V thresholds | Sensitivity outputs show the same leading subgroup factor across tested settings |