# External data run — exact steps

1. Copy the three downloaded Zenodo files into `data/external_raw/`:
   - `demographic.csv`
   - `phq9.csv`
   - `gad7.csv`
2. Do not modify any YAML mappings. `config/zenodo_config.yaml` is already completed.
3. Open PowerShell in the project root and run:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --config config\zenodo_config.yaml
```

4. Open `outputs/zenodo/schema_validation.json`. It must show `"passed": true` and `"n_rows": 24292`.
5. Open `outputs/zenodo/reliability.csv` and `outputs/zenodo/severity_distribution.csv`. These are the aggregate values to use in the portability table.

The pipeline intentionally produces only aggregate outputs. It does not publish student-level records.
