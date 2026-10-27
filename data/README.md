Source matrices and prediction arrays are intentionally excluded from version control and the source archive.

Run `python scripts/download_data.py` from the repository root after installation. Files are downloaded from the official Open Problems public bucket and checked against recorded SHA-256 hashes. `python scripts/run_study.py` also downloads missing inputs automatically.

`python scripts/make_report.py` downloads fixed Reactome release 93 when needed. The new code's MIT license does not replace the original data terms; see `DATACARD.md` and `REFERENCES.md`.
