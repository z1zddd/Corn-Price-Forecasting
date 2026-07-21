# GitHub Upload Scope Review

Status: proposal only; nothing has been staged, committed, or uploaded.

| Scope | Rows | Files | Bytes |
| --- | ---: | ---: | ---: |
| experiment artifact | 108 | 9180 | 4492232873 |
| reproducibility source | 7 | 48 | 200040 |
| catalog summary | 1 | 9 | 1182975 |
| repository metadata | 1 | 2 | 0 |

## Storage

- Git LFS: 1728 checkpoint files, 3948533032 bytes.
- Normal Git: 545082856 bytes before Git compression.
- Largest individual file: 40,264,095 bytes. No artifact is 50 MB or larger.
- Checkpoints still require the proposed LFS plan because 1,728 binary joblib files total about 3.95 GB.
- Excluded: artifacts.tar.gz, all ZIP bundles, work/, __pycache__, *.pyc, stale RUNNING/failed directories, credentials, PID files.

## Model Count

- Catalog models: 18.
- Python model-code files: 21 (18 entrypoints plus 3 shared adapter/network files).
- Catalog tasks: 36 (2 datasets x 18 models).

See `github_upload_scope.csv` for every run-level target and `archive_artifact_inventory.csv` for all 9,180 artifact file paths.
