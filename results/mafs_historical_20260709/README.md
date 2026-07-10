# MAFS Historical Experiment Record

This directory records the provenance of the material supplied in
`/Users/keyizhan/Downloads/agent/`. It is not an official result and is not
included in any deployment ensemble or leaderboard.

## What Was Supplied

- `同质版agent.py` and `异质版agent.py`: MAFS experiment launchers.
- `TiDEagent.py` and `timemixer agent.py`: fixed-split single-backbone tuning
  launchers.
- `run_mafs_simpletm_*.py`: MAFS plus SimpleTM experiment launchers.
- Markdown reports with results from 2026-07-09.

The source directory does not contain the MAFS model implementation. It
expects an external `MAFS-main` checkout, invokes its `run.py`, and evaluates a
fixed chronological 70/10/20 split. Its runner code therefore does not satisfy
this repository's reusable `operator` contract or its walk-forward backtest
contract.

## Recorded Findings

The supplied reports describe TimeMixer and TiDE as relatively balanced
single-backbone candidates, and the homogeneous MAFS setup as stronger than the
heterogeneous setup on the reported small monthly sample. These are historical
observations only. They use a short fixed test segment, mostly single-seed
runs, price-first direction conversion, and no formal validation-only threshold
calibration.

## Repository Decision

The reusable individual backbones are represented by `simpletm`, `timemixer`,
`tide`, and `xlinear` adapters under
`corn_forecast/operator/model/families/sequence/`. MAFS itself is intentionally
not exposed as a model operator until its full source is available and its
multi-agent fit/predict logic can be adapted to train-only expanding windows.
