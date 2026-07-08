---
name: forecast-framework-preflight
description: Preflight workflow for commodity and short-horizon time-series forecasting framework runs. Use before running, rerunning, modifying, or validating rolling backtests, spike labels, price-direction labels, new CSV data, news/PCA feature variants, ensembles, or best-model reproduction tasks. Forces user confirmation, data quality diagnosis, label audit, feature leakage audit, low-freedom framework mapping, and a run/no-run gate before training.
---

# Forecast Framework Preflight

Use this skill as a gate before any forecasting or backtest run in the commodity forecasting framework. The goal is to keep agents honest: confirm the experiment contract, diagnose the data, verify labels and leakage, map the request to the existing architecture with low freedom, and only then run training or evaluation.

## Hard Rule

Do not start model training, hyperparameter search, ensemble construction, threshold selection, or report generation until the preflight gate is satisfied. If the user asks only to discuss or plan, stay read-only and do not write files.

If the user already supplied every required choice, restate the contract and proceed to the audits. If material choices are missing or contradictory, ask concise questions before running.

## Required Sequence

1. Read the framework surface that will be used: configuration schema, CLI entry point, data loader, target builder, windowing, rolling split engine, model registry, metrics, and report writer. Prefer repository APIs and commands over custom scripts.

2. Confirm the experiment contract with the user. Cover:
   - input file path and data frequency
   - forecast target and label source: existing `spike`, price-derived direction, return regression, or another explicit target
   - horizon values, lookback values, test period, validation length, and rolling split mode
   - feature variants: no news, raw news, precomputed PCA news, or newly fitted PCA
   - model families and whether the run is formal, smoke, diagnostic, or reproduction
   - threshold policy, ensemble policy, primary metrics, and abnormality checks such as R2
   - compute location, GPU/CPU limits, allowed parallelism, output directory, and whether existing processes must be left alone

3. Run a data quality diagnosis before modeling:
   - report row and column counts, date range, inferred frequency, duplicate dates, sort order, and gaps
   - identify price columns, target columns, PCA/news columns, and obviously derived future columns
   - summarize missingness by rows, columns, and critical feature groups
   - check feature count versus training rows for each requested lookback/horizon
   - for daily-to-monthly tasks, state the aggregation rule for each feature group and compare overlapping monthly prices against any trusted monthly file when available

4. Audit labels explicitly:
   - compute the requested horizon labels from the stated price column when price-derived labels are requested
   - compare existing labels such as `spike` against price-derived labels for h1 and h2 when both are present
   - report agreement rate, class balance, number of positives/negatives, constant-label folds, and unusable folds
   - verify each training row only uses labels whose target date is known before the validation or test month
   - do not silently change from existing `spike` to price-derived direction, or the reverse; require an explicit contract

5. Audit leakage features:
   - flag columns containing target-like or lead information, including names with `future`, `next`, `lead`, `fwd`, `forward`, `target`, `label`, `spike`, `return_+h`, horizon-specific future returns, and post-event trend labels
   - do not drop legitimate market words such as `futures_price` merely because they contain "future"; inspect context and column meaning
   - exclude target/leakage columns from features through the framework's feature selection mechanism, not by ad hoc edits hidden from the config
   - verify scaling, PCA fitting, threshold selection, and model selection are fit only on training/validation data allowed by the rolling split

6. Check architecture fit:
   - map every requested choice to config fields, registry model names, CLI commands, and output files
   - classify any part that is outside the framework as a gap before writing code
   - keep new code minimal and additive: prefer configs, registry entries, or small framework extensions over standalone replacement pipelines
   - label results accurately: a smoke run is not a formal reproduction, and aggregating old predictions is not an end-to-end retrain

7. Decide the run gate:
   - `RUN`: data, labels, leakage, and architecture mapping are acceptable
   - `RUN_WITH_LIMITS`: acceptable only as smoke or diagnostic; name the limits
   - `STOP`: labels, leakage, missing data, ambiguity, or framework mismatch makes the run unreliable

## Low-Freedom Implementation Rules

Use the repository's existing CLI, config files, model registry, rolling split engine, scaler, metrics, and report writer. Do not hand-roll rolling backtests or metric calculations unless the framework lacks the needed capability and the gap is stated first.

Do not tune thresholds, choose models, fit scalers, fit PCA, or select ensembles on test months. Validation windows may be used only when the framework contract explicitly allows them.

When adding a model, use an official package or a clearly identified repository implementation when possible. If a model is a local approximation, name it as such and do not call it official or SOTA.

Preserve reproducibility: save the config, command, git commit or source version, input fingerprint, feature manifest, split summary, prediction file, metrics file, and audit notes.

## Evidence Rules

State metrics only after reading actual output files or command output. Cite the file path or command that produced the number.

Do not infer that a model ran because it appears in a planned list. Distinguish completed, failed, skipped, invalid, and incomplete combinations.

Do not hide abnormal diagnostics. For regression heads, report inverse-scaling behavior and R2; treat R2 below -0.1 as abnormal and R2 below 0 as suspicious unless explained by the target design.

If news/PCA features are requested, record whether PCA is precomputed in the input or fitted inside the rolling training loop. Precomputed PCA must have provenance; newly fitted PCA must be fit without future data.

## Required Output Shape

Before training, produce a concise preflight report with:
   - confirmed contract
   - data quality summary
   - label audit summary
   - leakage audit summary
   - architecture fit and gaps
   - run gate decision and exact next command or config change

After training, produce:
   - completed combinations and missing/invalid combinations
   - best metrics by horizon and feature variant, including AUC, AP, balanced accuracy, and R2 where applicable
   - the exact output directory and files written
   - a clear statement of whether the result is formal, smoke, diagnostic, or reproduction
