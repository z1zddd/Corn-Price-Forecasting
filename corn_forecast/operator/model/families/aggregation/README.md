# Aggregation Operators

This package contains model operators that aggregate already generated
prediction streams.

`deployment_vote.py` implements the fixed deployment vote aggregator registered
as `type: deployment_ensemble` for backward-compatible configs. It consumes
prediction-library files such as `all_rolling_predictions.csv`; it does not
train a base model from raw features.

The stored `candidate_weights` are normalized forward-replacement selection
frequencies. The metadata uses
`search_protocol=full_history_deployment_discovery`, which identifies a fixed
deployment candidate from a completed prediction library. It must not be
described as strict no-leakage walk-forward model selection.
