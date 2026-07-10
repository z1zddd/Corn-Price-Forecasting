# Graph Temporal Model Operators

Complete graph-temporal forecasting model wrappers live here, one model per
file. Each wrapper exposes:

- `fit_with_targets`
- `predict_proba`
- `predict_regression`
- `predict`
- `save`
- `save_lightweight`

The model cores are imported from mature upstream implementations at runtime:
Torch Spatiotemporal (`tsl`) for DCRNN, GraphWaveNet, AGCRN, and GRUGCN; PyTorch
Geometric Temporal for MTGNN, STGCN, ASTGCN, and MSTGCN.

Expected input window shape is `[n_samples, n_nodes, lookback]`.
