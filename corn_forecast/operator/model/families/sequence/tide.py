"""TiDE upstream-source adapter."""

from __future__ import annotations

from corn_forecast.operator.model.wrappers.source_tree import SourceTreeSequenceRegressorAdapter


def _tide_config(n_vars: int, lookback: int, params: dict) -> dict:
    return {
        "task_name": "long_term_forecast",
        "seq_len": lookback,
        "label_len": 0,
        "pred_len": 1,
        "enc_in": n_vars,
        "dec_in": n_vars,
        "c_out": n_vars,
        "d_model": int(params.get("d_model", 16)),
        "d_ff": int(params.get("d_ff", 64)),
        "e_layers": int(params.get("e_layers", 1)),
        "d_layers": int(params.get("d_layers", 1)),
        "dropout": float(params.get("dropout", 0.1)),
        "freq": str(params.get("freq", "m")),
    }


def _wrap_tide(source_model, target_feature_index: int):
    import torch
    from torch import nn

    class TiDEReturnHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.source_model = source_model
            self.target_feature_index = target_feature_index
            self.return_head = nn.Linear(2, 1)

        def forward(self, x):
            forecast = self.source_model(x, None, None, None)
            forecast_target = forecast[:, -1, self.target_feature_index]
            latest_target = x[:, -1, self.target_feature_index]
            return self.return_head(torch.stack([forecast_target, latest_target], dim=-1))

    return TiDEReturnHead()


def _forward_tide(model, x):
    return model(x)


def create_tide(params: dict) -> SourceTreeSequenceRegressorAdapter:
    return SourceTreeSequenceRegressorAdapter(
        model_name="tide",
        source_url="https://github.com/thuml/Time-Series-Library",
        relative_module_path="models/TiDE.py",
        config_builder=_tide_config,
        source_forward=_forward_tide,
        params=params,
        model_wrapper=_wrap_tide,
    )
