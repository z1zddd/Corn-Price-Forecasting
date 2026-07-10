"""XLinear upstream-source adapter."""

from __future__ import annotations

from corn_forecast.operator.model.wrappers.source_tree import SourceTreeSequenceRegressorAdapter


def _xlinear_config(n_vars: int, lookback: int, params: dict) -> dict:
    return {
        "seq_len": lookback,
        "pred_len": 1,
        "d_model": int(params.get("d_model", 64)),
        "enc_in": n_vars,
        "dec_in": n_vars,
        "c_out": 1,
        "t_ff": int(params.get("t_ff", 64)),
        "c_ff": int(params.get("c_ff", n_vars)),
        # The project has already applied a train-only sequence standardizer.
        "usenorm": bool(params.get("use_norm", False)),
        "embed_dropout": float(params.get("embed_dropout", 0.0)),
        "head_dropout": float(params.get("head_dropout", 0.1)),
        "t_dropout": float(params.get("t_dropout", 0.0)),
        "c_dropout": float(params.get("c_dropout", 0.0)),
        # The upstream model calls every non-"M" mode its exogenous variant.
        "features": "MS",
    }


def _forward_xlinear(model, x):
    return model(x)


def create_xlinear(params: dict) -> SourceTreeSequenceRegressorAdapter:
    return SourceTreeSequenceRegressorAdapter(
        model_name="xlinear",
        source_url="https://github.com/Zaiwen/XLinear",
        relative_module_path="models/XLinear.py",
        config_builder=_xlinear_config,
        source_forward=_forward_xlinear,
        params=params,
        # XLinear's MS variant consumes the target variable as the final channel.
        reorder_target_to_last=True,
    )
