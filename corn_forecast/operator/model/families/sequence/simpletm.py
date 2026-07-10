"""SimpleTM upstream-source adapter."""

from __future__ import annotations

from corn_forecast.operator.model.wrappers.source_tree import SourceTreeSequenceRegressorAdapter


def _simpletm_config(n_vars: int, lookback: int, params: dict) -> dict:
    return {
        "seq_len": lookback,
        "pred_len": 1,
        "d_model": int(params.get("d_model", 32)),
        "d_ff": int(params.get("d_ff", 64)),
        "e_layers": int(params.get("e_layers", 1)),
        "dropout": float(params.get("dropout", 0.1)),
        "factor": int(params.get("factor", 1)),
        "output_attention": False,
        # The project has already applied a train-only sequence standardizer.
        "use_norm": bool(params.get("use_norm", False)),
        "geomattn_dropout": float(params.get("geomattn_dropout", 0.1)),
        "alpha": float(params.get("alpha", 0.5)),
        "kernel_size": params.get("kernel_size"),
        "requires_grad": bool(params.get("requires_grad", True)),
        "wv": str(params.get("wv", "db1")),
        "m": int(params.get("m", 2)),
        "dec_in": n_vars,
        "embed": str(params.get("embed", "fixed")),
        "freq": str(params.get("freq", "m")),
        "activation": str(params.get("activation", "gelu")),
    }


def _forward_simpletm(model, x):
    return model(x, None, None, None)


def create_simpletm(params: dict) -> SourceTreeSequenceRegressorAdapter:
    return SourceTreeSequenceRegressorAdapter(
        model_name="simpletm",
        source_url="https://github.com/vsingh-group/SimpleTM",
        relative_module_path="model/SimpleTM.py",
        config_builder=_simpletm_config,
        source_forward=_forward_simpletm,
        params=params,
    )
