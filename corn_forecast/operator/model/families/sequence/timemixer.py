"""TimeMixer upstream-source adapter."""

from __future__ import annotations

from corn_forecast.operator.model.wrappers.source_tree import SourceTreeSequenceRegressorAdapter


def _timemixer_config(n_vars: int, lookback: int, params: dict) -> dict:
    down_sampling_window = int(params.get("down_sampling_window", 2))
    max_layers = 0
    while lookback // (down_sampling_window ** (max_layers + 1)) >= 1:
        max_layers += 1
    down_sampling_layers = min(int(params.get("down_sampling_layers", 1)), max_layers)
    return {
        "task_name": "long_term_forecast",
        "seq_len": lookback,
        "label_len": 0,
        "pred_len": 1,
        "enc_in": n_vars,
        "dec_in": n_vars,
        "c_out": n_vars,
        "d_model": int(params.get("d_model", 32)),
        "d_ff": int(params.get("d_ff", 64)),
        "e_layers": int(params.get("e_layers", 1)),
        "dropout": float(params.get("dropout", 0.1)),
        "down_sampling_window": down_sampling_window,
        "down_sampling_layers": down_sampling_layers,
        "down_sampling_method": str(params.get("down_sampling_method", "avg")),
        "channel_independence": int(params.get("channel_independence", 1)),
        "decomp_method": str(params.get("decomp_method", "moving_avg")),
        "moving_avg": int(params.get("moving_avg", 3)),
        "top_k": int(params.get("top_k", 5)),
        "use_future_temporal_feature": False,
        # The project has already applied a train-only sequence standardizer.
        "use_norm": int(params.get("use_norm", 0)),
        "embed": str(params.get("embed", "timeF")),
        "freq": str(params.get("freq", "m")),
        "num_class": 2,
    }


def _forward_timemixer(model, x):
    return model(x, None, None, None)


def create_timemixer(params: dict) -> SourceTreeSequenceRegressorAdapter:
    return SourceTreeSequenceRegressorAdapter(
        model_name="timemixer",
        source_url="https://github.com/kwuking/TimeMixer",
        relative_module_path="models/TimeMixer.py",
        config_builder=_timemixer_config,
        source_forward=_forward_timemixer,
        params=params,
    )
