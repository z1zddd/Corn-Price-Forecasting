"""Official Chronos-2 adapter.

This adapter uses Amazon's official `chronos-forecasting` package and the
`amazon/chronos-2` pretrained weights. It only converts this project's window
tensors into the pandas dataframe API expected by Chronos-2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.base import BaseModel


class Chronos2OfficialForecaster(BaseModel):
    _PIPELINE_CACHE: dict[tuple[str, str], object] = {}

    def __init__(
        self,
        input_size: int | None = None,
        seq_len: int | None = None,
        pred_len: int = 1,
        close_feature_idx: int = -1,
        model_id: str = "amazon/chronos-2",
        quantile_levels: list[float] | None = None,
        batch_size: int = 16,
        device_map: str | None = None,
    ):
        self.input_size = input_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.close_feature_idx = close_feature_idx
        self.model_id = model_id
        self.quantile_levels = quantile_levels or [0.1, 0.5, 0.9]
        self.batch_size = batch_size
        self.device_map = device_map or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "Chronos2OfficialForecaster":
        x_train = self._to_btv(X_train)
        self.input_size = x_train.shape[-1]
        self.seq_len = x_train.shape[1]
        self.pipeline = self._load_pipeline()
        return self

    def predict(self, X) -> np.ndarray:
        return self.predict_logits(X)

    def predict_logits(self, X) -> np.ndarray:
        """Return forecasted next-month close return as the Platt score."""
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted.")
        x_np = self._to_btv(X)
        close_idx = self.close_feature_idx if self.close_feature_idx >= 0 else x_np.shape[-1] + self.close_feature_idx
        current_close = x_np[:, -1, close_idx].astype("float64")
        scores = []
        for start in range(0, len(x_np), self.batch_size):
            batch = x_np[start : start + self.batch_size]
            pred_close = self._predict_batch(batch, close_idx)
            cur = current_close[start : start + len(pred_close)]
            score = pred_close / np.where(np.abs(cur) < 1e-8, np.nan, cur) - 1.0
            scores.append(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0))
        return np.concatenate(scores, axis=0).reshape(-1).astype("float32")

    def predict_proba(self, X) -> np.ndarray:
        logits = self.predict_logits(X)
        return (1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))).astype("float32")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "adapter": self.__class__.__name__,
                "params": self._params(),
                "note": "Official Chronos-2 weights are loaded through chronos-forecasting from model_id.",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path):
        payload = torch.load(path, map_location="cpu")
        return cls(**payload["params"])

    def _load_pipeline(self):
        cache_key = (self.model_id, self.device_map)
        if cache_key not in self._PIPELINE_CACHE:
            try:
                from chronos import Chronos2Pipeline
            except ImportError as exc:
                raise ImportError(
                    "Chronos-2 official adapter requires `pip install chronos-forecasting>=2.0`."
                ) from exc
            self._PIPELINE_CACHE[cache_key] = Chronos2Pipeline.from_pretrained(self.model_id, device_map=self.device_map)
        return self._PIPELINE_CACHE[cache_key]

    def _predict_batch(self, batch_btv: np.ndarray, close_idx: int) -> np.ndarray:
        context_df = self._batch_to_context_df(batch_btv, close_idx)
        pred_df = self.pipeline.predict_df(
            context_df,
            prediction_length=int(self.pred_len),
            quantile_levels=self.quantile_levels,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
        )
        value_col = "predictions" if "predictions" in pred_df.columns else "0.5"
        last_rows = pred_df.sort_values(["item_id", "timestamp"]).groupby("item_id", sort=False).tail(1)
        last_rows = last_rows.set_index("item_id").reindex([f"sample_{i}" for i in range(len(batch_btv))])
        return last_rows[value_col].to_numpy(dtype="float64")

    def _batch_to_context_df(self, batch_btv: np.ndarray, close_idx: int) -> pd.DataFrame:
        bsz, seq_len, n_vars = batch_btv.shape
        timestamps = pd.date_range("2000-01-01", periods=seq_len, freq="MS")
        covariate_indices = [idx for idx in range(n_vars) if idx != close_idx]
        cov_values = batch_btv[:, :, covariate_indices].reshape(bsz * seq_len, len(covariate_indices))
        data = {
            "item_id": np.repeat([f"sample_{sample_id}" for sample_id in range(bsz)], seq_len),
            "timestamp": np.tile(timestamps, bsz),
            "target": batch_btv[:, :, close_idx].reshape(-1),
        }
        data.update({f"cov_{cov_id:03d}": cov_values[:, cov_id] for cov_id in range(cov_values.shape[1])})
        return pd.DataFrame(data)

    def _params(self) -> dict:
        return {
            "input_size": self.input_size,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "close_feature_idx": self.close_feature_idx,
            "model_id": self.model_id,
            "quantile_levels": self.quantile_levels,
            "batch_size": self.batch_size,
            "device_map": self.device_map,
        }

    @staticmethod
    def _to_btv(X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected X with shape [N, V, T], got {arr.shape}.")
        return np.transpose(arr, (0, 2, 1)).astype("float32")
