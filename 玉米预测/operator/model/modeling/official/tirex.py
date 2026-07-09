"""Official TiRex adapter.

TiRex-1 is an official zero-shot univariate forecasting model. This adapter
uses the official `tirex-ts` package and converts its one-step close forecast
into the continuous score used by the shared Platt-calibrated classification
backtest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.base import BaseModel


class TiRexOfficialForecaster(BaseModel):
    _MODEL_CACHE: dict[tuple[str, str], object] = {}

    def __init__(
        self,
        input_size: int | None = None,
        seq_len: int | None = None,
        pred_len: int = 1,
        close_feature_idx: int = 0,
        model_id: str = "third_party/tirex_official/NX-AI_TiRex",
        backend: str = "torch",
        batch_size: int = 64,
        device: str | None = None,
    ):
        self.input_size = input_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.close_feature_idx = close_feature_idx
        self.model_id = model_id
        self.backend = backend
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "TiRexOfficialForecaster":
        x_train = self._to_btv(X_train)
        self.input_size = x_train.shape[-1]
        self.seq_len = x_train.shape[1]
        self.model = self._load_model()
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self

    def predict(self, X) -> np.ndarray:
        return self.predict_logits(X)

    def predict_logits(self, X) -> np.ndarray:
        """Return forecasted next-month close return as the Platt score."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        x_np = self._to_btv(X)
        close_idx = self.close_feature_idx if self.close_feature_idx >= 0 else x_np.shape[-1] + self.close_feature_idx
        current_close = x_np[:, -1, close_idx].astype("float64")
        scores = []
        with torch.no_grad():
            for start in range(0, len(x_np), self.batch_size):
                batch = x_np[start : start + self.batch_size, :, close_idx]
                context = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
                _, mean = self.model.forecast(context=context, prediction_length=int(self.pred_len))
                pred = mean.detach().cpu().numpy()
                if pred.ndim == 3:
                    pred = pred[..., 0]
                if pred.ndim == 1:
                    pred_close = pred.astype("float64")
                else:
                    pred_close = pred[:, 0].astype("float64")
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
                "note": "Official TiRex weights are loaded from model_id.",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path):
        payload = torch.load(path, map_location="cpu")
        return cls(**payload["params"])

    def _load_model(self):
        cache_key = (self.model_id, self.backend, self.device)
        if cache_key not in self._MODEL_CACHE:
            try:
                from tirex import load_model
                from tirex.base import PretrainedModel
            except ImportError as exc:
                raise ImportError("TiRex official adapter requires `pip install tirex-ts`.") from exc
            local_path = Path(self.model_id)
            if local_path.exists():
                checkpoint_path = local_path / "model.ckpt" if local_path.is_dir() else local_path
                model_cls = PretrainedModel.REGISTRY["TiRex"]
                model = model_cls.from_pretrained(
                    str(checkpoint_path),
                    backend=self.backend,
                    device=self.device,
                )
            else:
                model = load_model(self.model_id, backend=self.backend, device=self.device)
            self._MODEL_CACHE[cache_key] = model
        return self._MODEL_CACHE[cache_key]

    def _params(self) -> dict:
        return {
            "input_size": self.input_size,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "close_feature_idx": self.close_feature_idx,
            "model_id": self.model_id,
            "backend": self.backend,
            "batch_size": self.batch_size,
            "device": self.device,
        }

    @staticmethod
    def _to_btv(X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected X with shape [N, V, T], got {arr.shape}.")
        return np.transpose(arr, (0, 2, 1)).astype("float32")
