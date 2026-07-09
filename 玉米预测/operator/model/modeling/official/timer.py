"""Official Timer adapter.

Uses THUML's official HuggingFace remote-code model. Timer-base requires at
least one 96-point token as context, so the 12 observed monthly closes are
left-padded to 96 with the earliest observed close before calling `generate`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.base import BaseModel


class TimerOfficialForecaster(BaseModel):
    _MODEL_CACHE: dict[tuple[str, str], object] = {}

    def __init__(
        self,
        input_size: int | None = None,
        seq_len: int | None = None,
        pred_len: int = 1,
        close_feature_idx: int = 0,
        model_id: str = "third_party/timer_official/thuml_timer-base-84m",
        input_token_len: int = 96,
        generation_length: int = 96,
        pad_strategy: str = "repeat_first",
        batch_size: int = 16,
        device: str | None = None,
    ):
        self.input_size = input_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.close_feature_idx = close_feature_idx
        self.model_id = model_id
        self.input_token_len = input_token_len
        self.generation_length = generation_length
        self.pad_strategy = pad_strategy
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "TimerOfficialForecaster":
        x_train = self._to_btv(X_train)
        self.input_size = x_train.shape[-1]
        self.seq_len = x_train.shape[1]
        self.model = self._load_model()
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
        close_context = x_np[:, :, close_idx].astype("float32")
        current_close = close_context[:, -1].astype("float64")
        scores = []
        with torch.no_grad():
            for start in range(0, len(close_context), self.batch_size):
                batch = self._pad_context(close_context[start : start + self.batch_size])
                context = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
                generated = self.model.generate(context, max_new_tokens=int(self.generation_length))
                if generated.shape[1] >= context.shape[1] + int(self.pred_len):
                    forecast = generated[:, context.shape[1] : context.shape[1] + int(self.pred_len)]
                else:
                    forecast = generated[:, : int(self.pred_len)]
                pred_close = forecast[:, 0].detach().cpu().numpy().astype("float64")
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
                "note": "Official Timer weights are loaded from model_id.",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path):
        payload = torch.load(path, map_location="cpu")
        return cls(**payload["params"])

    def _load_model(self):
        cache_key = (self.model_id, self.device)
        if cache_key not in self._MODEL_CACHE:
            try:
                from transformers import AutoModelForCausalLM
            except ImportError as exc:
                raise ImportError("Timer official adapter requires `transformers`.") from exc
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                local_files_only=Path(self.model_id).exists(),
            )
            model = model.to(self.device)
            self._MODEL_CACHE[cache_key] = model
        return self._MODEL_CACHE[cache_key]

    def _pad_context(self, close_context: np.ndarray) -> np.ndarray:
        if close_context.shape[1] >= self.input_token_len:
            usable_len = (close_context.shape[1] // self.input_token_len) * self.input_token_len
            return close_context[:, -usable_len:].astype("float32")
        pad_len = self.input_token_len - close_context.shape[1]
        if self.pad_strategy != "repeat_first":
            raise ValueError(f"Unsupported pad_strategy={self.pad_strategy!r}.")
        pad = np.repeat(close_context[:, :1], pad_len, axis=1)
        return np.concatenate([pad, close_context], axis=1).astype("float32")

    def _params(self) -> dict:
        return {
            "input_size": self.input_size,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "close_feature_idx": self.close_feature_idx,
            "model_id": self.model_id,
            "input_token_len": self.input_token_len,
            "generation_length": self.generation_length,
            "pad_strategy": self.pad_strategy,
            "batch_size": self.batch_size,
            "device": self.device,
        }

    @staticmethod
    def _to_btv(X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected X with shape [N, V, T], got {arr.shape}.")
        return np.transpose(arr, (0, 2, 1)).astype("float32")
