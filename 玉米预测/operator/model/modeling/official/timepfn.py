"""Official TimePFN checkpoint adapter.

Uses the official implementation downloaded under `third_party/timepfn_official`.
The adapter does not reimplement the TimePFN architecture. It only converts this
project's window tensors to the official model input and converts the one-step
forecast for the endogenous close series into a continuous spike score.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.models.base import BaseModel


class TimePFNOfficialForecaster(BaseModel):
    def __init__(
        self,
        input_size: int | None = None,
        seq_len: int | None = None,
        pred_len: int = 1,
        close_feature_idx: int = -1,
        official_root: str = "third_party/timepfn_official",
        checkpoint_path: str = "checkpoints/TimePFN/checkpoint.pth",
        patch_size: int = 16,
        embed_dim: int = 256,
        d_model: int = 1024,
        d_ff: int = 512,
        e_layers: int = 8,
        n_heads: int = 8,
        factor: int = 5,
        dropout: float = 0.1,
        activation: str = "gelu",
        class_strategy: str = "projection",
        use_norm: bool = True,
        batch_size: int = 16,
        device: str | None = None,
    ):
        self.input_size = input_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.close_feature_idx = close_feature_idx
        self.official_root = official_root
        self.checkpoint_path = checkpoint_path
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.n_heads = n_heads
        self.factor = factor
        self.dropout = dropout
        self.activation = activation
        self.class_strategy = class_strategy
        self.use_norm = use_norm
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "TimePFNOfficialForecaster":
        x_train = self._to_btv(X_train)
        self.input_size = x_train.shape[-1]
        self.seq_len = x_train.shape[1]
        self.model = self._load_official_model().to(self.device)
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
        x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
        scores = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                out = self.model(x[start : start + self.batch_size], None, None, None)
                pred_close = out[:, 0, close_idx].detach().cpu().numpy().astype("float64")
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
        payload = {
            "adapter": self.__class__.__name__,
            "params": self._params(),
            "note": "Official TimePFN checkpoint is loaded from official_root/checkpoint_path.",
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path):
        payload = torch.load(path, map_location="cpu")
        return cls(**payload["params"])

    def _load_official_model(self):
        root = Path(self.official_root).resolve()
        checkpoint = root / self.checkpoint_path
        if not checkpoint.exists():
            raise FileNotFoundError(f"TimePFN checkpoint not found: {checkpoint}")
        if not root.exists():
            raise FileNotFoundError(f"TimePFN official root not found: {root}")

        cfg = SimpleNamespace(
            seq_len=int(self.seq_len),
            pred_len=int(self.pred_len),
            output_attention=False,
            use_norm=bool(self.use_norm),
            patch_size=int(self.patch_size),
            embed_dim=int(self.embed_dim),
            class_strategy=self.class_strategy,
            factor=int(self.factor),
            dropout=float(self.dropout),
            n_heads=int(self.n_heads),
            d_ff=int(self.d_ff),
            activation=self.activation,
            e_layers=int(self.e_layers),
            d_model=int(self.d_model),
        )
        with official_import_context(root):
            with contextlib.redirect_stdout(io.StringIO()):
                module = importlib.import_module("model.TimePFN_wrapper")
                model = module.Model(cfg)
            state = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(state)
        return model

    def _params(self) -> dict:
        return {
            "input_size": self.input_size,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "close_feature_idx": self.close_feature_idx,
            "official_root": self.official_root,
            "checkpoint_path": self.checkpoint_path,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "e_layers": self.e_layers,
            "n_heads": self.n_heads,
            "factor": self.factor,
            "dropout": self.dropout,
            "activation": self.activation,
            "class_strategy": self.class_strategy,
            "use_norm": self.use_norm,
            "batch_size": self.batch_size,
            "device": self.device,
        }

    @staticmethod
    def _to_btv(X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected X with shape [N, V, T], got {arr.shape}.")
        return np.transpose(arr, (0, 2, 1)).astype("float32")


@contextlib.contextmanager
def official_import_context(root: Path):
    """Import official TimePFN modules without using this project's same-name packages."""
    stub = types.ModuleType("reformer_pytorch")

    class LSHSelfAttention(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("reformer_pytorch is not installed; LSHSelfAttention is unused by TimePFN.")

    stub.LSHSelfAttention = LSHSelfAttention
    old_modules = {
        key: sys.modules[key]
        for key in list(sys.modules)
        if key == "layers"
        or key.startswith("layers.")
        or key == "model"
        or key.startswith("model.")
        or key == "utils"
        or key.startswith("utils.")
        or key == "reformer_pytorch"
    }
    for key in old_modules:
        sys.modules.pop(key, None)
    sys.modules["reformer_pytorch"] = stub
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass
        for key in list(sys.modules):
            if (
                key == "layers"
                or key.startswith("layers.")
                or key == "model"
                or key.startswith("model.")
                or key == "utils"
                or key.startswith("utils.")
                or key == "reformer_pytorch"
            ):
                sys.modules.pop(key, None)
        sys.modules.update(old_modules)
