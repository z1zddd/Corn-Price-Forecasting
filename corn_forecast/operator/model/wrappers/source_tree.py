"""Adapters for optional models loaded from an upstream source checkout."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable, Mapping

import numpy as np

from corn_forecast.pipeline.train.trainer import train_regressor


SourceConfigBuilder = Callable[[int, int, dict], Mapping[str, object]]
SourceForward = Callable[[object, object], object]
SourceModelWrapper = Callable[[object, int], object]


@contextmanager
def _source_import_path(source_root: Path):
    """Temporarily expose an upstream checkout for its relative imports."""

    root = str(source_root)
    sys.path.insert(0, root)
    try:
        yield
    finally:
        try:
            sys.path.remove(root)
        except ValueError:
            pass


def _clear_external_layers() -> None:
    """Avoid reusing a different upstream checkout's top-level ``layers`` package."""

    for name in list(sys.modules):
        if name == "layers" or name.startswith("layers."):
            sys.modules.pop(name, None)


def _load_source_module(source_root: Path, relative_module_path: str, model_name: str) -> ModuleType:
    module_path = source_root / relative_module_path
    if not module_path.is_file():
        raise FileNotFoundError(
            f"{model_name} source module was not found: {module_path}. "
            "Set params.source_root to a compatible upstream checkout."
        )

    digest = hashlib.sha1(str(module_path.resolve()).encode("utf-8")).hexdigest()[:12]
    module_name = f"_corn_forecast_upstream_{model_name}_{digest}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    _clear_external_layers()
    with _source_import_path(source_root):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    return module


class SourceTreeSequenceRegressorAdapter:
    """Train an upstream sequence forecaster on this project's continuous return target."""

    model_family = "external_sequence"
    disabled_by_default = True
    needs_feature_metadata = True

    def __init__(
        self,
        *,
        model_name: str,
        source_url: str,
        relative_module_path: str,
        config_builder: SourceConfigBuilder,
        source_forward: SourceForward,
        params: dict,
        reorder_target_to_last: bool = False,
        model_wrapper: SourceModelWrapper | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                f"torch is required for {model_name}. Install with: pip install -e .[deep]"
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.source_url = source_url
        self.relative_module_path = relative_module_path
        self.config_builder = config_builder
        self.source_forward = source_forward
        self.params = dict(params)
        self.reorder_target_to_last = bool(reorder_target_to_last)
        self.model_wrapper = model_wrapper
        self.epochs = int(self.params.get("epochs", 20))
        self.batch_size = int(self.params.get("batch_size", 32))
        self.lr = float(self.params.get("lr", 0.001))
        self.patience = int(self.params.get("patience", 5))
        self.random_state = int(self.params.get("random_state", 42))
        requested_device = str(self.params.get("device", "cpu"))
        self.device = requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu"
        self.min_train_samples = int(self.params.get("min_train_samples", 30))
        self.model = None
        self.input_shape: tuple[int, int] | None = None
        self.input_indices: list[int] = []
        self.target_feature_index: int | None = None
        self.output_feature_index: int | None = None
        self.target_scale_ = 1.0

    def _source_root(self) -> Path:
        source_root = self.params.get("source_root")
        if not source_root:
            raise ValueError(
                f"{self.model_name} requires params.source_root. Clone its upstream source from "
                f"{self.source_url} and point source_root at that checkout."
            )
        root = Path(str(source_root)).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"{self.model_name} params.source_root is not a directory: {root}. "
                f"Expected an upstream checkout from {self.source_url}."
            )
        return root

    def _configure_feature_layout(self, n_vars: int) -> None:
        feature_cols = self.params.get("feature_cols") or []
        feature_cols = [str(column) for column in feature_cols]
        if feature_cols and len(feature_cols) != n_vars:
            raise ValueError(
                f"{self.model_name} params.feature_cols must match the input feature count "
                f"({len(feature_cols)} != {n_vars})"
            )

        target_feature_index = self.params.get("target_feature_index")
        if target_feature_index is None:
            target_name = self.params.get("target_feature_name") or self.params.get("price_col")
            if target_name and target_name in feature_cols:
                target_feature_index = feature_cols.index(str(target_name))
            elif feature_cols:
                target_feature_index = len(feature_cols) - 1
            else:
                target_feature_index = n_vars - 1
        target_feature_index = int(target_feature_index)
        if not 0 <= target_feature_index < n_vars:
            raise ValueError(
                f"{self.model_name} target_feature_index must be within [0, {n_vars}), got {target_feature_index}"
            )

        self.target_feature_index = target_feature_index
        self.input_indices = list(range(n_vars))
        self.output_feature_index = target_feature_index
        if self.reorder_target_to_last:
            self.input_indices = [idx for idx in self.input_indices if idx != target_feature_index]
            self.input_indices.append(target_feature_index)
            self.output_feature_index = 0

    def _build_model(self, x_train: np.ndarray):
        n_vars = int(x_train.shape[1])
        lookback = int(x_train.shape[2])
        self.input_shape = (n_vars, lookback)
        self._configure_feature_layout(n_vars)
        source_module = _load_source_module(
            self._source_root(), self.relative_module_path, self.model_name
        )
        model_cls = getattr(source_module, "Model", None)
        if model_cls is None:
            raise ImportError(
                f"{self.model_name} source module {self.relative_module_path} does not expose Model"
            )
        config = SimpleNamespace(**dict(self.config_builder(n_vars, lookback, self.params)))
        source_model = model_cls(config)
        if self.model_wrapper is not None:
            return self.model_wrapper(source_model, int(self.target_feature_index or 0))
        return source_model

    def _prediction_tensor(self, model, x):
        if self.input_indices:
            x = x[:, self.input_indices, :]
        upstream_input = x.transpose(1, 2)
        output = self.source_forward(model, upstream_input)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not self.torch.is_tensor(output):
            raise TypeError(f"{self.model_name} returned {type(output).__name__}, expected a torch tensor")
        if output.ndim == 3:
            output_index = int(self.output_feature_index or 0)
            if output_index >= output.shape[-1]:
                raise ValueError(
                    f"{self.model_name} output has {output.shape[-1]} channels, "
                    f"cannot select target channel {output_index}"
                )
            output = output[:, -1, output_index]
        elif output.ndim == 2:
            if output.shape[-1] == 1:
                output = output[:, 0]
            else:
                output_index = int(self.output_feature_index or 0)
                output = output[:, output_index]
        elif output.ndim != 1:
            raise ValueError(f"{self.model_name} returned unsupported output shape {tuple(output.shape)}")
        return output.reshape(-1, 1)

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        return self._fit_regression(x_train, y_train, x_val, y_val)

    def fit_with_targets(
        self,
        x_train,
        y_class_train,
        y_return_train,
        x_val=None,
        y_class_val=None,
        y_return_val=None,
    ):
        return self._fit_regression(x_train, y_return_train, x_val, y_return_val)

    def _fit_regression(self, x_train, y_train, x_val=None, y_val=None):
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=float).reshape(-1)
        if len(x_train) < self.min_train_samples:
            raise ValueError(
                f"{self.model_name} requires at least {self.min_train_samples} train windows, got {len(x_train)}"
            )
        self.torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.model = self._build_model(x_train)
        target_scale = float(np.std(y_train))
        self.target_scale_ = target_scale if target_scale > 1e-12 else 1.0
        self.model = train_regressor(
            self.model,
            x_train,
            y_train,
            x_val=np.asarray(x_val, dtype=np.float32) if x_val is not None else None,
            y_val=np.asarray(y_val, dtype=float) if y_val is not None else None,
            prediction_fn=self._prediction_tensor,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            patience=self.patience,
            device=self.device,
        )
        return self

    def predict_regression(self, x_test) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        x = self.torch.as_tensor(np.asarray(x_test, dtype=np.float32), dtype=self.torch.float32, device=self.device)
        self.model.eval()
        with self.torch.no_grad():
            return self._prediction_tensor(self.model, x).detach().cpu().numpy().reshape(-1)

    def predict_proba(self, x_test) -> np.ndarray:
        raw = self.predict_regression(x_test)
        clipped = np.clip(raw / self.target_scale_, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def predict(self, x_test) -> np.ndarray:
        return (self.predict_regression(x_test) > 0.0).astype(int)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        self.torch.save(
            {
                "model_name": self.model_name,
                "source_url": self.source_url,
                "relative_module_path": self.relative_module_path,
                "state_dict": self.model.state_dict(),
                "params": self.params,
                "input_shape": self.input_shape,
                "target_feature_index": self.target_feature_index,
                "target_scale": self.target_scale_,
            },
            path,
        )
