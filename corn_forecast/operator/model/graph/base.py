"""Shared adapter for complete official graph-temporal models."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from corn_forecast.operator.layer.graph.adjacency import build_adjacency, graph_info
from corn_forecast.operator.model.graph.utils import as_windows, sigmoid


class OfficialGraphForecastAdapter:
    """Fit/predict adapter around mature graph-temporal model cores.

    Subclasses own only the official model construction and forward signature.
    This base class owns framework-facing targets, graph building, batching,
    losses, probability conversion, and persistence.
    """

    architecture: str = "graph_temporal"

    def __init__(
        self,
        *,
        graph_builder: str = "corr_abs_top5",
        hidden_size: int = 16,
        ff_size: int = 64,
        n_layers: int = 1,
        kernel_size: int = 2,
        temporal_kernel_size: int = 2,
        spatial_kernel_size: int = 2,
        dropout: float = 0.0,
        epochs: int = 30,
        batch_size: int = 16,
        lr: float = 0.001,
        patience: int = 5,
        random_state: int = 42,
        device: str = "cuda",
        objective: str = "return_mse",
        bce_weight: float = 0.5,
        logit_scale: float = 1.0,
        class_weight: str | None = None,
        feature_cols: list[str] | None = None,
    ) -> None:
        if objective not in {"return_mse", "return_sign_bce", "direction_bce"}:
            raise ValueError(f"Unknown graph model objective: {objective}")
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for graph temporal models. Install with: pip install -e .[deep]") from exc

        self.torch = torch
        self.graph_builder = graph_builder
        self.hidden_size = int(hidden_size)
        self.ff_size = int(ff_size)
        self.n_layers = int(n_layers)
        self.kernel_size = int(kernel_size)
        self.temporal_kernel_size = int(temporal_kernel_size)
        self.spatial_kernel_size = int(spatial_kernel_size)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.patience = int(patience)
        self.random_state = int(random_state)
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.objective = objective
        self.bce_weight = float(bce_weight)
        self.logit_scale = float(logit_scale) if float(logit_scale) > 1e-8 else 1.0
        self.class_weight = class_weight
        self.feature_cols = list(feature_cols or [])

        self.model = None
        self.edge_index_ = None
        self.edge_weight_ = None
        self.adjacency_tensor_ = None
        self.adjacency_: np.ndarray | None = None
        self.graph_info_: dict[str, float | int] = {}
        self.return_mean_ = 0.0
        self.return_std_ = 1.0
        self.zero_return_scaled_ = 0.0
        self.input_shape_: tuple[int, int] | None = None
        self.model_family = "official_graph_temporal"
        self.disabled_by_default = True

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        return self.fit_with_targets(x_train, y_train, np.asarray(y_train, dtype=float), x_val, y_val, None)

    def fit_with_targets(self, x_train, y_class_train, y_return_train, x_val=None, y_class_val=None, y_return_val=None):
        torch = self.torch
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        x_train = as_windows(x_train, name="x_train")
        y_class = np.asarray(y_class_train, dtype=float).reshape(-1)
        y_return = np.asarray(y_return_train, dtype=float).reshape(-1)
        if x_train.shape[0] != y_return.shape[0] or x_train.shape[0] != y_class.shape[0]:
            raise ValueError("x_train, y_class_train, and y_return_train have inconsistent lengths")

        self.input_shape_ = (int(x_train.shape[1]), int(x_train.shape[2]))
        self.adjacency_ = self._build_adjacency(x_train)
        self.edge_index_, self.edge_weight_ = self._edge_tensors(self.adjacency_)
        self.adjacency_tensor_ = torch.as_tensor(self.adjacency_, dtype=torch.float32, device=self.device)

        self.return_mean_ = float(np.nanmean(y_return))
        self.return_std_ = float(np.nanstd(y_return))
        if not np.isfinite(self.return_std_) or self.return_std_ < 1e-8:
            self.return_std_ = 1.0
        self.zero_return_scaled_ = (0.0 - self.return_mean_) / self.return_std_
        y_scaled = (y_return - self.return_mean_) / self.return_std_

        model = self._build_model(n_nodes=x_train.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        mse_loss = torch.nn.MSELoss()
        bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=self._pos_weight_tensor(y_class))
        train_x = self._to_graph_tensor(x_train)
        train_return_y = self._target_tensor(y_scaled, n_nodes=x_train.shape[1])
        train_class_y = self._target_tensor(y_class, n_nodes=x_train.shape[1])
        val_x = val_return_y = val_class_y = None
        if x_val is not None and y_return_val is not None and len(y_return_val):
            x_val_arr = as_windows(x_val, name="x_val")
            y_val_arr = (np.asarray(y_return_val, dtype=float).reshape(-1) - self.return_mean_) / self.return_std_
            y_val_class_arr = np.asarray(y_class_val, dtype=float).reshape(-1)
            val_x = self._to_graph_tensor(x_val_arr)
            val_return_y = self._target_tensor(y_val_arr, n_nodes=x_train.shape[1])
            val_class_y = self._target_tensor(y_val_class_arr, n_nodes=x_train.shape[1])

        best_state = None
        best_loss = float("inf")
        stale_epochs = 0
        for epoch in range(max(1, self.epochs)):
            model.train()
            order = np.random.default_rng(self.random_state + epoch).permutation(train_x.shape[0])
            for start in range(0, train_x.shape[0], self.batch_size):
                batch_idx = order[start : start + self.batch_size]
                if self._requires_full_batch() and len(batch_idx) != self.batch_size:
                    continue
                xb = train_x[batch_idx]
                optimizer.zero_grad(set_to_none=True)
                pred = self._forward_model(model, xb)
                loss = self._objective_loss(
                    pred,
                    train_return_y[batch_idx],
                    train_class_y[batch_idx],
                    mse_loss,
                    bce_loss,
                )
                loss.backward()
                optimizer.step()

            eval_x = val_x if val_x is not None else train_x
            eval_return_y = val_return_y if val_return_y is not None else train_return_y
            eval_class_y = val_class_y if val_class_y is not None else train_class_y
            current_loss = self._eval_loss(model, eval_x, eval_return_y, eval_class_y, mse_loss, bce_loss)
            if current_loss + 1e-8 < best_loss:
                best_loss = current_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if self.patience > 0 and stale_epochs >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        return self

    def predict_regression(self, x_test) -> np.ndarray | None:
        if self.objective == "direction_bce":
            return None
        scaled = self._predict_scaled_output(x_test)
        return scaled * self.return_std_ + self.return_mean_

    def predict_proba(self, x_test) -> np.ndarray:
        scaled = self._predict_scaled_output(x_test)
        logits = self._logits_from_output(scaled)
        return sigmoid(logits)

    def predict(self, x_test) -> np.ndarray:
        regression = self.predict_regression(x_test)
        if regression is None:
            return (self.predict_proba(x_test) > 0.5).astype(int)
        return (regression > 0.0).astype(int)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError(f"{self.architecture} graph model is not fitted")
        self.torch.save(
            {
                "architecture": self.architecture,
                "graph_builder": self.graph_builder,
                "state_dict": self.model.state_dict(),
                "input_shape": self.input_shape_,
                "return_mean": self.return_mean_,
                "return_std": self.return_std_,
                "zero_return_scaled": self.zero_return_scaled_,
                "objective": self.objective,
                "class_weight": self.class_weight,
                "graph_info": self.graph_info_,
            },
            path,
        )

    def save_lightweight(self, path: str | Path) -> None:
        joblib.dump(
            {
                "architecture": self.architecture,
                "graph_builder": self.graph_builder,
                "input_shape": self.input_shape_,
                "return_mean": self.return_mean_,
                "return_std": self.return_std_,
                "zero_return_scaled": self.zero_return_scaled_,
                "objective": self.objective,
                "class_weight": self.class_weight,
                "graph_info": self.graph_info_,
            },
            path,
        )

    def _build_model(self, *, n_nodes: int):
        raise NotImplementedError

    def _forward_official(self, model, x_tensor):
        raise NotImplementedError

    def _forward_model(self, model, x_tensor):
        return self._forward_official(model, x_tensor)

    def _predict_scaled_output(self, x_test) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{self.architecture} graph model is not fitted")
        self.model.eval()
        x = self._to_graph_tensor(as_windows(x_test, name="x_test"))
        preds: list[np.ndarray] = []
        with self.torch.no_grad():
            for start in range(0, x.shape[0], self.batch_size):
                batch = x[start : start + self.batch_size]
                if self._requires_full_batch() and batch.shape[0] != self.batch_size:
                    pad_count = self.batch_size - int(batch.shape[0])
                    pad = batch[-1:].repeat(pad_count, 1, 1, 1)
                    out = self._forward_model(self.model, self.torch.cat([batch, pad], dim=0))[: batch.shape[0]]
                else:
                    out = self._forward_model(self.model, batch)
                scalar = out.mean(dim=(1, 2, 3)).detach().cpu().numpy()
                preds.append(scalar)
        return np.concatenate(preds) if preds else np.asarray([], dtype=float)

    def _build_adjacency(self, x_train: np.ndarray) -> np.ndarray:
        feature_matrix = np.transpose(x_train, (0, 2, 1)).reshape(-1, x_train.shape[1])
        adjacency = build_adjacency(
            feature_matrix,
            builder=self.graph_builder,
            feature_cols=self.feature_cols,
            normalize=True,
        )
        self.graph_info_ = graph_info(adjacency)
        return adjacency

    def _edge_tensors(self, adjacency: np.ndarray):
        rows, cols = np.nonzero(np.asarray(adjacency, dtype=float) > 1e-12)
        if len(rows) == 0:
            rows = cols = np.arange(adjacency.shape[0])
        edge_index = self.torch.as_tensor(np.vstack([rows, cols]), dtype=self.torch.long, device=self.device)
        edge_weight = self.torch.as_tensor(adjacency[rows, cols], dtype=self.torch.float32, device=self.device)
        return edge_index, edge_weight

    def _to_graph_tensor(self, x: np.ndarray):
        arr = np.transpose(np.asarray(x, dtype=np.float32), (0, 2, 1))[:, :, :, None]
        return self.torch.as_tensor(arr, dtype=self.torch.float32, device=self.device)

    def _target_tensor(self, y_scaled: np.ndarray, *, n_nodes: int):
        target = np.asarray(y_scaled, dtype=np.float32).reshape(-1, 1, 1, 1)
        target = np.repeat(target, n_nodes, axis=2)
        return self.torch.as_tensor(target, dtype=self.torch.float32, device=self.device)

    def _objective_loss(self, pred, return_y, class_y, mse_loss, bce_loss):
        if self.objective == "return_mse":
            return mse_loss(pred, return_y)
        logits = self._logits_from_output(pred)
        if self.objective == "direction_bce":
            return bce_loss(logits, class_y)
        return mse_loss(pred, return_y) + self.bce_weight * bce_loss(logits, class_y)

    def _logits_from_output(self, output):
        if self.objective == "direction_bce":
            return output / self.logit_scale
        return (output - self.zero_return_scaled_) / self.logit_scale

    def _pos_weight_tensor(self, y_class: np.ndarray):
        if self.class_weight != "balanced":
            return None
        pos = float(np.sum(y_class > 0.5))
        neg = float(len(y_class) - pos)
        weight = 1.0 if pos <= 0.0 or neg <= 0.0 else neg / pos
        return self.torch.as_tensor([weight], dtype=self.torch.float32, device=self.device)

    def _eval_loss(self, model, x_tensor, return_y, class_y, mse_loss, bce_loss) -> float:
        model.eval()
        with self.torch.no_grad():
            if not self._requires_full_batch():
                pred = self._forward_model(model, x_tensor)
                loss = self._objective_loss(pred, return_y, class_y, mse_loss, bce_loss)
                return float(loss.detach().cpu().item())
            weighted_loss = 0.0
            n_seen = 0
            for start in range(0, x_tensor.shape[0], self.batch_size):
                batch_x = x_tensor[start : start + self.batch_size]
                true_size = int(batch_x.shape[0])
                if true_size == 0:
                    continue
                if true_size != self.batch_size:
                    pad_count = self.batch_size - true_size
                    pad_x = batch_x[-1:].repeat(pad_count, 1, 1, 1)
                    batch_pred = self._forward_model(model, self.torch.cat([batch_x, pad_x], dim=0))[:true_size]
                else:
                    batch_pred = self._forward_model(model, batch_x)
                batch_loss = self._objective_loss(
                    batch_pred,
                    return_y[start : start + true_size],
                    class_y[start : start + true_size],
                    mse_loss,
                    bce_loss,
                )
                weighted_loss += float(batch_loss.detach().cpu().item()) * true_size
                n_seen += true_size
        return weighted_loss / max(1, n_seen)

    def _requires_full_batch(self) -> bool:
        return False


def graph_model_params(params: dict | None) -> dict:
    """Copy params while allowing factory call sites to pass None."""

    return dict(params or {})
