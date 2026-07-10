"""Visibility graph statistics backed by ts2vg."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from corn_forecast.operator.layer.base import LayerMixin, ensure_3d_windows


def visibility_graph_features(x: np.ndarray, *, kind: str = "natural") -> np.ndarray:
    """Per-window visibility graph statistics using ts2vg's official builders."""

    try:
        from ts2vg import HorizontalVG, NaturalVG
    except ImportError as exc:
        raise ImportError("ts2vg is required for visibility graph layers. Install with: pip install ts2vg") from exc
    if kind not in {"natural", "horizontal"}:
        raise ValueError(f"Unknown visibility graph kind: {kind}")
    graph_cls = NaturalVG if kind == "natural" else HorizontalVG
    windows = ensure_3d_windows(x)
    n_cases, n_nodes, n_time = windows.shape
    out = np.zeros((n_cases, n_nodes * 7), dtype=float)
    for case_idx in range(n_cases):
        features: list[float] = []
        for node_idx in range(n_nodes):
            signal = np.nan_to_num(windows[case_idx, node_idx, :], nan=0.0, posinf=0.0, neginf=0.0)
            graph = graph_cls()
            graph.build(signal)
            edges = list(getattr(graph, "edges", []))
            degree = np.zeros(n_time, dtype=float)
            for left, right in edges:
                li = int(left)
                ri = int(right)
                if 0 <= li < n_time and 0 <= ri < n_time:
                    degree[li] += 1.0
                    degree[ri] += 1.0
            denom = max(1.0, n_time * (n_time - 1) / 2.0)
            corr = 0.0
            if n_time > 1 and np.std(signal) > 1e-12 and np.std(degree) > 1e-12:
                corr = float(np.corrcoef(signal, degree)[0, 1])
            features.extend(
                [
                    len(edges) / denom,
                    float(degree.mean()) if n_time else 0.0,
                    float(degree.std()) if n_time else 0.0,
                    float(degree.max()) if n_time else 0.0,
                    float(degree[-1]) if n_time else 0.0,
                    float(np.mean(np.abs(np.diff(degree)))) if n_time > 1 else 0.0,
                    corr,
                ]
            )
        out[case_idx, :] = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return out


@dataclass
class VisibilityGraphLayer(LayerMixin):
    """Natural or horizontal visibility graph statistics."""

    kind: str = "natural"
    is_fitted_: bool = field(default=False, init=False)

    def fit(self, x, y=None):
        ensure_3d_windows(x)
        if self.kind not in {"natural", "horizontal"}:
            raise ValueError(f"Unknown visibility graph kind: {self.kind}")
        self.is_fitted_ = True
        return self

    def transform(self, x) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("VisibilityGraphLayer is not fitted")
        return visibility_graph_features(x, kind=self.kind)
