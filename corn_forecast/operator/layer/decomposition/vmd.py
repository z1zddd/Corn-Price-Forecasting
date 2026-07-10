"""VMD expansion layer backed by vmdpy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from corn_forecast.operator.layer.base import LayerMixin, ensure_3d_windows


def vmd_expand_windows(
    x: np.ndarray,
    *,
    feature_cols: list[str] | None,
    selected_columns: list[str] | None,
    modes: int,
    alpha: float,
) -> tuple[np.ndarray, list[str]]:
    """Append per-window VMD modes as extra graph nodes without future leakage."""

    windows = ensure_3d_windows(x)
    if modes <= 0:
        return windows, list(feature_cols or [])
    selected = vmd_column_indices(feature_cols or [], selected_columns or [], windows.shape[1])
    if not selected:
        return windows, list(feature_cols or [])
    try:
        from vmdpy import VMD
    except ImportError as exc:
        raise ImportError("vmdpy is required for VMDLayer. Install with: pip install vmdpy") from exc

    extra_blocks: list[np.ndarray] = []
    extra_names: list[str] = []
    base_names = list(feature_cols or [f"feature_{idx}" for idx in range(windows.shape[1])])
    for idx in selected:
        modes_for_feature = np.zeros((windows.shape[0], modes, windows.shape[2]), dtype=float)
        for sample_idx in range(windows.shape[0]):
            modes_for_feature[sample_idx] = vmd_modes_for_signal(
                windows[sample_idx, idx, :],
                modes=modes,
                alpha=alpha,
                vmd_func=VMD,
            )
        extra_blocks.append(modes_for_feature)
        base_name = base_names[idx] if idx < len(base_names) else f"feature_{idx}"
        extra_names.extend([f"{base_name}_vmd_mode{k + 1}" for k in range(modes)])
    expanded = np.concatenate([windows, *extra_blocks], axis=1)
    return expanded.astype(float), base_names + extra_names


def vmd_column_indices(feature_cols: list[str], selected_columns: list[str], n_features: int) -> list[int]:
    """Resolve requested VMD columns by name, defaulting to dce_corn_close."""

    if not feature_cols or len(feature_cols) != n_features:
        return []
    if selected_columns:
        wanted = set(selected_columns)
        return [idx for idx, name in enumerate(feature_cols) if name in wanted]
    defaults = {"dce_corn_close"}
    return [idx for idx, name in enumerate(feature_cols) if name in defaults]


def vmd_modes_for_signal(signal: np.ndarray, *, modes: int, alpha: float, vmd_func) -> np.ndarray:
    """Return robust VMD modes for one short signal."""

    centered = np.nan_to_num(np.asarray(signal, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if centered.size < 3 or np.nanstd(centered) < 1e-10:
        return np.tile(centered.reshape(1, -1), (modes, 1))
    try:
        u, _u_hat, _omega = vmd_func(centered, alpha, 0.0, modes, 0, 1, 1e-7)
        arr = np.asarray(u, dtype=float)
        if arr.shape[0] != modes:
            arr = np.resize(arr, (modes, centered.size))
        if arr.shape[1] != centered.size:
            grid_new = np.linspace(0, 1, centered.size)
            grid_old = np.linspace(0, 1, arr.shape[1])
            arr = np.asarray([np.interp(grid_new, grid_old, row) for row in arr])
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return np.tile(centered.reshape(1, -1), (modes, 1))


@dataclass
class VMDLayer(LayerMixin):
    """Append VMD modes as additional graph nodes."""

    feature_cols: list[str] | None = None
    selected_columns: list[str] | None = None
    modes: int = 0
    alpha: float = 2000.0
    output_feature_cols_: list[str] = field(default_factory=list, init=False)
    is_fitted_: bool = field(default=False, init=False)

    def fit(self, x, y=None):
        ensure_3d_windows(x)
        self.is_fitted_ = True
        return self

    def transform(self, x) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("VMDLayer is not fitted")
        expanded, names = vmd_expand_windows(
            x,
            feature_cols=self.feature_cols,
            selected_columns=self.selected_columns,
            modes=int(self.modes),
            alpha=float(self.alpha),
        )
        self.output_feature_cols_ = names
        return expanded
