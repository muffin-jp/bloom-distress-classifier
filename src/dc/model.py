"""The trained model, as the consumer sees it: a weight vector and a layout.

After training, scikit-learn has done its job. What ships is ``coef``, an
intercept, and a :class:`FeatureLayout` that says exactly how to rebuild the
input — which embedder, which revision, whether the feeling is appended and in
what column order. Prediction is one dot product and a sigmoid in numpy.

That is a deliberate property, not a shortcut. The consumer in ``bloom-langgraph``
does not need scikit-learn, cannot be broken by a scikit-learn upgrade, and never
unpickles anything: the model is data it can read, not code it has to trust.
"""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dc.features import FEELINGS, Embedder, build_features
from dc.schema import Row
from dc.selection import Config, make_estimator

__all__ = ["DistressModel", "FeatureLayout", "fit_model", "sigmoid"]


@dataclass(frozen=True)
class FeatureLayout:
    """Everything needed to rebuild the model's input from a row."""

    embed_model: str
    embed_revision: str
    embed_dim: int
    include_feeling: bool
    feelings: tuple[str, ...] = tuple(f.value for f in FEELINGS)

    @property
    def n_features(self) -> int:
        return self.embed_dim + (len(self.feelings) if self.include_feeling else 0)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic: no overflow warning for large negative logits."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(z, dtype=float)))


@dataclass(frozen=True, eq=False)
class DistressModel:
    coef: np.ndarray
    intercept: float
    layout: FeatureLayout

    def __post_init__(self) -> None:
        if self.coef.ndim != 1 or self.coef.shape[0] != self.layout.n_features:
            raise ValueError(
                f"coef has shape {self.coef.shape}, layout expects ({self.layout.n_features},)"
            )
        if not np.all(np.isfinite(self.coef)) or not np.isfinite(self.intercept):
            raise ValueError("model weights must be finite")

    def predict_proba_features(self, x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.layout.n_features:
            raise ValueError(
                f"features have shape {matrix.shape}, model expects (n, {self.layout.n_features})"
            )
        return sigmoid(matrix @ self.coef + self.intercept)

    def predict_proba(self, rows: Sequence[Row], embedder: Embedder) -> np.ndarray:
        if embedder.dim and embedder.dim != self.layout.embed_dim:
            raise ValueError(
                f"embedder produces {embedder.dim}-d vectors; model was trained on "
                f"{self.layout.embed_dim}-d"
            )
        return self.predict_proba_features(
            build_features(rows, embedder, include_feeling=self.layout.include_feeling)
        )


def fit_model(
    config: Config, x: np.ndarray, y: np.ndarray, layout: FeatureLayout, *, seed: int = 0
) -> DistressModel:
    """Fit once with scikit-learn, then keep only the numbers."""
    if config.include_feeling != layout.include_feeling:
        raise ValueError("config and layout disagree about the feeling features")
    estimator = make_estimator(config, seed=seed)
    estimator.fit(x, y)
    return DistressModel(
        coef=np.asarray(estimator.coef_[0], dtype=np.float64),
        intercept=float(np.asarray(estimator.intercept_, dtype=np.float64).ravel()[0]),
        layout=layout,
    )
