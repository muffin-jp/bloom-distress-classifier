"""``model.npz`` + ``model.json``: the model as two files anyone can read.

**No pickle, no joblib.** The weights are a float array in ``.npz`` loaded with
``allow_pickle=False``; everything else is JSON. The consuming service never has
to execute anything to load the model — the same discipline ``bloom-langgraph``
applies to its retrieval index.

Weights are stored as float64 rather than float32. The artifact is ~3KB either
way, and float64 makes the *weights* round-trip exactly, so a CI check can
compare a rebuilt artifact's weights for equality. Predictions are a different
matter: numpy and scikit-learn apply the same weights in a different floating-point
order, and agree to about 1e-7 — close enough to never change a routing
decision, not close enough to call identical.

Loading is strict. An artifact trained against a different embedder revision is
refused rather than served, because it would still produce confident scores —
over the wrong vectors.
"""

# NumPy's savez/load stubs leak Unknown under pyright strict at this boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from dc.features import EMBED_MODEL, EMBED_MODEL_REVISION
from dc.model import DistressModel, FeatureLayout
from dc.text import Segmentation

__all__ = [
    "ARTIFACT_DIR",
    "SCHEMA_VERSION",
    "ArtifactError",
    "LoadedArtifact",
    "dataset_sha256",
    "is_servable",
    "load",
    "read_segmentation",
    "save",
    "source_commit",
]

#: Bumped to 2 when the skip band moved from whole notes to segments: an artifact
#: without a `segmentation` block records a decision rule this code no longer
#: implements, so refusing it outright is the only safe reading.
SCHEMA_VERSION = 2
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts"
NPZ_NAME = "model.npz"
#: Paths whose uncommitted changes make an artifact's provenance "dirty".
TRAINING_INPUTS = ("src", "data", "pyproject.toml", "uv.lock")
JSON_NAME = "model.json"


class ArtifactError(ValueError):
    """The artifact is malformed, or was built for a different embedder."""


@dataclass(frozen=True)
class LoadedArtifact:
    model: DistressModel
    metadata: dict[str, Any]


def save(
    model: DistressModel, metadata: dict[str, Any], directory: Path = ARTIFACT_DIR
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / NPZ_NAME
    json_path = directory / JSON_NAME
    np.savez(
        npz_path,
        coef=np.asarray(model.coef, dtype=np.float64),
        intercept=np.asarray([model.intercept], dtype=np.float64),
    )
    layout = model.layout
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "embedder": {
            "model": layout.embed_model,
            "revision": layout.embed_revision,
            "dim": layout.embed_dim,
        },
        "features": {
            "include_feeling": layout.include_feeling,
            "feelings": list(layout.feelings),
            "n_features": layout.n_features,
        },
        **metadata,
    }
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return npz_path, json_path


def load(
    directory: Path = ARTIFACT_DIR,
    *,
    embed_model: str = EMBED_MODEL,
    embed_revision: str = EMBED_MODEL_REVISION,
) -> LoadedArtifact:
    json_path = directory / JSON_NAME
    npz_path = directory / NPZ_NAME
    if not json_path.exists() or not npz_path.exists():
        raise ArtifactError(f"missing {JSON_NAME} or {NPZ_NAME} in {directory}")

    raw: Any = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ArtifactError(f"{JSON_NAME} is not a JSON object")
    document = cast("dict[str, Any]", raw)

    if document.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(
            f"schema_version {document.get('schema_version')!r} is not {SCHEMA_VERSION}"
        )
    embedder = cast("dict[str, Any]", document.get("embedder") or {})
    if embedder.get("model") != embed_model or embedder.get("revision") != embed_revision:
        raise ArtifactError(
            f"artifact was trained on {embedder.get('model')}@{embedder.get('revision')}, "
            f"but this service embeds with {embed_model}@{embed_revision}. Scores over the "
            "wrong vectors look normal and mean nothing — refusing to load."
        )
    features = cast("dict[str, Any]", document.get("features") or {})

    with np.load(npz_path, allow_pickle=False) as data:
        coef = np.asarray(data["coef"], dtype=np.float64)
        intercept = float(np.asarray(data["intercept"], dtype=np.float64)[0])

    layout = FeatureLayout(
        embed_model=str(embedder["model"]),
        embed_revision=str(embedder["revision"]),
        embed_dim=int(embedder["dim"]),
        include_feeling=bool(features.get("include_feeling", False)),
        feelings=tuple(str(f) for f in cast("list[Any]", features.get("feelings") or [])),
    )
    try:
        model = DistressModel(coef=coef, intercept=intercept, layout=layout)
    except ValueError as exc:
        raise ArtifactError(str(exc)) from exc
    return LoadedArtifact(model=model, metadata=document)


def read_segmentation(metadata: dict[str, Any]) -> Segmentation:
    """The splitting rule the artifact was fitted with, as an object.

    Raises rather than falling back to :data:`dc.text.SEGMENTATION`. A default here
    would let an artifact fitted under one rule be served under another, which is
    the precise failure the recorded parameters exist to prevent — and it would be
    invisible, because both rules produce perfectly ordinary-looking scores.
    """
    raw = metadata.get("segmentation")
    if not isinstance(raw, dict):
        raise ArtifactError(
            "artifact records no segmentation, so the skip band's rule is unknown. It was "
            "fitted before segment scoring; retrain rather than guessing the parameters."
        )
    block = cast("dict[str, Any]", raw)
    window, stride, boundary = block.get("window"), block.get("stride"), block.get("boundary")
    if not (isinstance(window, int) and isinstance(stride, int) and isinstance(boundary, str)):
        raise ArtifactError(f"segmentation block is malformed: {block!r}")
    try:
        return Segmentation(window=window, stride=stride, boundary=boundary)
    except ValueError as exc:
        raise ArtifactError(str(exc)) from exc


def is_servable(metadata: dict[str, Any]) -> bool:
    """True only once operating thresholds have been fitted.

    A model with no thresholds has scores and no decision rule. Serving it would
    mean someone picking 0.5 at the call site — the exact choice this project
    exists to make deliberately, on a cost model, instead.
    """
    thresholds = metadata.get("thresholds")
    if not isinstance(thresholds, dict):
        return False
    bands = cast("dict[str, Any]", thresholds)
    low, high = bands.get("low"), bands.get("high")
    if not (
        isinstance(low, (int, float))
        and isinstance(high, (int, float))
        and 0.0 <= float(low) <= float(high) <= 1.0
    ):
        return False
    # The skip band is only safe as fitted if the consumer splits notes the same
    # way, so an artifact that does not say how is not servable either.
    try:
        read_segmentation(metadata)
    except ArtifactError:
        return False
    return True


def dataset_sha256(paths: Sequence[Path]) -> str:
    """One hash over the files that define the training data, in the given order."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes() if path.exists() else b"<missing>")
    return digest.hexdigest()


def source_commit(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """HEAD at training time, and whether the *training inputs* were dirty.

    "Dirty" is scoped to what can change a trained model: the package source,
    the data, and the locked dependencies. An edited README cannot, and neither
    can the artifacts and reports a training run rewrites — counting those made
    the first committed artifact claim a dirty tree it did not have.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", *TRAINING_INPUTS],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status)}
