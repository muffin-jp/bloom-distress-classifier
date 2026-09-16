"""The shipped model: weights, layout, and the two files that carry them."""

# NumPy's stubs leak Unknown under pyright strict at this numeric boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from conftest import make_row
from sklearn.linear_model import LogisticRegression

from dc.artifact import (
    ArtifactError,
    dataset_sha256,
    is_servable,
    load,
    read_segmentation,
    save,
    source_commit,
)
from dc.features import EMBED_MODEL, EMBED_MODEL_REVISION, FEELINGS, build_features, feeling_one_hot
from dc.model import DistressModel, FeatureLayout, fit_model, sigmoid
from dc.schema import Feeling
from dc.selection import Config
from dc.text import SEGMENTATION

#: The segmentation as it appears in an artifact, matching dc.report's writer.
SEGMENTATION_JSON = {
    "window": SEGMENTATION.window,
    "stride": SEGMENTATION.stride,
    "boundary": SEGMENTATION.boundary,
}


def layout(dim: int = 4, *, include_feeling: bool = False) -> FeatureLayout:
    return FeatureLayout(EMBED_MODEL, EMBED_MODEL_REVISION, dim, include_feeling)


def model(dim: int = 4) -> DistressModel:
    return DistressModel(coef=np.arange(dim, dtype=float) / 10, intercept=-0.3, layout=layout(dim))


class FixedEmbedder:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.ones((len(texts), self.dim), dtype=np.float32)


# --- features ---------------------------------------------------------------


def test_feeling_one_hot_follows_the_recorded_column_order() -> None:
    encoded = feeling_one_hot([Feeling.CUSTOM, Feeling.PROUD])
    assert encoded.shape == (2, len(FEELINGS))
    assert encoded[0, FEELINGS.index(Feeling.CUSTOM)] == 1.0
    assert encoded[1, FEELINGS.index(Feeling.PROUD)] == 1.0
    assert np.all(encoded.sum(axis=1) == 1.0)


def test_build_features_appends_the_feeling_only_when_asked() -> None:
    rows = [make_row("a"), make_row("b")]
    assert build_features(rows, FixedEmbedder(4), include_feeling=False).shape == (2, 4)
    assert build_features(rows, FixedEmbedder(4), include_feeling=True).shape == (2, 4 + 7)


# --- model ------------------------------------------------------------------


def test_sigmoid_is_stable_for_extreme_logits() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        values = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert values.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_numpy_prediction_matches_scikit_learn() -> None:
    """The consumer serves with numpy alone. It must agree with what was trained."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 6))
    y = (x[:, 0] + rng.normal(scale=0.5, size=80) > 0).astype(int)
    estimator = LogisticRegression(C=1.0, max_iter=5000).fit(x, y)
    ours = DistressModel(
        coef=np.asarray(estimator.coef_[0], dtype=float),
        intercept=float(np.asarray(estimator.intercept_).ravel()[0]),
        layout=layout(6),
    )
    assert np.allclose(ours.predict_proba_features(x), estimator.predict_proba(x)[:, 1], atol=1e-9)


def test_model_rejects_weights_that_do_not_fit_the_layout() -> None:
    with pytest.raises(ValueError, match="layout expects"):
        DistressModel(coef=np.zeros(3), intercept=0.0, layout=layout(4))


def test_model_rejects_non_finite_weights() -> None:
    with pytest.raises(ValueError, match="finite"):
        DistressModel(coef=np.array([np.nan, 0, 0, 0]), intercept=0.0, layout=layout(4))


def test_model_rejects_an_embedder_of_the_wrong_width() -> None:
    with pytest.raises(ValueError, match="trained on 4-d"):
        model(4).predict_proba([make_row("a")], FixedEmbedder(8))


def test_fit_model_rejects_a_config_that_disagrees_with_the_layout() -> None:
    with pytest.raises(ValueError, match="disagree"):
        fit_model(Config(1.0, "none", True), np.zeros((4, 4)), np.array([0, 1, 0, 1]), layout(4))


# --- artifact ---------------------------------------------------------------


def test_artifact_round_trips_weights_exactly(tmp_path: Path) -> None:
    original = model()
    save(original, {"thresholds": None}, tmp_path)
    loaded = load(tmp_path)
    assert np.array_equal(loaded.model.coef, original.coef)
    assert loaded.model.intercept == original.intercept
    assert loaded.model.layout == original.layout


def test_artifact_weights_load_without_pickle(tmp_path: Path) -> None:
    save(model(), {}, tmp_path)
    with np.load(tmp_path / "model.npz", allow_pickle=False) as data:
        assert data["coef"].dtype == np.float64
        assert data["intercept"].shape == (1,)


def test_artifact_from_another_embedder_revision_is_refused(tmp_path: Path) -> None:
    # Scores over the wrong vectors look normal and mean nothing.
    save(model(), {}, tmp_path)
    with pytest.raises(ArtifactError, match="refusing to load"):
        load(tmp_path, embed_revision="0000000")


def test_artifact_with_an_unknown_schema_version_is_refused(tmp_path: Path) -> None:
    save(model(), {}, tmp_path)
    document = json.loads((tmp_path / "model.json").read_text())
    document["schema_version"] = 99
    (tmp_path / "model.json").write_text(json.dumps(document))
    with pytest.raises(ArtifactError, match="schema_version"):
        load(tmp_path)


def test_artifact_whose_json_disagrees_with_its_weights_is_refused(tmp_path: Path) -> None:
    save(model(4), {}, tmp_path)
    document = json.loads((tmp_path / "model.json").read_text())
    document["embedder"]["dim"] = 8
    (tmp_path / "model.json").write_text(json.dumps(document))
    with pytest.raises(ArtifactError, match="layout expects"):
        load(tmp_path)


def test_missing_artifact_files_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="missing"):
        load(tmp_path)


def test_a_model_without_thresholds_is_not_servable() -> None:
    assert not is_servable({"thresholds": None})
    assert not is_servable({})


def test_a_model_with_ordered_thresholds_and_a_segmentation_is_servable() -> None:
    assert is_servable({"thresholds": {"low": 0.1, "high": 0.8}, "segmentation": SEGMENTATION_JSON})


def test_a_model_without_a_segmentation_is_not_servable() -> None:
    """The skip band is only safe as fitted if the consumer splits notes the same way.

    An artifact from before segment scoring has thresholds that look perfectly
    valid and a rule this code no longer implements, so "has thresholds" stopped
    being enough to serve on.
    """
    assert not is_servable({"thresholds": {"low": 0.1, "high": 0.8}})
    assert not is_servable({"thresholds": {"low": 0.1, "high": 0.8}, "segmentation": {}})


@pytest.mark.parametrize(
    "bands", [{"low": 0.8, "high": 0.1}, {"low": -0.1, "high": 0.5}, {"low": 0.1, "high": 1.5}]
)
def test_inverted_or_out_of_range_thresholds_are_not_servable(bands: dict[str, float]) -> None:
    assert not is_servable({"thresholds": bands, "segmentation": SEGMENTATION_JSON})


def test_the_recorded_segmentation_round_trips() -> None:
    read = read_segmentation({"segmentation": SEGMENTATION_JSON})
    assert (read.window, read.stride, read.boundary) == (
        SEGMENTATION.window,
        SEGMENTATION.stride,
        SEGMENTATION.boundary,
    )


def test_an_artifact_without_a_segmentation_is_refused_rather_than_defaulted() -> None:
    """Falling back to the current default would serve one rule under another's thresholds."""
    with pytest.raises(ArtifactError, match="records no segmentation"):
        read_segmentation({"thresholds": {"low": 0.1, "high": 0.8}})


@pytest.mark.parametrize(
    "block",
    [
        {"window": 8, "stride": 4},
        {"window": "8", "stride": 4, "boundary": r"\s+"},
        {"window": 0, "stride": 4, "boundary": r"\s+"},
        {"window": 8, "stride": 4, "boundary": "([unclosed"},
    ],
)
def test_a_malformed_segmentation_is_refused(block: dict[str, object]) -> None:
    with pytest.raises(ArtifactError):
        read_segmentation({"segmentation": block})


def test_dataset_hash_tracks_content_and_order(tmp_path: Path) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("one\n")
    b.write_text("two\n")
    first = dataset_sha256([a, b])
    assert dataset_sha256([a, b]) == first
    assert dataset_sha256([b, a]) != first
    a.write_text("changed\n")
    assert dataset_sha256([a, b]) != first


def test_source_commit_degrades_outside_a_repository(tmp_path: Path) -> None:
    assert source_commit(tmp_path) == {"commit": None, "dirty": None}
