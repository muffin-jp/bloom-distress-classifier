"""The embedding seam.

``Embedder`` is the only view any model has of a sentence encoder:
``embed(texts) -> (n, dim) float32``, unit vectors so cosine is a dot product.
Copied in spirit from ``bloom-langgraph``'s ``app/rag/embedder.py``, and for the
same reason — a stub drives the tests with no model and no network, and the real
model drops in behind the same interface.

The weights are the **same pinned revision** the companion repo serves. That is
not tidiness: a classifier trained on one revision's vectors and served against
another's is silently wrong, with no error and no obvious symptom.
"""

# sentence_transformers is untyped and numpy's stubs leak Unknown under pyright
# strict. Our own logic stays typed; narrowly relax those rules here.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingImports=false
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from dc.schema import Feeling, Row

__all__ = [
    "EMBED_MODEL",
    "EMBED_MODEL_REVISION",
    "FEELINGS",
    "Embedder",
    "MODEL_DIR",
    "SentenceTransformerEmbedder",
    "build_features",
    "feeling_one_hot",
    "fetch_model",
    "load_embedder",
]

# Pinned to match bloom-langgraph's app/config.py exactly.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "all-MiniLM-L6-v2"


class Embedder(Protocol):
    """``embed`` returns ``(n, dim)`` float32 rows, L2-normalised."""

    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """The pinned local model, loaded from vendored weights.

    ``local_files_only=True`` keeps the failure loud: if the weights are not
    vendored this raises at construction rather than quietly downloading a
    different revision than the one the numbers were produced with.
    """

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        from sentence_transformers import SentenceTransformer

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Embedding model not vendored at {model_dir}. Run `make vendor-model` "
                "(one download, ~90MB) before training or evaluating baseline 3."
            )
        self._model = SentenceTransformer(str(model_dir), local_files_only=True)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension", None
        )
        self.dim = int(get_dim() or 0) if get_dim is not None else 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def fetch_model(model_dir: Path = MODEL_DIR) -> Path:
    """Vendor the pinned weights locally. Build-time only; the network is open here."""
    from huggingface_hub import snapshot_download

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=EMBED_MODEL,
        revision=EMBED_MODEL_REVISION,
        local_dir=str(model_dir),
        ignore_patterns=["*.bin", "*.h5", "*.ot", "*.msgpack", "onnx/*", "openvino/*"],
    )
    return model_dir


#: Column order of the feeling one-hot. Recorded in the artifact: a model trained
#: with one column order and served with another would misread every feeling,
#: silently, with plausible-looking scores.
FEELINGS: tuple[Feeling, ...] = tuple(Feeling)


def feeling_one_hot(feelings: Sequence[Feeling]) -> np.ndarray:
    """``(n, 7)`` one-hot of the chip each player picked, in :data:`FEELINGS` order."""
    index = {feeling: column for column, feeling in enumerate(FEELINGS)}
    out = np.zeros((len(feelings), len(FEELINGS)), dtype=np.float32)
    for row, feeling in enumerate(feelings):
        out[row, index[feeling]] = 1.0
    return out


def build_features(rows: Sequence[Row], embedder: Embedder, *, include_feeling: bool) -> np.ndarray:
    """The model's input: sentence embedding, optionally followed by the feeling.

    ``include_feeling`` exists to be ablated, not to be switched on. In this
    dataset the chip is close to a label leak — see the feeling-swap probe in
    ``dc.selection`` for why a model that reads it is unsafe to serve.
    """
    text = np.asarray(embedder.embed([row.free_text for row in rows]), dtype=np.float32)
    if not include_feeling:
        return text
    return np.hstack([text, feeling_one_hot([row.feeling for row in rows])])


def load_embedder(model_dir: Path = MODEL_DIR) -> Embedder:
    return SentenceTransformerEmbedder(model_dir)


def main() -> None:
    """`python -m dc.features` vendors the weights."""
    path = fetch_model()
    print(f"Vendored {EMBED_MODEL} @ {EMBED_MODEL_REVISION[:8]} into {path}.")


if __name__ == "__main__":
    main()
