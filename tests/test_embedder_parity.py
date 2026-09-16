"""This project's embedder snapshot and production's must stay byte-identical.

The artifact was built on MiniLM revision c9745ed1; bloom-langgraph serves ea78891.
That is safe only because the two snapshots are the same bytes on every file that
affects an embedding — verified once, and re-verified here whenever both repos'
vendored weights are present. If this fails, the served classifier scores vectors
it was not trained on, and it would do so without any visible error.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

OURS = Path(__file__).resolve().parents[1] / "models" / "all-MiniLM-L6-v2"
THEIRS = Path(__file__).resolve().parents[2] / "bloom-langgraph" / "app" / "rag" / "model"

EMBEDDING_FILES = (
    "model.safetensors",
    "tokenizer.json",
    "vocab.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "modules.json",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
    "1_Pooling/config.json",
)

pytestmark = pytest.mark.skipif(
    not (OURS.exists() and THEIRS.exists()),
    reason="needs both repos' vendored embedder weights checked out side by side",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", EMBEDDING_FILES)
def test_snapshot_file_is_byte_identical(name: str) -> None:
    ours, theirs = OURS / name, THEIRS / name
    assert ours.exists() and theirs.exists(), f"{name} missing from one snapshot"
    assert digest(ours) == digest(theirs), (
        f"{name} differs between this project's embedder snapshot and bloom-langgraph's. "
        "The served classifier would score vectors it was not trained on."
    )
