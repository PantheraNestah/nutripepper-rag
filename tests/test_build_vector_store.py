"""data/scripts/build_vector_store.py: idempotent, incremental, no real embedding model."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import chromadb
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "data" / "scripts" / "build_vector_store.py"


class FakeModel:
    """Deterministic 8-dimensional 'embedding' that records which texts it was asked to embed."""
    def __init__(self):
        self.seen = []

    def encode(self, texts):
        self.seen.extend(texts)
        return np.array([[float(len(t) % 7 + i) for i in range(8)] for t in texts])


@pytest.fixture
def builder(monkeypatch):
    fake = FakeModel()
    stub = types.ModuleType("embedder2")
    stub.embedding_model = fake
    monkeypatch.setitem(sys.modules, "embedder2", stub)
    spec = importlib.util.spec_from_file_location("build_vector_store_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, fake


def write_kb(path, texts):
    chunks = [{"chunk_id": f"doc_{i}", "text": t, "source_document": "doc.pdf", "authority": "WHO",
               "year": 2024, "domain": "nutrition", "kenya_relevance": False} for i, t in enumerate(texts)]
    path.write_text(json.dumps(chunks))


def test_builds_collection_with_expected_metadata(builder, tmp_path):
    mod, fake = builder
    kb, out = tmp_path / "kb.json", tmp_path / "store"
    write_kb(kb, ["alpha", "beta", "gamma"])
    assert mod.build_store(kb, out, batch_size=2, reset=False) == 3
    col = chromadb.PersistentClient(path=str(out)).get_collection("nutrition_kb")
    meta = col.get(ids=["doc_0"])["metadatas"][0]
    assert meta == {"source_document": "doc.pdf", "authority": "WHO", "year": 2024,
                    "domain": "nutrition", "kenya_relevance": "False"}


def test_rerun_does_not_re_embed_unchanged_chunks(builder, tmp_path):
    mod, fake = builder
    kb, out = tmp_path / "kb.json", tmp_path / "store"
    write_kb(kb, ["alpha", "beta", "gamma"])
    mod.build_store(kb, out, 32, False)
    fake.seen.clear()
    assert mod.build_store(kb, out, 32, False) == 3       # no duplicate-ID error, same count
    assert fake.seen == []


def test_only_new_or_edited_chunks_are_embedded(builder, tmp_path):
    mod, fake = builder
    kb, out = tmp_path / "kb.json", tmp_path / "store"
    write_kb(kb, ["alpha", "beta", "gamma"])
    mod.build_store(kb, out, 32, False)
    fake.seen.clear()
    write_kb(kb, ["alpha", "BETA EDITED", "gamma", "delta"])
    assert mod.build_store(kb, out, 32, False) == 4
    assert sorted(fake.seen) == ["BETA EDITED", "delta"]


def test_reset_removes_chunks_no_longer_in_the_json(builder, tmp_path):
    mod, fake = builder
    kb, out = tmp_path / "kb.json", tmp_path / "store"
    write_kb(kb, ["alpha", "beta", "gamma"])
    mod.build_store(kb, out, 32, False)
    write_kb(kb, ["alpha"])
    assert mod.build_store(kb, out, 32, False) == 3       # stale chunks linger without --reset
    assert mod.build_store(kb, out, 32, True) == 1
