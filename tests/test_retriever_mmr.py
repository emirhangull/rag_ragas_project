"""Tests for MMR and global lexical search in Retriever."""
from __future__ import annotations

import numpy as np
import pytest

from rag_mvp.retriever import Retriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_retriever(use_mmr: bool = False, mmr_lambda: float = 0.6) -> Retriever:
    r = Retriever.__new__(Retriever)
    r.top_k = 5
    r.fetch_k = 20
    r.use_mmr = use_mmr
    r.mmr_lambda = mmr_lambda
    r.lexical_global_limit = 500
    r._cross_encoder = None

    # Minimal embedder stub: returns deterministic unit vectors
    class _FakeEmbedder:
        def embed_texts(self, texts):
            rng = np.random.default_rng(42)
            vecs = rng.standard_normal((len(texts), 4)).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return (vecs / norms).tolist()

    r.embedder = _FakeEmbedder()
    return r


def _make_rows(n: int = 10) -> list[dict]:
    return [
        {
            "text": f"Belge metni içeriği {i}. Bu cümlede farklı kelimeler var.",
            "score": (n - i) / n,
            "doc_id": f"doc_{i % 3}",
            "chunk_id": f"chunk_{i}",
            "chunk_index": i,
            "source_name": f"source_{i % 3}.pdf",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# MMR tests
# ---------------------------------------------------------------------------

class TestMMR:
    def test_mmr_returns_correct_count(self) -> None:
        r = _make_retriever()
        rows = _make_rows(10)
        query_vec = [0.5, 0.5, 0.5, 0.5]
        selected = r._mmr_select(rows, query_vec, top_k=4)
        assert len(selected) == 4

    def test_mmr_returns_all_when_fewer_than_top_k(self) -> None:
        r = _make_retriever()
        rows = _make_rows(3)
        query_vec = [1.0, 0.0, 0.0, 0.0]
        selected = r._mmr_select(rows, query_vec, top_k=10)
        assert len(selected) == 3

    def test_mmr_preserves_row_structure(self) -> None:
        r = _make_retriever()
        rows = _make_rows(8)
        query_vec = [0.1, 0.9, 0.2, 0.4]
        selected = r._mmr_select(rows, query_vec, top_k=3)
        for row in selected:
            assert "text" in row
            assert "chunk_id" in row

    def test_mmr_lambda_1_picks_highest_relevance(self) -> None:
        """lambda=1 → pure relevance → first pick must be the most relevant."""
        r = _make_retriever(mmr_lambda=1.0)
        # Make rows with clear relevance ordering via score
        rows = [{"text": f"text {i}", "score": float(i), "doc_id": "d", "chunk_id": f"c{i}", "chunk_index": i, "source_name": "s"} for i in range(5)]
        query_vec = [1.0, 0.0, 0.0, 0.0]

        class _FixedEmbedder:
            def embed_texts(self, texts):
                # Return vectors where later indices are more similar to query
                return [[float(i)/5, 0, 0, 0] for i in range(len(texts))]

        r.embedder = _FixedEmbedder()
        selected = r._mmr_select(rows, query_vec, top_k=1, lambda_param=1.0)
        # The most similar to query should be selected
        assert selected[0]["chunk_id"] == "c4"


# ---------------------------------------------------------------------------
# Global lexical search (doc_id=None)
# ---------------------------------------------------------------------------

class TestSearchLexicalGlobal:
    def test_global_returns_empty_on_no_rows(self) -> None:
        r = _make_retriever()

        class _FakeIndexer:
            def scroll_all(self, limit=2000):
                return []
            def list_chunks(self, doc_id, limit=200):
                return []

        r.indexer = _FakeIndexer()
        result = r.search_lexical("test sorusu", doc_id=None, top_k=5, limit=100)
        assert result == []

    def test_global_search_uses_scroll_all_when_no_doc_id(self) -> None:
        r = _make_retriever()
        called_with = {}

        class _FakeIndexer:
            def scroll_all(self, limit=2000):
                called_with["scroll_all"] = True
                return [
                    {"text": "test sorusu içerik", "doc_id": "d1", "chunk_id": "c1",
                     "chunk_index": 0, "source_name": "f1.pdf"},
                ]
            def list_chunks(self, doc_id, limit=200):
                called_with["list_chunks"] = True
                return []

        r.indexer = _FakeIndexer()
        results = r.search_lexical("test sorusu", doc_id=None, top_k=5, limit=100)
        assert called_with.get("scroll_all") is True
        assert "list_chunks" not in called_with
        assert len(results) >= 1

    def test_per_doc_search_uses_list_chunks(self) -> None:
        r = _make_retriever()
        called_with = {}

        class _FakeIndexer:
            def scroll_all(self, limit=2000):
                called_with["scroll_all"] = True
                return []
            def list_chunks(self, doc_id, limit=200):
                called_with["list_chunks"] = doc_id
                return [
                    {"text": "test sorusu belgede", "doc_id": doc_id, "chunk_id": "c1",
                     "chunk_index": 0, "source_name": "f.pdf"},
                ]

        r.indexer = _FakeIndexer()
        results = r.search_lexical("test sorusu", doc_id="doc_abc", top_k=5, limit=100)
        assert called_with.get("list_chunks") == "doc_abc"
        assert "scroll_all" not in called_with
