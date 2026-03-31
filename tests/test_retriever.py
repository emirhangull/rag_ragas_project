"""Tests for retriever.py – tokenization, reranking, and query expansion."""
from __future__ import annotations

from rag_mvp.retriever import Retriever


class TestTokenize:
    def test_basic_tokenization(self) -> None:
        tokens = Retriever._tokenize("Raporda kullanılan modeller nelerdir")
        assert isinstance(tokens, set)
        assert len(tokens) > 0

    def test_stopwords_filtered(self) -> None:
        tokens = Retriever._tokenize("bu bir test ve deneme için yapılır")
        assert "bu" not in tokens
        assert "bir" not in tokens
        assert "ve" not in tokens

    def test_model_not_a_stopword(self) -> None:
        """'model' was previously a stopword — verify it is no longer filtered."""
        tokens = Retriever._tokenize("model performansı nasıl")
        assert "model" in tokens

    def test_single_char_filtered(self) -> None:
        tokens = Retriever._tokenize("a b c test kelime")
        assert "a" not in tokens
        assert "test" in tokens

    def test_turkish_chars_preserved(self) -> None:
        tokens = Retriever._tokenize("çğöşü karakter testi")
        assert len(tokens) > 0

    def test_numbers_included(self) -> None:
        tokens = Retriever._tokenize("2024 yılında 3 model")
        assert "2024" in tokens
        assert "model" in tokens


class TestRerank:
    def _make_retriever(self) -> Retriever:
        r = Retriever.__new__(Retriever)
        r.top_k = 5
        r.fetch_k = 20
        r._cross_encoder = None
        return r

    def test_rerank_respects_top_k(self) -> None:
        r = self._make_retriever()
        rows = [{"text": f"metin {i}", "score": float(i) / 10} for i in range(10)]
        result = r._rerank("metin test", rows, top_k=3)
        assert len(result) <= 3

    def test_rerank_adds_rerank_score(self) -> None:
        r = self._make_retriever()
        rows = [{"text": "test kelime", "score": 0.8}]
        result = r._rerank("test", rows, top_k=5)
        assert "rerank_score" in result[0]

    def test_rerank_sorts_descending(self) -> None:
        r = self._make_retriever()
        rows = [
            {"text": "tamamen ayrı", "score": 0.9},
            {"text": "metin test deneme sorgu", "score": 0.5},
        ]
        result = r._rerank("metin test sorgu", rows, top_k=2)
        assert result[0]["rerank_score"] >= result[1]["rerank_score"]

    def test_rerank_empty_query_uses_dense(self) -> None:
        r = self._make_retriever()
        rows = [
            {"text": "some text", "score": 0.8},
            {"text": "other text", "score": 0.3},
        ]
        result = r._rerank("", rows, top_k=5)
        scores = [row["rerank_score"] for row in result]
        assert scores == sorted(scores, reverse=True)


class TestExpandQuery:
    def _make_retriever(self) -> Retriever:
        return Retriever.__new__(Retriever)

    def test_base_question_always_first(self) -> None:
        r = self._make_retriever()
        result = r.expand_query("Test sorusu nedir")
        assert result[0] == "Test sorusu nedir"

    def test_returns_multiple_variants(self) -> None:
        r = self._make_retriever()
        result = r.expand_query("Raporda kullanılan model nedir")
        assert len(result) >= 3

    def test_no_duplicates(self) -> None:
        r = self._make_retriever()
        result = r.expand_query("model nedir")
        lowered = [x.lower() for x in result]
        assert len(lowered) == len(set(lowered))


class TestStripQuestionWords:
    def test_removes_question_particles(self) -> None:
        result = Retriever._strip_question_words("raporda kullanılan model nedir")
        assert "nedir" not in result.lower()
        assert "raporda" not in result.lower()

    def test_keeps_content_words(self) -> None:
        result = Retriever._strip_question_words("raporda kullanılan model nedir")
        assert "model" in result.lower()
