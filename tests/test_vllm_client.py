"""Tests for vllm_client.py – _clean_answer logic.

vllm_client.py only imports `re`, `logging`, `time`, and `requests` — all stdlib or
lightweight. No relative imports. We can import it directly by adjusting sys.path.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_client import NO_ANSWER_TEXT, VllmClient


def _make_client() -> VllmClient:
    return VllmClient(
        base_url="http://localhost:9999/v1",
        model="test",
        temperature=0.0,
        max_tokens=512,
        timeout_s=5,
    )


class TestCleanAnswer:
    def test_normal_answer(self) -> None:
        assert _make_client()._clean_answer("Cevap budur.") == "Cevap budur."

    def test_empty_returns_no_answer(self) -> None:
        assert _make_client()._clean_answer("") == NO_ANSWER_TEXT

    def test_none_like_empty(self) -> None:
        # _clean_answer does `(raw_answer or "").strip()` so None is safe
        assert _make_client()._clean_answer(None) == NO_ANSWER_TEXT  # type: ignore[arg-type]

    def test_whitespace_only_returns_no_answer(self) -> None:
        assert _make_client()._clean_answer("   \n  ") == NO_ANSWER_TEXT

    def test_think_tags_stripped(self) -> None:
        raw = "<think>reasoning here</think>Gerçek cevap."
        result = _make_client()._clean_answer(raw)
        assert "think" not in result.lower()
        assert "Gerçek cevap." in result

    def test_multiline_think_stripped(self) -> None:
        raw = "<think>uzun düşünce\nsatır satır</think>\n\nCevap: 42"
        result = _make_client()._clean_answer(raw)
        assert "düşünce" not in result
        assert "42" in result

    def test_reasoning_keyword_returns_no_answer(self) -> None:
        assert _make_client()._clean_answer("This is my thinking process and reasoning.") == NO_ANSWER_TEXT

    def test_final_answer_extracted(self) -> None:
        raw = "Bazı açıklamalar. Cevap: Sonuç budur."
        result = _make_client()._clean_answer(raw)
        assert "Sonuç budur." in result

    def test_no_answer_text_passthrough(self) -> None:
        assert _make_client()._clean_answer(NO_ANSWER_TEXT) == NO_ANSWER_TEXT

    def test_quoted_no_answer_passthrough(self) -> None:
        assert _make_client()._clean_answer(f'"{NO_ANSWER_TEXT}"') == NO_ANSWER_TEXT

    def test_long_answer_truncated(self) -> None:
        long_text = "Bu bir cümle. " * 100
        result = _make_client()._clean_answer(long_text)
        assert len(result) <= 750

    def test_quotes_stripped(self) -> None:
        result = _make_client()._clean_answer('"Tırnaklı cevap"')
        assert not result.startswith('"')
        assert not result.endswith('"')

    def test_multiline_compacted(self) -> None:
        raw = "Satır 1\nSatır 2\n\nSatır 3"
        result = _make_client()._clean_answer(raw)
        assert "\n" not in result
        assert "Satır 1" in result
        assert "Satır 3" in result
