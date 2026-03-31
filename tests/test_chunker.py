"""Tests for chunker.py – RecursiveTextChunker logic."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunker import RecursiveTextChunker, TextChunk

import pytest


class TestRecursiveTextChunker:
    """Unit tests for RecursiveTextChunker."""

    def test_empty_text_returns_empty(self) -> None:
        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
        assert chunker.split_text("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
        assert chunker.split_text("   \n\n  \t  ") == []

    def test_small_text_single_chunk(self) -> None:
        chunker = RecursiveTextChunker(chunk_size=500, chunk_overlap=50)
        text = "Bu kısa bir test metnidir."
        result = chunker.split_text(text)
        assert len(result) == 1
        assert result[0].text == text
        assert result[0].chunk_id == "chunk_0"

    def test_large_text_multiple_chunks(self) -> None:
        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
        text = "Kelime " * 200  # ~1400 chars
        result = chunker.split_text(text)
        assert len(result) > 1
        for chunk in result:
            assert isinstance(chunk, TextChunk)
            assert len(chunk.text) <= chunker.chunk_size + chunker.chunk_overlap

    def test_paragraph_boundaries_preserved(self) -> None:
        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
        text = "Paragraf bir.\n\nParagraf iki.\n\nParagraf üç.\n\nParagraf dört ve uzun bir metin " + "x " * 80
        result = chunker.split_text(text)
        assert len(result) > 1
        # The first chunk should contain paragraph separator-based split
        assert "Paragraf bir." in result[0].text

    def test_chunk_ids_sequential(self) -> None:
        chunker = RecursiveTextChunker(chunk_size=50, chunk_overlap=10)
        text = "A " * 100
        result = chunker.split_text(text)
        for idx, chunk in enumerate(result):
            assert chunk.chunk_id == f"chunk_{idx}"

    def test_overlap_less_than_chunk_size_required(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
            RecursiveTextChunker(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
            RecursiveTextChunker(chunk_size=100, chunk_overlap=150)

    def test_normalize_whitespace_preserves_newlines(self) -> None:
        chunker = RecursiveTextChunker()
        text = "Satır 1\n\nSatır 2\nSatır 3"
        normalized = chunker._normalize_whitespace(text)
        assert "\n\n" in normalized
        assert "\n" in normalized

    def test_normalize_whitespace_collapses_tabs_and_spaces(self) -> None:
        chunker = RecursiveTextChunker()
        text = "A   B\t\tC     D"
        normalized = chunker._normalize_whitespace(text)
        assert "  " not in normalized  # No double spaces
        assert "\t" not in normalized  # No tabs

    def test_word_boundary_tail_respects_words(self) -> None:
        text = "bu bir uzun metin parçasıdır ve kesilmeli"
        tail = RecursiveTextChunker._word_boundary_tail(text, 20)
        # Should not cut mid-word
        assert len(tail) <= 20
        assert not tail.startswith("ça")  # Should not start in middle of "parçasıdır"


class TestNormalizeWhitespace:
    """Focused tests for _normalize_whitespace."""

    def test_triple_newlines_collapsed(self) -> None:
        text = "A\n\n\nB"
        result = RecursiveTextChunker._normalize_whitespace(text)
        assert result == "A\n\nB"

    def test_single_newline_kept(self) -> None:
        text = "A\nB"
        result = RecursiveTextChunker._normalize_whitespace(text)
        assert result == "A\nB"

    def test_double_newline_kept(self) -> None:
        text = "A\n\nB"
        result = RecursiveTextChunker._normalize_whitespace(text)
        assert result == "A\n\nB"
