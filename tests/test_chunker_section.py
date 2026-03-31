"""Tests for the new section-aware chunker features."""
from __future__ import annotations

import pytest
from rag_mvp.chunker import (
    RecursiveTextChunker,
    TextChunk,
    _has_section_markers,
    _split_on_sections,
)


class TestSectionDetection:
    def test_markdown_heading_detected(self) -> None:
        text = "# Giriş\nBu bir giriş.\n\n## Yöntem\nYöntem açıklaması."
        assert _has_section_markers(text) is True

    def test_caps_heading_detected(self) -> None:
        text = "SONUÇLAR\nBu bölümde sonuçlar açıklanmaktadır."
        assert _has_section_markers(text) is True

    def test_no_markers_plain_text(self) -> None:
        text = "Bu basit bir paragraf metnidir. Herhangi bir başlık içermez."
        assert _has_section_markers(text) is False

    def test_split_on_markdown_headings(self) -> None:
        text = "# Bölüm 1\nİçerik bir.\n# Bölüm 2\nİçerik iki."
        parts = _split_on_sections(text)
        assert len(parts) >= 2
        # second part should start with the heading
        assert any("Bölüm 2" in p for p in parts)


class TestRecursiveTextChunkerSection:
    def _make(self, chunk_size: int = 200, overlap: int = 30) -> RecursiveTextChunker:
        return RecursiveTextChunker(chunk_size=chunk_size, chunk_overlap=overlap)

    def test_returns_text_chunks(self) -> None:
        chunker = self._make()
        text = "# Başlık\nBasit bir metin.\n\n## İkinci Başlık\nBaşka bir içerik."
        results = chunker.split_text(text)
        assert len(results) >= 1
        assert all(isinstance(c, TextChunk) for c in results)

    def test_chunk_ids_are_unique(self) -> None:
        chunker = self._make()
        text = "\n\n".join([f"Paragraf {i}: " + "A" * 180 for i in range(5)])
        results = chunker.split_text(text)
        ids = [c.chunk_id for c in results]
        assert len(ids) == len(set(ids))

    def test_no_chunk_exceeds_size(self) -> None:
        chunker = self._make(chunk_size=300, overlap=50)
        text = "# Bölüm\n" + " ".join(["kelime"] * 500)
        results = chunker.split_text(text)
        for c in results:
            assert len(c.text) <= 300, f"Chunk too long: {len(c.text)}"

    def test_empty_text_returns_empty(self) -> None:
        chunker = self._make()
        assert chunker.split_text("") == []
        assert chunker.split_text("   ") == []

    def test_section_split_preserves_content(self) -> None:
        """All words from the original text should appear somewhere in chunks."""
        chunker = self._make(chunk_size=150, overlap=20)
        sections = ["# Giriş\nGiriş metni.", "# Yöntem\nYöntem metni.", "# Sonuç\nSonuç metni."]
        original = "\n\n".join(sections)
        results = chunker.split_text(original)
        combined = " ".join(c.text for c in results)
        for word in ["Giriş", "Yöntem", "Sonuç"]:
            assert word in combined, f"'{word}' kayboldu!"

    def test_overlap_not_larger_than_chunk_size(self) -> None:
        with pytest.raises(ValueError):
            RecursiveTextChunker(chunk_size=100, chunk_overlap=100)
