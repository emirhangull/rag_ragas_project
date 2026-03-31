"""Tests for file_loader.py – _normalize_text and file loading."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from file_loader import _normalize_text, load_text_from_path

import pytest


class TestNormalizeText:
    """Unit tests for _normalize_text."""

    def test_empty_string(self) -> None:
        assert _normalize_text("") == ""

    def test_none_like_empty(self) -> None:
        assert _normalize_text("") == ""

    def test_unicode_normalization(self) -> None:
        # NFKC should normalize compatibility characters
        text = "ﬁ"  # U+FB01 latin small ligature fi
        result = _normalize_text(text)
        assert "fi" in result

    def test_soft_hyphen_removed(self) -> None:
        text = "kelime\u00adtest"
        result = _normalize_text(text)
        assert "\u00ad" not in result
        assert "kelimetest" in result

    def test_zero_width_space_replaced(self) -> None:
        text = "A\u200bB"
        result = _normalize_text(text)
        assert "\u200b" not in result

    def test_hyphenated_line_break_joined(self) -> None:
        text = "kelime-\ndevamı"
        result = _normalize_text(text)
        assert "kelimedevamı" in result

    def test_extra_spaces_collapsed(self) -> None:
        text = "A     B   C"
        result = _normalize_text(text)
        assert "A B C" == result

    def test_newline_whitespace_collapsed(self) -> None:
        text = "  \n  "
        result = _normalize_text(text)
        # After strip, should be empty or minimal
        assert result == ""


class TestLoadTextFromPath:
    """Unit tests for load_text_from_path."""

    def test_load_txt_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Test metin içeriği.")
            f.flush()
            result = load_text_from_path(f.name)
            assert "Test metin içeriği." in result
            Path(f.name).unlink()

    def test_load_unsupported_raises(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"data")
            f.flush()
            with pytest.raises(ValueError, match="Unsupported file type"):
                load_text_from_path(f.name)
            Path(f.name).unlink()

    def test_load_json_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write('{"key": "value"}')
            f.flush()
            result = load_text_from_path(f.name)
            assert "key" in result
            Path(f.name).unlink()
