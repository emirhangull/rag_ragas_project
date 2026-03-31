from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    chunk_id: str
    text: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches a Markdown heading line:  # Foo  /  ## Bar  /  ### Baz
_MD_HEADING_RE = re.compile(r"(?m)^#{1,6}\s+\S")

# Matches an ALL-CAPS line that looks like a section title (≥3 letters, ≤120 chars)
_CAPS_HEADING_RE = re.compile(
    r"(?m)^(?:[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ0-9\s:–\-\.]{2,119})$"
)


def _has_section_markers(text: str) -> bool:
    """Return True when the text contains Markdown headings or ALL-CAPS section titles."""
    return bool(_MD_HEADING_RE.search(text)) or bool(_CAPS_HEADING_RE.search(text))


def _split_on_sections(text: str) -> list[str]:
    """Split text on Markdown headings **and** ALL-CAPS title lines.

    Each heading / title line is kept as the *start* of the following chunk
    (i.e. it is prepended to the content that comes after it).
    """
    # We split on newline-preceded heading patterns so the heading itself stays
    # with the text that follows it.
    pattern = re.compile(
        r"(?m)(?=^#{1,6}\s+\S)|(?=^(?:[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ0-9\s:–\-\.]{2,119})$\n)"
    )
    parts = pattern.split(text)
    return [p for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------


class RecursiveTextChunker:
    """Recursive, section-aware text chunker.

    Split strategy (highest → lowest priority):
      1. Markdown headings / ALL-CAPS section titles  (when present)
      2. Double newline  (paragraph break)
      3. Single newline
      4. Sentence boundary (. )
      5. Word boundary (space)
      6. Hard character split (fallback)
    """

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 150) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Default separator hierarchy (section separators prepended dynamically)
        self._separators = ["\n\n", "\n", ". ", " "]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_text(self, text: str) -> list[TextChunk]:
        text = self._normalize_whitespace(text)
        if not text:
            return []

        pieces = self._split_recursive(text, self._get_separators(text))
        merged = self._merge_with_overlap(pieces)
        return [TextChunk(chunk_id=f"chunk_{idx}", text=value) for idx, value in enumerate(merged)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_separators(self, text: str) -> list[str]:
        """Return separator list, prepending section-level separator when applicable."""
        if _has_section_markers(text):
            # _SECTION_ is handled specially in _split_recursive
            return ["_SECTION_"] + self._separators
        return self._separators

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            return [text[i: i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep = separators[0]

        # Special handling: section split (regex-based)
        if sep == "_SECTION_":
            if _has_section_markers(text):
                parts = _split_on_sections(text)
                if len(parts) > 1:
                    return self._process_parts(parts, sep="", separators=separators[1:])
            # No section markers found – fall through
            return self._split_recursive(text, separators[1:])

        parts = text.split(sep)
        if len(parts) == 1:
            return self._split_recursive(text, separators[1:])

        return self._process_parts(parts, sep=sep, separators=separators[1:])

    def _process_parts(
        self, parts: list[str], sep: str, separators: list[str]
    ) -> list[str]:
        """Greedy merge parts → recurse when a candidate still exceeds chunk_size."""
        current = ""
        output: list[str] = []
        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                output.extend(self._split_recursive(current, separators))
            current = part
        if current:
            output.extend(self._split_recursive(current, separators))
        return output

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if not chunks:
                chunks.append(piece)
                continue
            tail = self._word_boundary_tail(chunks[-1], self.chunk_overlap)
            candidate = f"{tail} {piece}".strip() if tail else piece
            if len(candidate) <= self.chunk_size:
                chunks.append(candidate)
                continue
            chunks.append(piece)
        return chunks

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse runs of spaces/tabs per line; keep newline structure."""
        text = text.replace("\t", " ")
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _word_boundary_tail(text: str, max_chars: int) -> str:
        """Return up to *max_chars* from the end of *text* starting at a word boundary."""
        if len(text) <= max_chars:
            return text
        tail_raw = text[-max_chars:]
        space_idx = tail_raw.find(" ")
        if space_idx == -1 or space_idx >= len(tail_raw) - 1:
            return tail_raw
        return tail_raw[space_idx + 1:]
