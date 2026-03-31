from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from fastapi import UploadFile


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = cleaned.replace("\u00ad", "")
    cleaned = cleaned.replace("\u200b", " ").replace("\ufeff", " ")
    cleaned = cleaned.replace("˘", "").replace("ˆ", "").replace("¸", "")
    cleaned = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", cleaned)
    cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def load_text_from_path(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown", ".log", ".csv", ".json"}:
        return _normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as exc:
            raise RuntimeError("PDF reading requires pypdf package.") from exc
        reader = PdfReader(str(path))
        return _normalize_text("\n".join((page.extract_text() or "") for page in reader.pages))
    raise ValueError(f"Unsupported file type: {suffix}")


async def load_text_from_upload(upload: UploadFile) -> str:
    filename = upload.filename or ""
    suffix = Path(filename).suffix.lower()
    data = await upload.read()

    if suffix in {".txt", ".md", ".markdown", ".log", ".csv", ".json"}:
        return _normalize_text(data.decode("utf-8", errors="ignore"))

    if suffix == ".pdf":
        try:
            from io import BytesIO
            from pypdf import PdfReader
        except Exception as exc:
            raise RuntimeError("PDF reading requires pypdf package.") from exc
        reader = PdfReader(BytesIO(data))
        return _normalize_text("\n".join((page.extract_text() or "") for page in reader.pages))

    raise ValueError(f"Unsupported uploaded file type: {suffix}")
