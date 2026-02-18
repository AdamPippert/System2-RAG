from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from pypdf import PdfReader


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n\n".join(parts)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def normalize_ws(s: str) -> str:
    # canonical whitespace for deterministic quote checking
    return re.sub(r"\s+", " ", s).strip()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    text: str


def chunk_text(text: str, *, source_id: str, chunk_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # basic cleanup to avoid pathological empty chunks
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    chunks: list[Chunk] = []
    i = 0
    n = len(text)
    c = 0

    while i < n:
        j = min(n, i + chunk_chars)
        # try to end on a sentence boundary if possible
        window = text[i:j]
        m = re.search(r"(.*[\.!\?])\s", window[::-1])
        # reversed search is messy; keep it simple: prefer last double newline or period near end
        cut = None
        for pat in ["\n\n", ". ", "; ", ", "]:
            idx = window.rfind(pat)
            if idx != -1 and idx > int(chunk_chars * 0.6):
                cut = idx + len(pat)
                break
        if cut is not None:
            j = i + cut

        chunk = text[i:j].strip()
        if chunk:
            chunks.append(Chunk(chunk_id=f"{source_id}::c{c}", source_id=source_id, text=chunk))
            c += 1

        if j >= n:
            break
        i = max(j - overlap, i + 1)

    return chunks
