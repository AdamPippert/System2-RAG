from __future__ import annotations

import argparse
import os
from pathlib import Path

from app.config import settings
from app.rag.index import SimpleIndex
from app.rag.text import read_pdf, read_text, chunk_text


def ingest_dir(docs_dir: str) -> None:
    index = SimpleIndex(settings.data_dir)
    p = Path(docs_dir)
    if not p.exists():
        raise SystemExit(f"Docs dir not found: {docs_dir}")

    added = 0
    for path in sorted(p.rglob("*")):
        if path.is_dir():
            continue
        ext = path.suffix.lower()
        if ext not in {".txt", ".md", ".pdf"}:
            continue

        source_id = path.name
        if ext == ".pdf":
            text = read_pdf(str(path))
        else:
            text = read_text(str(path))

        chunks = chunk_text(text, source_id=source_id, chunk_chars=settings.chunk_chars, overlap=settings.chunk_overlap)
        index.add_chunks(chunks)
        added += len(chunks)
        print(f"Ingested {path} -> {len(chunks)} chunks")

    print(f"Index size: {index.size} chunks (added {added})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="Directory containing .txt/.md/.pdf docs")
    args = ap.parse_args()
    ingest_dir(args.docs)


if __name__ == "__main__":
    main()
