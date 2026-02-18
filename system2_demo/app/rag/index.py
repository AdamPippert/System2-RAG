from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from .text import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class SimpleIndex:
    """
    Tiny RAG index:
    - Stores chunk metadata in JSONL
    - Stores embeddings in NPY
    - Retrieval is cosine similarity over numpy

    This is deliberately simple for a demo. For production, swap in a vector DB.
    """

    def __init__(self, data_dir: str, *, embed_model: str = "all-MiniLM-L6-v2"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.data_dir / "chunks.jsonl"
        self.emb_path = self.data_dir / "embeddings.npy"

        self.model = SentenceTransformer(embed_model)

        self._chunks: list[Chunk] = []
        self._emb: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        self._chunks = []
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    self._chunks.append(Chunk(**obj))
        if self.emb_path.exists():
            self._emb = np.load(self.emb_path)
        else:
            self._emb = None

        # sanity: align lengths
        if self._emb is not None and len(self._chunks) != self._emb.shape[0]:
            raise RuntimeError(
                f"Index corrupted: {len(self._chunks)} chunks but embeddings shape {self._emb.shape}"
            )

    def persist(self) -> None:
        # metadata is already appended; embeddings saved as full array
        if self._emb is not None:
            np.save(self.emb_path, self._emb)

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        emb = self.model.encode(texts, normalize_embeddings=True)
        emb = np.asarray(emb, dtype=np.float32)

        # append metadata
        with open(self.meta_path, "a", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps({"chunk_id": c.chunk_id, "source_id": c.source_id, "text": c.text}) + "\n")

        # append embeddings
        if self._emb is None:
            self._emb = emb
        else:
            self._emb = np.vstack([self._emb, emb])

        self._chunks.extend(chunks)
        self.persist()

    def search(self, query: str, *, top_k: int = 6) -> list[SearchResult]:
        if self._emb is None or not self._chunks:
            return []
        q = self.model.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype=np.float32)[0]
        # cosine similarity because embeddings normalized
        scores = self._emb @ q
        idx = np.argsort(-scores)[:top_k]
        results: list[SearchResult] = []
        for i in idx:
            results.append(SearchResult(chunk=self._chunks[int(i)], score=float(scores[int(i)])))
        return results
