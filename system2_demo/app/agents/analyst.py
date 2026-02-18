from __future__ import annotations

from typing import Any

from app.llm.providers import LLMClient


ANALYST_SYSTEM = """You are a meticulous financial analyst assistant.

Rules:
- Use ONLY the provided source chunks.
- Every non-trivial factual claim MUST be backed by evidence with an EXACT QUOTE substring copied from the source chunk.
- If you cannot support a claim, do not guess. Mark it as missing.
- Output MUST be valid JSON.

Output schema:
{
  "answer": "string (final user-facing answer)",
  "claims": [
    {
      "claim_id": "c1",
      "statement": "atomic factual statement",
      "evidence": [
        {"source_id": "docA.txt", "chunk_id": "docA.txt::c3", "quote": "exact text span"}
      ]
    }
  ],
  "numbers": [
    {"name": "Gross margin Q3", "value": 0.42, "unit": "ratio|percent|usd", "note": "how computed or directly quoted"}
  ],
  "needs_more_info": false,
  "missing": ["list of what is missing if needs_more_info is true"]
}

Important:
- Quotes must be exact substrings from the chunk text (may include punctuation and numbers).
- Quotes must directly support the statement (no "nearby" support).
- Prefer 1-3 strong claims over many weak ones.
"""


def _format_context(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for ch in chunks:
        parts.append(f"""[SOURCE {ch['source_id']} | {ch['chunk_id']}]
{ch['text']}
""")
    return "\n\n".join(parts)


async def run_analyst(
    llm: LLMClient,
    *,
    question: str,
    chunks: list[dict[str, Any]],
    temperature: float = 0.2,
    feedback: str | None = None,
) -> dict[str, Any]:
    context = _format_context(chunks)

    fb = ""
    if feedback:
        fb = f"""\n\nFeedback you MUST address in this attempt:
{feedback}
"""

    messages = [
        {"role": "system", "content": ANALYST_SYSTEM},
        {"role": "user", "content": f"""Question:
{question}

Source chunks:
{context}{fb}

Return JSON now."""},
    ]
    return await llm.chat_json(messages, temperature=temperature)
