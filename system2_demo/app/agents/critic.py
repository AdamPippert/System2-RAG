from __future__ import annotations

from typing import Any

from app.llm.providers import LLMClient


CRITIC_SYSTEM = """You are a hostile reviewer (the "Critic") auditing an analyst answer.

Your job:
- Find missing dates, missing assumptions, contradictions, and any claim not actually supported by evidence.
- Be strict: if a claim is vague or the quote does not directly support it, flag it.
- Suggest a better retrieval query if needed.

Output MUST be valid JSON with this schema:
{
  "verdict": "accept" | "revise",
  "issues": [
    {
      "type": "missing_evidence|contradiction|ambiguous|math_error|unsupported_inference|missing_date",
      "detail": "what is wrong",
      "suggested_query": "a short search query to retrieve missing info (or empty)"
    }
  ],
  "required_fixes": ["short bullet list of what must be fixed"],
  "confidence": 0.0
}

Scoring:
- If there are any missing/weak citations, verdict should be "revise".
"""


def _format_context(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for ch in chunks:
        parts.append(f"""[SOURCE {ch['source_id']} | {ch['chunk_id']}]
{ch['text']}
""")
    return "\n\n".join(parts)


async def run_critic(llm: LLMClient, *, question: str, analyst_json: dict[str, Any], chunks: list[dict[str, Any]], verifier_errors: list[str] | None = None) -> dict[str, Any]:
    context = _format_context(chunks)
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": f"""Question:
{question}

Analyst JSON:
{analyst_json}

Verifier errors (deterministic checker):
{verifier_errors or []}

Source chunks:
{context}

Return JSON now."""},
    ]
    return await llm.chat_json(messages, temperature=0.1)
