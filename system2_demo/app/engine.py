from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import settings
from app.llm.providers import LLMClient
from app.rag.index import SimpleIndex
from app.verify import verify_quotes
from app.agents.analyst import run_analyst
from app.agents.critic import run_critic


@dataclass
class AnswerPackage:
    answer: str
    claims: list[dict[str, Any]]
    numbers: list[dict[str, Any]]
    needs_more_info: bool
    missing: list[str]
    critic: dict[str, Any]
    uncertainty: dict[str, Any]
    context_chunks: list[dict[str, Any]]


def _extract_chunks(results) -> list[dict[str, Any]]:
    out = []
    for r in results:
        out.append({"chunk_id": r.chunk.chunk_id, "source_id": r.chunk.source_id, "text": r.chunk.text, "score": r.score})
    return out


def _make_chunk_lookup(chunks: list[dict[str, Any]]) -> dict[str, str]:
    return {c["chunk_id"]: c["text"] for c in chunks}


def _parse_numeric_values(numbers: list[dict[str, Any]]) -> list[float]:
    vals: list[float] = []
    for n in numbers or []:
        v = n.get("value")
        if isinstance(v, (int, float)):
            vals.append(float(v))
        elif isinstance(v, str):
            m = re.search(r"-?\d+(?:\.\d+)?", v.replace(",", ""))
            if m:
                vals.append(float(m.group(0)))
    return vals


def _build_feedback(verifier_errors: list[str], critic_json: dict[str, Any]) -> str:
    lines: list[str] = []
    if verifier_errors:
        lines.append("Deterministic quote-verifier errors (you must fix quotes to be exact substrings):")
        for e in verifier_errors[:10]:
            lines.append(f"- {e}")
    req = critic_json.get("required_fixes") or []
    if req:
        lines.append("Critic required fixes:")
        for r in req[:10]:
            lines.append(f"- {r}")
    return "\n".join(lines).strip()


async def answer_question(index: SimpleIndex, llm: LLMClient, *, question: str) -> AnswerPackage:
    # 1) initial retrieval
    results = index.search(question, top_k=settings.top_k)
    chunks = _extract_chunks(results)

    critic_json: dict[str, Any] = {}
    analyst_json: dict[str, Any] = {}
    verifier_errors: list[str] = []

    feedback: str | None = None

    # 2) revision loop (Analyst → Verify → Critic → (expand retrieval) → Analyst ...)
    for round_i in range(settings.max_revision_rounds + 1):
        analyst_json = await run_analyst(llm, question=question, chunks=chunks, temperature=0.2, feedback=feedback)

        # Deterministic verification of quote spans
        chunk_lookup = _make_chunk_lookup(chunks)
        vr = verify_quotes(analyst_json.get("claims", []), chunk_lookup)
        verifier_errors = vr.errors

        critic_json = await run_critic(
            llm,
            question=question,
            analyst_json=analyst_json,
            chunks=chunks,
            verifier_errors=verifier_errors,
        )

        verdict = (critic_json.get("verdict") or "revise").lower()
        if vr.ok and verdict == "accept":
            break

        # build feedback for the next analyst attempt
        feedback = _build_feedback(verifier_errors, critic_json) or None

        # if we have suggested queries, expand retrieval and retry
        suggested: list[str] = []
        for issue in critic_json.get("issues", []) or []:
            q = (issue.get("suggested_query") or "").strip()
            if q:
                suggested.append(q)

        if suggested and round_i < settings.max_revision_rounds:
            extra = []
            for q in suggested[:3]:
                extra.extend(index.search(q, top_k=max(2, settings.top_k // 2)))

            # merge chunks (unique by chunk_id)
            seen = {c["chunk_id"] for c in chunks}
            for r in extra:
                cid = r.chunk.chunk_id
                if cid not in seen:
                    chunks.append({"chunk_id": cid, "source_id": r.chunk.source_id, "text": r.chunk.text, "score": r.score})
                    seen.add(cid)

    # 3) self-consistency sampling for uncertainty (focus on numeric fields)
    samples: list[dict[str, Any]] = []
    vals: list[float] = []
    for _ in range(settings.self_consistency_n):
        s = await run_analyst(llm, question=question, chunks=chunks, temperature=0.7)
        samples.append(s)
        vals.extend(_parse_numeric_values(s.get("numbers", [])))

    uncertainty: dict[str, Any] = {"status": "unknown", "n": settings.self_consistency_n}
    if vals:
        arr = np.asarray(vals, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        cv = float(std / abs(mean)) if mean != 0 else float("inf")
        uncertainty.update({"mean": mean, "std": std, "cv": cv, "all_values": [float(x) for x in arr.tolist()]})
        uncertainty["status"] = "uncertain" if cv >= settings.uncertainty_cv_threshold else "stable"
    else:
        uncertainty.update({"status": "no_numeric_signal", "all_values": []})

    return AnswerPackage(
        answer=str(analyst_json.get("answer", "")),
        claims=analyst_json.get("claims", []) or [],
        numbers=analyst_json.get("numbers", []) or [],
        needs_more_info=bool(analyst_json.get("needs_more_info", False)),
        missing=analyst_json.get("missing", []) or [],
        critic=critic_json or {},
        uncertainty=uncertainty,
        context_chunks=chunks,
    )
