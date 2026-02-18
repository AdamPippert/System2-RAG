from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.rag.text import normalize_ws


@dataclass
class VerificationResult:
    ok: bool
    errors: list[str]


def verify_quotes(claims: list[dict[str, Any]], chunk_lookup: dict[str, str]) -> VerificationResult:
    """
    Deterministic verification:
    - Each evidence item must have a quote that appears in the retrieved chunk text.
    - We normalize whitespace to be robust to line breaks.
    """
    errors: list[str] = []
    for c in claims:
        claim_id = c.get("claim_id", "?")
        for ev in c.get("evidence", []):
            chunk_id = ev.get("chunk_id")
            quote = ev.get("quote", "")
            if not chunk_id or chunk_id not in chunk_lookup:
                errors.append(f"Claim {claim_id}: evidence references missing chunk_id={chunk_id!r}")
                continue
            chunk_text = chunk_lookup[chunk_id]

            qn = normalize_ws(quote)
            tn = normalize_ws(chunk_text)

            if not qn:
                errors.append(f"Claim {claim_id}: empty quote for chunk {chunk_id}")
                continue

            if qn not in tn:
                # show short diagnostics
                errors.append(
                    f"Claim {claim_id}: quote not found verbatim in chunk {chunk_id} (after whitespace-normalization)."
                )
    return VerificationResult(ok=(len(errors) == 0), errors=errors)
