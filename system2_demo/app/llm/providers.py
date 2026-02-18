from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from app.config import settings


Message = dict[str, str]


@dataclass
class LLMResponse:
    text: str
    raw: dict[str, Any] | None = None


class LLMClient:
    def __init__(self):
        self.provider = settings.llm_provider
        if self.provider not in {"ollama", "openai"}:
            raise ValueError(f"Unsupported LLM_PROVIDER={self.provider!r}. Use 'ollama' or 'openai'.")

    async def chat_json(self, messages: list[Message], *, temperature: float = 0.2, max_tokens: int = 1200) -> dict[str, Any]:
        """
        Returns parsed JSON. If parsing fails, raises ValueError.
        """
        if self.provider == "ollama":
            txt, raw = await self._ollama_chat(messages, temperature=temperature, max_tokens=max_tokens, force_json=True)
        else:
            txt, raw = await self._openai_chat(messages, temperature=temperature, max_tokens=max_tokens, force_json=True)
        try:
            return json.loads(txt)
        except Exception as e:
            raise ValueError(f"Model did not return valid JSON. Text was: {txt[:500]}...") from e

    async def chat_text(self, messages: list[Message], *, temperature: float = 0.2, max_tokens: int = 1200) -> str:
        if self.provider == "ollama":
            txt, _ = await self._ollama_chat(messages, temperature=temperature, max_tokens=max_tokens, force_json=False)
        else:
            txt, _ = await self._openai_chat(messages, temperature=temperature, max_tokens=max_tokens, force_json=False)
        return txt

    async def _ollama_chat(self, messages: list[Message], *, temperature: float, max_tokens: int, force_json: bool) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": settings.ollama_model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
        }
        if force_json:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{settings.ollama_host}/api/chat", json=payload)
            r.raise_for_status()
            raw = r.json()
            txt = raw.get("message", {}).get("content", "")
            return txt, raw

    async def _openai_chat(self, messages: list[Message], *, temperature: float, max_tokens: int, force_json: bool) -> tuple[str, dict[str, Any]]:
        api_key = settings.openai_api_key
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set, but LLM_PROVIDER=openai.")
        model = settings.openai_model

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Lightweight JSON-enforcement: rely on instruction + post-parse.
        # If you want stricter, wire response_format with json_schema for models that support it.
        if force_json:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            raw = r.json()
            txt = raw["choices"][0]["message"]["content"]
            return txt, raw
