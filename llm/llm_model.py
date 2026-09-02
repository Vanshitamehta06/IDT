"""Ollama SDK client wrapper compatible with ollama >= 0.2.0."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
import time

from ollama import Client

from config import (
    MAX_PROMPT_CHARS,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TIMEOUT_SECONDS,
)
from llm.prompt_builder import bound_text, compose_raw_prompt

logger = logging.getLogger(__name__)


class GenerateStats:
    def __init__(
        self,
        text: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_duration_ns: int = 0,
        eval_duration_ns: int = 0,
        error: str | None = None,
    ) -> None:
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_duration_ns = total_duration_ns
        self.eval_duration_ns = eval_duration_ns
        self.error = error

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)


def _extract_content(response: Any) -> str:
    """Parse chat responses via object attributes, not legacy dict indexing."""
    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
    if isinstance(response, dict):
        nested = response.get("message") or {}
        if isinstance(nested, dict) and isinstance(nested.get("content"), str):
            return nested["content"].strip()
    return str(response).strip()


class OllamaLLM:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.host = host or OLLAMA_HOST
        self.model = model or OLLAMA_MODEL
        self.client = Client(host=self.host, timeout=timeout or OLLAMA_TIMEOUT_SECONDS)
        self.last_call: dict[str, Any] = {}

    def ping(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama ping failed: %s", exc)
            return False

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_prompt_chars: int | None = None,
    ) -> str:
        return self.generate_with_stats(
            prompt,
            system=system,
            temperature=temperature,
            max_prompt_chars=max_prompt_chars,
        ).text

    def generate_with_stats(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_prompt_chars: int | None = None,
    ) -> GenerateStats:
        limit = max_prompt_chars or MAX_PROMPT_CHARS
        bounded = bound_text(prompt, limit)
        system_bounded = bound_text(system, 1200) if system else None
        messages: list[dict[str, str]] = []
        if system_bounded:
            messages.append({"role": "system", "content": system_bounded})
        messages.append({"role": "user", "content": bounded})
        raw_prompt = compose_raw_prompt(system_bounded, bounded)
        started = time.perf_counter()
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_ctx": OLLAMA_NUM_CTX,
                    "num_predict": OLLAMA_NUM_PREDICT,
                },
            )
            latency_ms = (time.perf_counter() - started) * 1000
            prompt_tokens = int(getattr(response, "prompt_eval_count", 0) or 0)
            completion_tokens = int(getattr(response, "eval_count", 0) or 0)
            eval_ns = int(getattr(response, "eval_duration", 0) or 0)
            tokens_per_sec = (
                (completion_tokens / (eval_ns / 1_000_000_000)) if eval_ns > 0 and completion_tokens else 0.0
            )
            stats = GenerateStats(
                text=_extract_content(response),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_duration_ns=int(getattr(response, "total_duration", 0) or 0),
                eval_duration_ns=eval_ns,
            )
            self.last_call = {
                "model": self.model,
                "raw_prompt": raw_prompt,
                "system": system_bounded,
                "user": bounded,
                "inference_latency_ms": round(latency_ms, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": stats.total_tokens,
                "tokens_per_sec": round(tokens_per_sec, 3),
                "eval_duration_ns": eval_ns,
            }
            return stats
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ollama generate_with_stats failed")
            self.last_call = {
                "model": self.model,
                "raw_prompt": raw_prompt,
                "system": system_bounded,
                "user": bounded,
                "error": str(exc),
            }
            return GenerateStats(
                text=f"The local language model is unavailable. Host={self.host} model={self.model} error={exc}",
                error=str(exc),
            )

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.generate, prompt, **kwargs)
