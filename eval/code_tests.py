"""Executable checks for code-generation eval items."""

from __future__ import annotations

import re
from typing import Any

_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python(answer: str) -> str:
    fences = _FENCE.findall(answer or "")
    if fences:
        return fences[0].strip()
    return (answer or "").strip()


def _exec_fn(source: str, name: str) -> Any:
    scope: dict[str, Any] = {}
    exec(source, scope, scope)  # noqa: S102 — eval harness for student-generated snippets
    fn = scope.get(name)
    if not callable(fn):
        raise AssertionError(f"{name} is not defined")
    return fn


def check_rrf_score(answer: str) -> bool:
    fn = _exec_fn(extract_python(answer), "rrf_score")
    return abs(fn(1, 60) - (1.0 / 61.0)) < 1e-9 and abs(fn(0) - (1.0 / 60.0)) < 1e-9


def check_bound_text(answer: str) -> bool:
    fn = _exec_fn(extract_python(answer), "bound_text")
    out = fn("a  b   c" + "x" * 50, 10)
    return isinstance(out, str) and len(out) <= 10 and out.endswith("…")


def check_chunk_step(answer: str) -> bool:
    fn = _exec_fn(extract_python(answer), "chunk_step")
    return fn(512, 100) == 412 and fn(10, 20) == 1


CODE_TESTS = {
    "rrf_score": check_rrf_score,
    "bound_text": check_bound_text,
    "chunk_step": check_chunk_step,
}


def run_code_test(name: str | None, answer: str) -> dict[str, Any]:
    if not name:
        return {"applicable": False, "passed": None}
    checker = CODE_TESTS.get(name)
    if not checker:
        return {"applicable": False, "passed": None, "error": f"unknown test {name}"}
    try:
        passed = bool(checker(answer))
        return {"applicable": True, "passed": passed}
    except Exception as exc:  # noqa: BLE001
        return {"applicable": True, "passed": False, "error": str(exc)}
