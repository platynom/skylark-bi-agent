"""Probe configured Gemini model availability with one tiny call per endpoint.

Usage:
    python scripts/probe_models.py

The probe performs 13 calls by default: nine Vertex model/endpoint pairs and the
four AI Studio models configured in ``agent.config``. It never falls back from
the requested model or endpoint, so every row is an independent availability
measurement.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.11+ is required by the app
    tomllib = None


def _load_local_secrets() -> None:
    path = ROOT / ".streamlit" / "secrets.toml"
    if not path.is_file() or tomllib is None:
        return
    with path.open("rb") as handle:
        for key, value in tomllib.load(handle).items():
            os.environ.setdefault(key, str(value))


_load_local_secrets()

from agent import config  # noqa: E402
from agent.llm import VertexAuthManager  # noqa: E402

VERTEX_MODELS = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
)
VERTEX_ENDPOINTS = ("asia-south1", "us-central1", "global")


def ai_studio_models() -> list[str]:
    """Return configured AI Studio models once, retaining configured order."""
    return list(dict.fromkeys([config.GEMINI_MODEL, *config.GEMINI_FALLBACK_MODELS]))


def _minimal_body() -> dict[str, Any]:
    return {
        "contents": [{"role": "user", "parts": [{"text": "Reply OK"}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 4},
    }


def _error_class(response: requests.Response | None, exc: Exception | None) -> str:
    if exc is not None:
        return type(exc).__name__
    if response is None:
        return "UnknownError"
    try:
        payload = response.json()
        status = ((payload.get("error") or {}).get("status") or "").strip()
        if status:
            return status
    except (ValueError, AttributeError):
        pass
    return f"HTTP_{response.status_code}"


def _vertex_url(project: str, endpoint: str, model: str) -> str:
    host = "aiplatform.googleapis.com" if endpoint == "global" else f"{endpoint}-aiplatform.googleapis.com"
    return (
        f"https://{host}/v1/projects/{project}/locations/{endpoint}/"
        f"publishers/google/models/{model}:generateContent"
    )


def probe_one(
    session: requests.Session,
    *,
    provider: str,
    model: str,
    endpoint: str,
    token: str | None = None,
    api_key: str | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    """Make exactly one HTTP request for a provider/model/endpoint tuple."""
    started = time.perf_counter()
    response: requests.Response | None = None
    exc: Exception | None = None
    try:
        if provider == "vertex":
            if not token:
                raise RuntimeError("Vertex credentials unavailable")
            url = _vertex_url(project or "", endpoint, model)
            response = session.post(
                url,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=_minimal_body(),
                timeout=15,
            )
        else:
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY unavailable")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            response = session.post(
                url,
                params={"key": api_key},
                json=_minimal_body(),
                timeout=15,
            )
    except Exception as caught:  # each row must be reported; probing continues
        exc = caught

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    ok = response is not None and response.status_code == 200
    return {
        "provider": provider,
        "model": model,
        "endpoint": endpoint,
        "status": "OK" if ok else "FAIL",
        "latency_ms": elapsed_ms,
        "error_class": "-" if ok else _error_class(response, exc),
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = ("provider", "model", "endpoint", "status", "latency_ms", "error_class")
    widths = {header: max(len(header), *(len(str(row[header])) for row in rows)) for header in headers}
    print(" | ".join(header.ljust(widths[header]) for header in headers), flush=True)
    print("-+-".join("-" * widths[header] for header in headers), flush=True)
    for row in rows:
        print(" | ".join(str(row[header]).ljust(widths[header]) for header in headers), flush=True)


def main() -> int:
    session = requests.Session()
    token = VertexAuthManager().get_token()
    rows: list[dict[str, Any]] = []
    for model in VERTEX_MODELS:
        for endpoint in VERTEX_ENDPOINTS:
            rows.append(
                probe_one(
                    session,
                    provider="vertex",
                    model=model,
                    endpoint=endpoint,
                    token=token,
                    project=config.get("GCP_PROJECT", "project-4f0f85f8-1fbe-4abe-b7e"),
                )
            )
    for model in ai_studio_models():
        rows.append(
            probe_one(
                session,
                provider="ai_studio",
                model=model,
                endpoint="v1beta",
                api_key=config.GEMINI_API_KEY,
            )
        )
    _print_table(rows)
    return 0 if all(row["status"] == "OK" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
