"""
Gemini access over the plain REST endpoint.

Deliberately no vendor SDK: the Google GenAI Python packages have churned through
several incompatible interfaces, and a hosted demo that breaks because a
transitive pin moved is a bad trade for the two convenience methods the SDK adds.
`requests` + a documented JSON contract is stable.
"""
from __future__ import annotations

import json
import re
import time

import requests

from . import config

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class LLMError(RuntimeError):
    pass


class Gemini:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model = model or config.GEMINI_MODEL
        if not self.api_key:
            raise LLMError(
                "GEMINI_API_KEY is not set. Add it to Streamlit secrets or the environment."
            )
        self._session = requests.Session()
        self._working_model: str | None = None

    def _candidates(self) -> list[str]:
        seen, out = set(), []
        for m in [self.model] + config.GEMINI_FALLBACK_MODELS:
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def generate(self, system: str, user: str, *, temperature: float = 0.1,
                 max_tokens: int = 2048, json_mode: bool = False) -> str:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        models = [self._working_model] if self._working_model else self._candidates()
        last_err: Exception | None = None

        for model in models:
            for attempt in range(3):
                try:
                    resp = self._session.post(
                        _ENDPOINT.format(model=model),
                        params={"key": self.api_key},
                        json=body,
                        timeout=90,
                    )
                except requests.RequestException as exc:
                    last_err = exc
                    time.sleep(2 ** attempt)
                    continue

                if resp.status_code == 429:
                    last_err = LLMError("Gemini rate limit (free tier). Retrying.")
                    time.sleep(5 * (attempt + 1))
                    continue
                if resp.status_code in (500, 502, 503, 504):
                    last_err = LLMError(f"Gemini HTTP {resp.status_code}")
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code == 404:
                    last_err = LLMError(f"Model '{model}' unavailable for this key.")
                    break  # try the next model
                if resp.status_code == 400 and "API key" in resp.text:
                    raise LLMError("Gemini rejected the API key. Check GEMINI_API_KEY.")
                if resp.status_code >= 400:
                    raise LLMError(f"Gemini HTTP {resp.status_code}: {resp.text[:400]}")

                data = resp.json()
                cands = data.get("candidates") or []
                if not cands:
                    fb = (data.get("promptFeedback") or {}).get("blockReason")
                    raise LLMError(f"Gemini returned no candidates{f' ({fb})' if fb else ''}.")
                parts = (cands[0].get("content") or {}).get("parts") or []
                text = "".join(p.get("text", "") for p in parts).strip()
                if not text:
                    raise LLMError("Gemini returned an empty response.")
                self._working_model = model
                return text

        raise LLMError(f"Gemini unavailable: {last_err}")

    def generate_json(self, system: str, user: str, **kw) -> dict:
        raw = self.generate(system, user, json_mode=True, **kw)
        return parse_json(raw)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json(raw: str) -> dict:
    """Tolerant JSON extraction -- models still wrap output in fences sometimes."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = _FENCE_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise LLMError(f"Could not parse model output as JSON: {raw[:300]}")
