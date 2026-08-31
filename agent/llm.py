"""
Multi-provider LLM client with Vertex AI primary, Gemini AI Studio secondary,
circuit breakers, fast failover, and strict request deadlines.

Target provider chain (3 legs):
  1. Vertex AI           - gemini-3.5-flash-lite on the global endpoint
  2. Gemini AI Studio    - the same model on its independent quota pool
  3. Deterministic floor - built-in answer templates when narration is unavailable
"""
from __future__ import annotations

import base64
import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import requests

try:
    import google.auth
    import google.auth.transport.requests
    from google.oauth2 import service_account
    _HAS_GOOGLE_AUTH = True
except ImportError:
    _HAS_GOOGLE_AUTH = False

from . import config

LOGGER = logging.getLogger("skylark.llm")

AI_STUDIO_ENDPOINT_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMError(RuntimeError):
    pass


class RateLimitError(LLMError):
    pass


class ModelUnavailableError(LLMError):
    pass


@dataclass
class ProviderState:
    name: str
    is_available: bool = True
    consecutive_failures: int = 0
    throttled_until: float = 0.0
    last_error: str | None = None
    total_calls: int = 0
    total_failures: int = 0

    @property
    def status(self) -> Literal["healthy", "throttled", "standby"]:
        now = time.time()
        if not self.is_available:
            return "standby"
        if now < self.throttled_until:
            return "throttled"
        return "healthy"

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.total_calls += 1
        self.last_error = None

    def record_failure(self, error: Exception, is_rate_limit: bool = False) -> None:
        self.total_calls += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_error = str(error)

        if self.consecutive_failures >= 3:
            # Trip circuit breaker for 60 seconds
            self.throttled_until = time.time() + 60.0
            LOGGER.warning(
                "Circuit breaker tripped for provider '%s' (consecutive_failures=%d). Throttled for 60s.",
                self.name, self.consecutive_failures,
            )


class VertexAuthManager:
    """Manages GCP credentials and caches access tokens until 5 min before expiry."""
    def __init__(self):
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expiry: float = 0.0

    def get_token(self) -> str | None:
        now = time.time()
        if self._token and (self._expiry - now) > 300:  # 5 min buffer
            return self._token

        with self._lock:
            # Double check under lock
            if self._token and (self._expiry - now) > 300:
                return self._token

            token, expiry = self._mint_token()
            if token:
                self._token = token
                self._expiry = expiry
                return token
            return None

    def _mint_token(self) -> tuple[str | None, float]:
        if not _HAS_GOOGLE_AUTH:
            return None, 0.0

        now = time.time()

        # 1. Base64 Service Account JSON in env
        sa_b64 = config.GCP_SERVICE_ACCOUNT_B64
        if sa_b64:
            try:
                sa_json = base64.b64decode(sa_b64).decode("utf-8")
                info = json.loads(sa_json)
                creds = service_account.Credentials.from_service_account_info(
                    info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                req = google.auth.transport.requests.Request()
                creds.refresh(req)
                expiry = creds.expiry.timestamp() if getattr(creds, "expiry", None) else now + 3600
                return creds.token, expiry
            except Exception as exc:
                LOGGER.warning("Failed to mint token from GCP_SERVICE_ACCOUNT_B64: %s", exc)

        # 2. Application Default Credentials (ADC) or local gcloud auth
        try:
            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            req = google.auth.transport.requests.Request()
            creds.refresh(req)
            expiry = creds.expiry.timestamp() if getattr(creds, "expiry", None) else now + 3600
            return creds.token, expiry
        except Exception as exc:
            LOGGER.debug("google.auth.default() token mint failed: %s", exc)

        return None, 0.0


@dataclass
class GenerationResult:
    text: str
    provider: str
    model: str
    latency_ms: float
    provider_chain_attempted: list[str] = field(default_factory=list)


class UnifiedLLM:
    """Multi-provider LLM router with automatic failover, circuit breakers, and timeouts."""

    _auth_mgr = VertexAuthManager()
    _states: dict[str, ProviderState] = {
        "vertex": ProviderState(name="vertex"),
        "ai_studio": ProviderState(name="ai_studio"),
        "deterministic": ProviderState(name="deterministic"),
    }
    _session: requests.Session | None = None
    # Availability is provider-specific. The same model name on AI Studio and
    # Vertex represents two independent failure domains and must not cross-disable.
    _disabled_models: set[tuple[str, str]] = set()

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        vertex_region: str | None = None,
        vertex_project: str | None = None,
        vertex_model: str | None = None,
    ):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.ai_studio_model = model or config.GEMINI_MODEL or "gemini-2.5-flash"
        self.vertex_region = vertex_region or config.VERTEX_REGION
        self.vertex_project = vertex_project or config.GCP_PROJECT
        self.vertex_model = vertex_model or config.VERTEX_MODEL

        if UnifiedLLM._session is None:
            UnifiedLLM._session = requests.Session()

        # Telemetry tracking for latest call
        self.last_provider: str = "none"
        self.last_model: str = "none"
        self.last_latency_ms: float = 0.0
        self.last_chain: list[str] = []

    @property
    def _working_model(self) -> str:
        return self.last_model

    @classmethod
    def get_provider_status(cls) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "status": state.status,
                "consecutive_failures": state.consecutive_failures,
                "throttled_remaining_sec": max(0, int(state.throttled_until - time.time())),
                "last_error": state.last_error,
                "total_calls": state.total_calls,
                "total_failures": state.total_failures,
            }
            for name, state in cls._states.items()
        }

    def _call_vertex(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        deadline: float,
    ) -> str:
        token = self._auth_mgr.get_token()
        if not token:
            raise LLMError("Vertex AI credentials not available (no token).")

        model = self.vertex_model
        if ("vertex", model) in self._disabled_models:
            raise ModelUnavailableError(f"Vertex model '{model}' was permanently disabled.")

        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        region = self.vertex_region
        host = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
        url = (
            f"https://{host}/v1/projects/{self.vertex_project}/locations/{region}/"
            f"publishers/google/models/{model}:generateContent"
        )
        timeout = max(1.0, min(20.0, deadline - time.time()))
        if time.time() >= deadline:
            raise LLMError("Request deadline exceeded before Vertex call.")
        try:
            resp = self._session.post(
                url,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=body,
                timeout=timeout,
            )
        except requests.Timeout as exc:
            raise RateLimitError(f"Vertex {region} call timed out.") from exc
        except requests.RequestException as exc:
            raise LLMError(f"Vertex {region} network failure: {exc}") from exc

        if resp.status_code == 200:
            data = resp.json()
            cands = data.get("candidates") or []
            if not cands:
                fb = (data.get("promptFeedback") or {}).get("blockReason")
                raise LLMError(f"Vertex returned no candidates{f' ({fb})' if fb else ''}.")
            parts = (cands[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts).strip()
            if not text:
                raise LLMError("Vertex returned empty response.")
            return text
        if resp.status_code == 429:
            raise RateLimitError(f"Vertex {region} rate limit (429).")
        if resp.status_code == 404:
            raise ModelUnavailableError(f"Vertex model '{model}' is unavailable on {region}.")
        if resp.status_code in (500, 502, 503, 504):
            raise LLMError(f"Vertex {region} upstream HTTP {resp.status_code}")
        raise LLMError(f"Vertex {region} HTTP {resp.status_code}: {resp.text[:200]}")

    def _call_ai_studio(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        deadline: float,
    ) -> str:
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not configured.")

        models = [self.ai_studio_model] + [m for m in config.GEMINI_FALLBACK_MODELS if m != self.ai_studio_model]

        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        last_err: Exception | None = None

        for model in models:
            if ("ai_studio", model) in self._disabled_models:
                continue

            url = AI_STUDIO_ENDPOINT_TEMPLATE.format(model=model)

            for attempt in range(2):
                timeout = max(1.0, min(10.0, deadline - time.time()))
                if time.time() >= deadline:
                    raise LLMError("Request deadline exceeded before AI Studio call.")

                try:
                    resp = self._session.post(
                        url,
                        params={"key": self.api_key},
                        json=body,
                        timeout=timeout,
                    )
                except requests.Timeout as exc:
                    last_err = exc
                    break
                except requests.RequestException as exc:
                    last_err = exc
                    if attempt == 0 and (time.time() + 1.0) < deadline:
                        time.sleep(0.2)
                        continue
                    break

                if resp.status_code == 200:
                    data = resp.json()
                    cands = data.get("candidates") or []
                    if not cands:
                        fb = (data.get("promptFeedback") or {}).get("blockReason")
                        raise LLMError(f"AI Studio returned no candidates{f' ({fb})' if fb else ''}.")
                    parts = (cands[0].get("content") or {}).get("parts") or []
                    text = "".join(p.get("text", "") for p in parts).strip()
                    if not text:
                        raise LLMError("AI Studio returned empty response.")
                    return text

                if resp.status_code == 429:
                    if attempt == 0 and (time.time() + 1.5) < deadline:
                        time.sleep(1.0 + random.uniform(0.1, 0.5))
                        continue
                    raise RateLimitError("AI Studio rate limit (429).")

                if resp.status_code == 404:
                    self._disabled_models.add(("ai_studio", model))
                    last_err = ModelUnavailableError(f"AI Studio model '{model}' unavailable.")
                    break  # try next model

                if resp.status_code in (500, 502, 503, 504):
                    last_err = LLMError(f"AI Studio HTTP {resp.status_code}")
                    if attempt == 0 and (time.time() + 1.0) < deadline:
                        time.sleep(0.4)
                        continue
                    break

                if resp.status_code == 400 and "API key" in resp.text:
                    raise LLMError("AI Studio rejected API key.")

                raise LLMError(f"AI Studio HTTP {resp.status_code}: {resp.text[:300]}")

        raise LLMError(f"AI Studio unavailable: {last_err}")

    def generate(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        json_mode: bool = False,
        force_provider: str | None = None,
        max_total_timeout: float = 30.0,
    ) -> str:
        """Executes prompt through the 3-leg provider chain with hard 20s budget."""
        start_time = time.perf_counter()
        deadline = time.time() + max_total_timeout
        chain_attempted: list[str] = []

        # Determine eligible providers based on force_provider and circuit breaker
        target = (force_provider or "auto").strip().lower()

        if target == "vertex":
            provider_plan = ["vertex"]
        elif target in ("ai_studio", "aistudio", "gemini"):
            provider_plan = ["ai_studio"]
        elif target == "deterministic":
            provider_plan = ["deterministic"]
        else:  # "auto": two independent quota pools, then the caller's deterministic floor
            if self._states["vertex"].status == "healthy":
                provider_plan = ["vertex", "ai_studio"]
            else:
                provider_plan = ["ai_studio"]

        last_error: Exception | None = None

        for provider in provider_plan:
            if time.time() >= deadline:
                break

            if provider == "vertex":
                try:
                    t0 = time.perf_counter()
                    text = self._call_vertex(
                        system, user,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                        deadline=deadline,
                    )
                    dt = time.perf_counter() - t0
                    self._states["vertex"].record_success()
                    chain_attempted.append(f"vertex ({dt:.2f}s)")
                    self.last_provider = "vertex"
                    self.last_model = self.vertex_model
                    self.last_latency_ms = round(dt * 1000, 1)
                    self.last_chain = chain_attempted
                    return text
                except Exception as exc:
                    is_429 = isinstance(exc, RateLimitError) or "429" in str(exc)
                    self._states["vertex"].record_failure(exc, is_rate_limit=is_429)
                    tag = "429" if is_429 else "error"
                    chain_attempted.append(f"vertex ({tag})")
                    last_error = exc
                    LOGGER.info("Vertex attempt failed (%s); advancing to next provider.", exc)
                    continue

            elif provider == "ai_studio":
                try:
                    t0 = time.perf_counter()
                    text = self._call_ai_studio(
                        system, user,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                        deadline=deadline,
                    )
                    dt = time.perf_counter() - t0
                    self._states["ai_studio"].record_success()
                    chain_attempted.append(f"ai_studio ({dt:.2f}s)")
                    self.last_provider = "ai_studio"
                    self.last_model = self.ai_studio_model
                    self.last_latency_ms = round(dt * 1000, 1)
                    self.last_chain = chain_attempted
                    return text
                except Exception as exc:
                    is_429 = isinstance(exc, RateLimitError) or "429" in str(exc)
                    self._states["ai_studio"].record_failure(exc, is_rate_limit=is_429)
                    tag = "429" if is_429 else "error"
                    chain_attempted.append(f"ai_studio ({tag})")
                    last_error = exc
                    LOGGER.info("AI Studio attempt failed (%s); advancing.", exc)
                    continue

            elif provider == "deterministic":
                chain_attempted.append("deterministic")
                self.last_provider = "deterministic"
                self.last_model = "template"
                self.last_latency_ms = 0.1
                self.last_chain = chain_attempted
                raise LLMError("Deterministic provider forced.")

        # If all LLM providers failed or exhausted deadline
        self.last_provider = "deterministic"
        self.last_model = "none"
        self.last_latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        self.last_chain = chain_attempted
        raise LLMError(f"All LLM providers failed (attempted: {chain_attempted}): {last_error}")

    def generate_json(self, system: str, user: str, **kw) -> dict:
        raw = self.generate(system, user, json_mode=True, **kw)
        return parse_json(raw)


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


# Backward-compatible alias for existing imports
Gemini = UnifiedLLM
