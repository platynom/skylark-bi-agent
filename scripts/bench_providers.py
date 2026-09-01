"""Repeatable provider/model benchmark for the tuned and held-out eval suites.

Example:
    python scripts/bench_providers.py --provider vertex \
        --model gemini-3.1-flash-lite --endpoint global --suite both \
        --out results/vertex-31fl.json

The runner deliberately makes no provider, model, or endpoint fallback. Results
are written after every case so interrupted runs can be resumed safely.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import tomllib
except ImportError:  # pragma: no cover
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
from agent.agent import PLANNER_SYSTEM, AgentTurn, narrate_turn, plan_and_execute  # noqa: E402
from agent.llm import (  # noqa: E402
    LLMError,
    ModelUnavailableError,
    RateLimitError,
    UnifiedLLM,
)
from tests.test_questions import CASES, _warehouse  # noqa: E402
from tests.test_questions_holdout import HOLDOUT_CASES  # noqa: E402

LOGGER = logging.getLogger("skylark.benchmark")
STATUSES = ("PASS", "FAIL_QUALITY", "FAIL_VALIDATOR", "FAIL_PROVIDER")
PROVIDER_ERROR_MARKERS = (
    "429",
    "rate limit",
    "timed out",
    "timeout",
    "circuit breaker",
    "upstream http 5",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "network failure",
    "connectionerror",
    "connection error",
    "credentials unavailable",
    "no token",
    "all llm providers failed",
)
STRUCTURAL_MARKERS = (
    "must include item_id",
    "include item_id for provenance",
    "must label the owner metric",
    "invoice workflow label must not define",
    "each side must be aggregated before",
)
CACHE_PATH = ROOT / "results" / ".provider_bench_cache.json"


@dataclass(frozen=True)
class BenchCase:
    suite: str
    index: int
    question: str
    validator: Callable[[AgentTurn], None]
    narrate: bool = False


class ExactProviderLLM(UnifiedLLM):
    """UnifiedLLM variant pinned to exactly one provider/model/endpoint."""

    def __init__(self, provider: str, model: str, endpoint: str):
        super().__init__(
            model=model,
            vertex_region=endpoint,
            vertex_model=model,
        )
        self.bench_provider = provider
        self.bench_endpoint = endpoint
        self.call_records: list[dict[str, Any]] = []

    @staticmethod
    def _body(system: str, user: str, temperature: float, max_tokens: int, json_mode: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        return body

    @staticmethod
    def _text(response: requests.Response, provider: str) -> str:
        payload = response.json()
        candidates = payload.get("candidates") or []
        if not candidates:
            block = (payload.get("promptFeedback") or {}).get("blockReason")
            raise LLMError(f"{provider} returned no candidates{f' ({block})' if block else ''}.")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(str(part.get("text", "")) for part in parts).strip()
        if not text:
            raise LLMError(f"{provider} returned an empty response.")
        return text

    @staticmethod
    def _raise_http(provider: str, response: requests.Response) -> None:
        code = response.status_code
        detail = response.text[:300]
        if code == 429:
            raise RateLimitError(f"{provider} HTTP 429: {detail}")
        if code == 404:
            raise ModelUnavailableError(f"{provider} HTTP 404: {detail}")
        if code in (500, 502, 503, 504):
            raise LLMError(f"{provider} upstream HTTP {code}: {detail}")
        raise LLMError(f"{provider} HTTP {code}: {detail}")

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
            raise LLMError("Vertex credentials unavailable (no token).")
        endpoint = self.bench_endpoint
        host = "aiplatform.googleapis.com" if endpoint == "global" else f"{endpoint}-aiplatform.googleapis.com"
        url = (
            f"https://{host}/v1/projects/{self.vertex_project}/locations/{endpoint}/"
            f"publishers/google/models/{self.vertex_model}:generateContent"
        )
        try:
            response = self._session.post(
                url,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=self._body(system, user, temperature, max_tokens, json_mode),
                timeout=max(1.0, min(20.0, deadline - time.time())),
            )
        except requests.Timeout as exc:
            raise RateLimitError("Vertex request timed out.") from exc
        except requests.RequestException as exc:
            raise LLMError(f"Vertex network failure: {type(exc).__name__}: {exc}") from exc
        if response.status_code != 200:
            self._raise_http("Vertex", response)
        return self._text(response, "Vertex")

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
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.ai_studio_model}:generateContent"
        try:
            response = self._session.post(
                url,
                params={"key": self.api_key},
                json=self._body(system, user, temperature, max_tokens, json_mode),
                timeout=max(1.0, min(20.0, deadline - time.time())),
            )
        except requests.Timeout as exc:
            raise RateLimitError("AI Studio request timed out.") from exc
        except requests.RequestException as exc:
            raise LLMError(f"AI Studio network failure: {type(exc).__name__}: {exc}") from exc
        if response.status_code != 200:
            self._raise_http("AI Studio", response)
        return self._text(response, "AI Studio")

    def generate(self, system: str, user: str, **kwargs: Any) -> str:
        started = time.perf_counter()
        error: Exception | None = None
        try:
            kwargs["force_provider"] = self.bench_provider
            kwargs["max_total_timeout"] = min(float(kwargs.get("max_total_timeout", 20.0)), 20.0)
            return super().generate(system, user, **kwargs)
        except Exception as exc:
            error = exc
            raise
        finally:
            self.call_records.append(
                {
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "error_class": type(error).__name__ if error else None,
                    "error": str(error) if error else None,
                }
            )


def cache_key(question: str, provider: str, model: str, endpoint: str) -> str:
    material = "\0".join((question, PLANNER_SYSTEM, provider, model, endpoint))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
            return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _serialize_turn(turn: AgentTurn) -> dict[str, Any]:
    return {
        "question": turn.question,
        "action": turn.action,
        "intent": turn.intent,
        "sql": turn.sql,
        "assumptions": turn.assumptions,
        "clarify": turn.clarify,
        "options": turn.options,
        "answer": turn.answer,
        "error": turn.error,
        "attempts": turn.attempts,
        "result": None if turn.result is None else turn.result.to_dict("records"),
    }


def _deserialize_turn(question: str, payload: dict[str, Any]) -> AgentTurn:
    turn = AgentTurn(
        question=question,
        action=payload.get("action", "error"),
        intent=payload.get("intent"),
        sql=payload.get("sql"),
        assumptions=payload.get("assumptions") or [],
        clarify=payload.get("clarify"),
        options=payload.get("options") or [],
        answer=payload.get("answer"),
        error=payload.get("error"),
        attempts=payload.get("attempts") or [],
    )
    records = payload.get("result")
    if records is not None:
        turn.result = pd.DataFrame(records)
    return turn


def _numbers(turn: AgentTurn) -> list[float]:
    if turn.result is None:
        return []
    return [
        float(value)
        for value in turn.result.to_numpy().ravel()
        if isinstance(value, Real) and not isinstance(value, bool) and pd.notna(value)
    ]


def _is_provider_failure(turn: AgentTurn, calls: list[dict[str, Any]]) -> bool:
    if any(call.get("error_class") for call in calls):
        return True
    error = (turn.error or "").casefold()
    return turn.action == "error" and any(marker in error for marker in PROVIDER_ERROR_MARKERS)


def _is_validator_failure(case: BenchCase, turn: AgentTurn, reason: str) -> bool:
    """Identify assertions reached only after the expected value was established.

    The tuned validators put their SQL/provenance shape assertions after numeric
    ground-truth checks. Q35 in the held-out suite is the one exception, so its
    expected top value is verified explicitly here before calling it structural.
    """
    validator_name = getattr(case.validator, "__name__", "")
    lowered = reason.casefold()
    if validator_name == "_validate_overlap" and ("operator" in lowered or "join" in lowered or "intersect" in lowered or any(marker in lowered for marker in STRUCTURAL_MARKERS)):
        return any(abs(value - 52.0) <= 0.02 for value in _numbers(turn))
    if validator_name in {"_validate_completed_uninvoiced", "_validate_owner", "_validate_open_matched"} and not reason:
        return turn.result is not None
    if validator_name == "_validate_top_open_deal" and "item_id" in lowered:
        return any(abs(value - 305_850_000.0) <= 0.02 for value in _numbers(turn))
    if validator_name == "_validate_ambiguous_deal_list" and "item_id" in lowered:
        return turn.result is not None
    if validator_name == "_v_q35" and "item_id" in lowered:
        return any(abs(value - 67_834_773.08) <= 500 for value in _numbers(turn))
    semantic_first = {
        "_validate_completed_uninvoiced",
        "_validate_owner",
        "_validate_open_matched",
    }
    if validator_name in semantic_first and any(marker in lowered for marker in STRUCTURAL_MARKERS):
        return turn.result is not None
    return False


def classify_case(
    case: BenchCase,
    turn: AgentTurn,
    calls: list[dict[str, Any]],
    *,
    wh: Any | None = None,
    llm: UnifiedLLM | None = None,
) -> tuple[str, str | None]:
    if _is_provider_failure(turn, calls):
        detail = next((call.get("error") for call in calls if call.get("error")), None) or turn.error
        return "FAIL_PROVIDER", detail
    try:
        case.validator(turn)
        if case.narrate and wh is not None and llm is not None:
            answer = narrate_turn(wh, turn, llm=llm, force_provider=llm.bench_provider)
            if _is_provider_failure(turn, llm.call_records):
                detail = next((call.get("error") for call in llm.call_records if call.get("error")), None)
                return "FAIL_PROVIDER", detail
            assert "raw" in answer.casefold() or "unweighted" in answer.casefold(), (
                "owner answer must state that raw/unweighted pipeline was used"
            )
        return "PASS", None
    except AssertionError as exc:
        reason = str(exc)
        if _is_validator_failure(case, turn, reason):
            return "FAIL_VALIDATOR", reason or "structural validator rejected a semantically correct result"
        return "FAIL_QUALITY", reason or "quality validator failed"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.999999)))
    return ordered[rank]


def summarize(cases: Iterable[dict[str, Any]], wall_time_ms: float) -> dict[str, Any]:
    records = list(cases)
    counts = {status: sum(record.get("status") == status for record in records) for status in STATUSES}
    denominator = counts["PASS"] + counts["FAIL_QUALITY"]
    latencies = [float(record["latency_ms"]) for record in records]
    p95 = _percentile(latencies, 0.95)
    return {
        "counts": counts,
        "quality_score": round(100 * counts["PASS"] / denominator, 2) if denominator else None,
        "quality_denominator": denominator,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 1) if latencies else 0.0,
            "median": round(statistics.median(latencies), 1) if latencies else 0.0,
            "p95": round(p95, 1),
            "max": round(max(latencies), 1) if latencies else 0.0,
            "total_suite_wall": round(wall_time_ms, 1),
            "p95_over_15s": p95 > 15_000,
        },
    }


def all_cases(suite: str) -> list[BenchCase]:
    cases: list[BenchCase] = []
    if suite in {"tuned", "both"}:
        cases.extend(
            BenchCase("tuned", index, case.question, case.validate, case.narrate)
            for index, case in enumerate(CASES, 1)
        )
    if suite in {"heldout", "both"}:
        cases.extend(
            BenchCase("heldout", index, case.question, case.validator)
            for index, case in enumerate(HOLDOUT_CASES, 1)
        )
    return cases


def _new_document(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "configuration": {
            "provider": args.provider,
            "model": args.model,
            "endpoint": args.endpoint,
            "suite": args.suite,
            "planner_prompt_sha256": hashlib.sha256(PLANNER_SYSTEM.encode("utf-8")).hexdigest(),
        },
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed": False,
        "cases": [],
        "summary": {},
    }


def _same_configuration(document: dict[str, Any], args: argparse.Namespace) -> bool:
    expected = _new_document(args)["configuration"]
    return document.get("configuration") == expected


def _case_id(case: BenchCase) -> str:
    return f"{case.suite}:Q{case.index:02d}"


def run(args: argparse.Namespace) -> int:
    out_path = Path(args.out).resolve()
    existing = _load_json(out_path) if args.resume or args.failed_only else {}
    if existing and not _same_configuration(existing, args):
        raise SystemExit(f"Refusing to resume {out_path}: configuration does not match the requested run.")
    document = existing or _new_document(args)
    cache = {} if args.no_cache else _load_json(CACHE_PATH)
    previous = {_case_id_from_result(record): record for record in document.get("cases", [])}

    selected: list[BenchCase] = []
    for case in all_cases(args.suite):
        if args.only and args.only.casefold() not in case.question.casefold():
            continue
        old = previous.get(_case_id(case))
        if args.failed_only and (old is None or old.get("status") == "PASS"):
            continue
        if args.resume and old is not None and not args.failed_only:
            continue
        selected.append(case)

    if args.failed_only and not selected:
        print("No previously failed cases matched.", flush=True)
        return 0

    wh = _warehouse()
    suite_started = time.perf_counter()
    previous_wall_ms = float(document.get("wall_time_ms", 0.0))
    total = len(selected)
    for run_index, case in enumerate(selected, 1):
        key = cache_key(case.question, args.provider, args.model, args.endpoint)
        cached = None if args.no_cache else cache.get(key)
        started = time.perf_counter()
        calls: list[dict[str, Any]] = []
        if cached:
            turn = _deserialize_turn(case.question, cached["turn"])
            status = cached["status"]
            reason = cached.get("reason")
            latency_ms = float(cached.get("latency_ms", 0.0))
            calls = cached.get("llm_calls") or []
            source = "cache"
        else:
            llm = ExactProviderLLM(args.provider, args.model, args.endpoint)
            turn = plan_and_execute(wh, case.question, llm=llm, force_provider=args.provider)
            planner_calls = list(llm.call_records)
            status, reason = classify_case(case, turn, planner_calls, wh=wh, llm=llm)
            calls = list(llm.call_records)
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            source = "live"

        result = {
            "id": _case_id(case),
            "suite": case.suite,
            "index": case.index,
            "question": case.question,
            "cache_key": key,
            "status": status,
            "reason": reason,
            "latency_ms": latency_ms,
            "source": source,
            "llm_calls": calls,
            "turn": _serialize_turn(turn),
            "answer_signature": answer_signature(turn),
        }
        previous[_case_id(case)] = result
        document["cases"] = sorted(previous.values(), key=lambda row: (row["suite"], row["index"]))
        wall_ms = previous_wall_ms + (time.perf_counter() - suite_started) * 1000
        document["wall_time_ms"] = round(wall_ms, 1)
        document["summary"] = summarize(document["cases"], wall_ms)
        _atomic_json(out_path, document)
        if not args.no_cache and source == "live":
            cache[key] = result
            _atomic_json(CACHE_PATH, cache)

        label = " ".join(case.question.split()[:5])
        print(
            f"[{run_index:02d}/{total:02d}] {status:<14} {label:<38} "
            f"{latency_ms / 1000:.1f}s  {args.provider}/{args.model} [{source}]",
            flush=True,
        )

    document["completed"] = True
    document["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wall_ms = previous_wall_ms + (time.perf_counter() - suite_started) * 1000
    document["wall_time_ms"] = round(wall_ms, 1)
    document["summary"] = summarize(document["cases"], wall_ms)
    _atomic_json(out_path, document)
    summary = document["summary"]
    print(
        f"QUALITY {summary['quality_score']}% "
        f"(PASS={summary['counts']['PASS']}, QUALITY={summary['counts']['FAIL_QUALITY']}, "
        f"VALIDATOR={summary['counts']['FAIL_VALIDATOR']}, PROVIDER={summary['counts']['FAIL_PROVIDER']})",
        flush=True,
    )
    if summary["latency_ms"]["p95_over_15s"]:
        print("WARNING: p95 latency exceeds 15s; two-call questions risk the 20s budget.", flush=True)
    return 0


def _case_id_from_result(record: dict[str, Any]) -> str:
    return str(record.get("id") or f"{record.get('suite')}:Q{int(record.get('index', 0)):02d}")


def answer_signature(turn: AgentTurn) -> str:
    payload = {
        "action": turn.action,
        "result": None if turn.result is None else turn.result.to_dict("records"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one exact Gemini provider/model/endpoint configuration.")
    parser.add_argument("--provider", choices=("vertex", "ai_studio"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True, help="Vertex location or 'v1beta' for AI Studio.")
    parser.add_argument("--suite", choices=("tuned", "heldout", "both"), default="both")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", default="", metavar="PATTERN")
    parser.add_argument("--failed-only", action="store_true")
    args = parser.parse_args(argv)
    if args.provider == "ai_studio" and args.endpoint != "v1beta":
        parser.error("AI Studio endpoint must be 'v1beta'.")
    return args


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
