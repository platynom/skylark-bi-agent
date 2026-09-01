import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from agent.agent import AgentTurn
from scripts.bench_providers import (
    BenchCase,
    ExactProviderLLM,
    cache_key,
    classify_case,
    summarize,
)
from scripts.compare_runs import disagreements, expand_paths
from scripts.probe_models import probe_one
from tests.test_questions import _validate_overlap


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    def post(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


class BenchmarkToolingTests(unittest.TestCase):
    def test_probe_makes_exactly_one_call_and_reports_http_class(self) -> None:
        session = FakeSession(FakeResponse(429, {"error": {"status": "RESOURCE_EXHAUSTED"}}))
        row = probe_one(
            session,
            provider="ai_studio",
            model="gemini-test",
            endpoint="v1beta",
            api_key="secret",
        )
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(row["status"], "FAIL")
        self.assertEqual(row["error_class"], "RESOURCE_EXHAUSTED")

    def test_exact_ai_studio_client_never_falls_back_models(self) -> None:
        response = FakeResponse(
            200,
            {"candidates": [{"content": {"parts": [{"text": '{"action":"unsupported"}'}]}}]},
        )
        session = FakeSession(response)
        llm = ExactProviderLLM("ai_studio", "gemini-exact", "v1beta")
        llm.api_key = "secret"
        llm._session = session
        text = llm.generate("system", "user", json_mode=True)
        self.assertIn("unsupported", text)
        self.assertEqual(len(session.calls), 1)
        self.assertIn("/gemini-exact:generateContent", session.calls[0][0])

    def test_cache_key_invalidates_every_requested_dimension(self) -> None:
        base = cache_key("q", "vertex", "m1", "global")
        self.assertNotEqual(base, cache_key("different", "vertex", "m1", "global"))
        self.assertNotEqual(base, cache_key("q", "ai_studio", "m1", "global"))
        self.assertNotEqual(base, cache_key("q", "vertex", "m2", "global"))
        self.assertNotEqual(base, cache_key("q", "vertex", "m1", "us-central1"))

    def test_provider_failure_is_excluded_from_quality(self) -> None:
        case = BenchCase("tuned", 1, "q", lambda turn: None)
        turn = AgentTurn(question="q", action="error", error="All LLM providers failed: HTTP 429")
        status, _ = classify_case(
            case,
            turn,
            [{"error_class": "RateLimitError", "error": "HTTP 429"}],
        )
        self.assertEqual(status, "FAIL_PROVIDER")

    def test_semantically_correct_structural_failure_is_validator_failure(self) -> None:
        case = BenchCase("tuned", 14, "overlap", _validate_overlap)
        turn = AgentTurn(
            question="overlap",
            action="sql",
            sql="SELECT 52 AS shared_deal_names",
            result=pd.DataFrame([{"shared_deal_names": 52}]),
        )
        status, _ = classify_case(case, turn, [])
        self.assertEqual(status, "FAIL_VALIDATOR")

    def test_wrong_value_is_quality_failure(self) -> None:
        def validator(turn: AgentTurn) -> None:
            assert turn.result is not None and int(turn.result.iloc[0, 0]) == 10, "expected 10"

        case = BenchCase("heldout", 1, "q", validator)
        turn = AgentTurn(question="q", action="sql", result=pd.DataFrame([{"count": 9}]))
        status, _ = classify_case(case, turn, [])
        self.assertEqual(status, "FAIL_QUALITY")

    def test_missing_provenance_is_not_validator_failure_when_value_is_wrong(self) -> None:
        def heldout_q35(turn: AgentTurn) -> None:
            assert turn.result is not None and "item_id" in turn.result.columns, (
                "individual work order list must include item_id"
            )

        heldout_q35.__name__ = "_v_q35"
        case = BenchCase("heldout", 35, "q", heldout_q35)
        turn = AgentTurn(question="q", action="sql", result=pd.DataFrame([{"value": 1.0}]))
        status, _ = classify_case(case, turn, [])
        self.assertEqual(status, "FAIL_QUALITY")

    def test_summary_quality_denominator_excludes_infrastructure_and_validator(self) -> None:
        records = [
            {"status": "PASS", "latency_ms": 1000},
            {"status": "FAIL_QUALITY", "latency_ms": 2000},
            {"status": "FAIL_VALIDATOR", "latency_ms": 3000},
            {"status": "FAIL_PROVIDER", "latency_ms": 20_000},
        ]
        summary = summarize(records, 26_000)
        self.assertEqual(summary["quality_score"], 50.0)
        self.assertEqual(summary["quality_denominator"], 2)
        self.assertTrue(summary["latency_ms"]["p95_over_15s"])

    def test_compare_flags_answer_signature_disagreement(self) -> None:
        def run(name: str, signature: str) -> dict:
            return {
                "configuration": {"provider": name, "model": "m", "endpoint": "e"},
                "cases": [{
                    "id": "tuned:Q01",
                    "question": "q",
                    "status": "PASS",
                    "answer_signature": signature,
                    "turn": {"action": "sql", "result": [{"x": 1}]},
                }],
            }

        found = disagreements([run("vertex", "a"), run("ai_studio", "b")])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0][0], "tuned:Q01")

    def test_compare_expands_windows_literal_glob(self) -> None:
        with patch("scripts.compare_runs.glob.glob", return_value=["results/b.json", "results/a.json"]):
            self.assertEqual(
                expand_paths(["results/*.json"]),
                [Path("results/a.json"), Path("results/b.json")],
            )


if __name__ == "__main__":
    unittest.main()
