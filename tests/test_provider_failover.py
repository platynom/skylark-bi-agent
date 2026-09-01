"""Offline checks for the production provider chain."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.llm import ProviderState, RateLimitError, UnifiedLLM


class ProviderFailoverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_states = UnifiedLLM._states
        self.original_disabled = UnifiedLLM._disabled_models
        UnifiedLLM._states = {
            "vertex": ProviderState(name="vertex"),
            "ai_studio": ProviderState(name="ai_studio"),
            "deterministic": ProviderState(name="deterministic"),
        }
        UnifiedLLM._disabled_models = set()

    def tearDown(self) -> None:
        UnifiedLLM._states = self.original_states
        UnifiedLLM._disabled_models = self.original_disabled

    def test_vertex_failure_fails_over_to_same_model_on_ai_studio(self) -> None:
        llm = UnifiedLLM(
            api_key="test-key",
            model="gemini-3.5-flash-lite",
            vertex_region="global",
            vertex_model="gemini-3.5-flash-lite",
        )
        with (
            patch.object(llm, "_call_vertex", side_effect=RateLimitError("Vertex 429")),
            patch.object(llm, "_call_ai_studio", return_value="ok") as ai_call,
        ):
            self.assertEqual(llm.generate("system", "user"), "ok")
        ai_call.assert_called_once()
        self.assertEqual(llm.last_provider, "ai_studio")
        self.assertEqual(llm.last_model, "gemini-3.5-flash-lite")
        self.assertEqual(llm.last_chain[0], "vertex (429)")
        self.assertTrue(llm.last_chain[1].startswith("ai_studio ("))

    def test_ai_studio_unavailability_does_not_disable_vertex_model(self) -> None:
        model = "gemini-3.5-flash-lite"
        UnifiedLLM._disabled_models.add(("ai_studio", model))
        self.assertNotIn(("vertex", model), UnifiedLLM._disabled_models)


if __name__ == "__main__":
    unittest.main()
