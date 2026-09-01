import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import config
from agent.llm import ProviderState, RateLimitError, UnifiedLLM, VertexAuthManager


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

    def test_vercel_oidc_uses_wif_service_account_impersonation(self) -> None:
        class FakeCredentials:
            token = "short-lived-google-token"
            expiry = None

            def refresh(self, request) -> None:
                return None

        with (
            patch.object(config, "GCP_PROJECT_NUMBER", "1019474383903"),
            patch.object(config, "GCP_SERVICE_ACCOUNT_EMAIL", "vertex@example.iam.gserviceaccount.com"),
            patch.object(config, "GCP_WORKLOAD_IDENTITY_POOL_ID", "vercel-skylark"),
            patch.object(config, "GCP_WORKLOAD_IDENTITY_POOL_PROVIDER_ID", "vercel"),
            patch("agent.llm.identity_pool.Credentials", return_value=FakeCredentials()) as factory,
        ):
            token, expiry = VertexAuthManager()._mint_token("vercel.oidc.jwt")

        self.assertEqual(token, "short-lived-google-token")
        self.assertGreater(expiry, 0)
        kwargs = factory.call_args.kwargs
        self.assertEqual(
            kwargs["audience"],
            "//iam.googleapis.com/projects/1019474383903/locations/global/"
            "workloadIdentityPools/vercel-skylark/providers/vercel",
        )
        self.assertIn("projects/-/serviceAccounts/vertex@example.iam.gserviceaccount.com", kwargs["service_account_impersonation_url"])
        self.assertEqual(
            kwargs["subject_token_supplier"].get_subject_token(None, None),
            "vercel.oidc.jwt",
        )


if __name__ == "__main__":
    unittest.main()
