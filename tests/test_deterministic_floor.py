"""
Test deterministic floor fallback under total LLM provider outage.

Verifies that when all LLM providers fail (circuit-breaker tripped, 429 quota
exhausted, network down), the agent:
1. Answers standard founder questions deterministically with real SQL and rows.
2. Returns provider="deterministic" and action="sql" (or graceful "unsupported").
3. NEVER returns action="error" for provider outages.
4. Executes all registered template SQL queries cleanly against the warehouse.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from agent.agent import AgentTurn, _missing_row_provenance, answer_question, plan_and_execute
from agent.llm import LLMError, UnifiedLLM
from agent.templates import TEMPLATES, match_template
from agent.warehouse import Warehouse, run_sql
from tests.test_questions import _warehouse


class DeadLLM(UnifiedLLM):
    """Mock LLM where every network/model call raises an LLMError."""

    def __init__(self) -> None:
        super().__init__()
        self.last_provider = "none"
        self.last_model = "none"
        self.last_latency_ms = 0.0
        self.last_chain = ["vertex (429)", "ai_studio (429)"]
        self.generate_calls = 0
        self.generate_json_calls = 0

    def generate(self, *args: Any, **kwargs: Any) -> str:
        self.generate_calls += 1
        raise LLMError("Simulated LLM outage: live capacity exhausted across all providers (429).")

    def generate_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.generate_json_calls += 1
        raise LLMError("Simulated LLM outage: live capacity exhausted across all providers (429).")


# 40 realistic founder-style questions tested against the dead-LLM floor
FOUNDER_QUESTIONS: list[str] = [
    # Win rate and outcomes
    "What's our win rate?",
    "What's our win rate, and does it differ by sector?",
    "Calculate our deal win percentage across closed deals.",
    "What is the win rate for OWNER_001?",
    "What is our total won deal value across all owners?",
    "What is the average deal value of our won opportunities?",
    "Which deal owner has won the most deals by count?",
    "How much potential revenue was lost in dead deals?",
    # Pipeline & Opportunities
    "How many open deals do we have, how many carry value, and what is the total raw open pipeline value?",
    "What is our total open pipeline value?",
    "Which deal owner has the largest open pipeline?",
    "Show open pipeline breakdown by sector.",
    "Which fiscal quarter does our open pipeline actually close in?",
    "What is our single biggest open deal?",
    "List our top largest deals overall.",
    "What is our total probability-weighted pipeline for all open deals?",
    "How many open Renewables deals do we have and what is their pipeline value?",
    "How much open pipeline do we have in Mining?",
    "How much open pipeline do we have in Railways?",
    "What is the open pipeline value in Powerline?",
    "How many deals are on hold and what is their recorded value?",
    "What are the oldest open opportunities in our pipeline?",
    "Break down deals by stage in our funnel.",
    "Break down all deals by deal status.",
    # Accounts Receivable & Billing
    "What's our total outstanding receivable, and which sector is it concentrated in?",
    "What is our total outstanding receivable balance?",
    "How much outstanding receivable is owed by customers in the Mining sector?",
    "What is the total unbilled amount across all our work orders?",
    "Show unbilled balance breakdown by sector.",
    "What are total contracted, billed, and collected amounts?",
    "Which customer has the largest outstanding balance?",
    "Which customer has the highest total contracted work order amount?",
    "How many work orders have been fully paid with zero outstanding balance?",
    # Work Order Execution
    "Which work orders are completed but haven't been invoiced yet?",
    "Show me completed jobs that have zero billing.",
    "Break down all work orders by execution status.",
    "How many work orders are ongoing and what is their total value?",
    "List our top 5 largest completed work orders by contract value.",
    "Break down work orders by sector.",
    "How much contracted work order revenue was booked by fiscal quarter?",
    # Out of scope / unsupported questions (should return action="unsupported", NEVER action="error")
    "What is the monthly payroll expense for our field operations team?",
    "Break down work orders by expected billing month.",
]


def test_every_template_sql_executes_on_warehouse() -> None:
    """Assert every template SQL in the catalog executes cleanly against DuckDB."""
    wh = _warehouse()
    assert len(TEMPLATES) >= 30, f"Expected >= 30 templates, got {len(TEMPLATES)}"

    for template in TEMPLATES:
        df = run_sql(wh, template.sql)
        assert isinstance(df, pd.DataFrame), f"Template {template.id} did not return DataFrame"
        assert not df.empty, f"Template {template.id} returned 0 rows"
        assert not _missing_row_provenance(template.sql, df), (
            f"Template {template.id} returns row-level data without item_id provenance"
        )


def test_deterministic_floor_under_total_llm_outage() -> None:
    """Assert that with all LLMs dead, >= 24/40 questions return action='sql' and 0 return 'error'."""
    wh = _warehouse()
    dead_llm = DeadLLM()

    actions_count: dict[str, int] = {"sql": 0, "unsupported": 0, "clarify": 0, "error": 0}
    sql_turns: list[tuple[str, AgentTurn]] = []
    unsupported_turns: list[tuple[str, AgentTurn]] = []

    for question in FOUNDER_QUESTIONS[:40]:
        turn = answer_question(wh, question, llm=dead_llm)
        actions_count[turn.action] = actions_count.get(turn.action, 0) + 1

        # Strict invariant: action="error" must NEVER be returned for a provider outage
        assert turn.action != "error", (
            f"Expected non-error turn for question '{question}'; got error: {turn.error}"
        )

        if turn.action == "sql":
            sql_turns.append((question, turn))
            assert turn.provider == "deterministic", (
                f"Expected provider='deterministic', got '{turn.provider}' for '{question}'"
            )
            assert turn.result is not None and not turn.result.empty, (
                f"Expected non-empty result frame for '{question}'"
            )
            assert turn.intent is not None, f"Expected turn.intent to be set for '{question}'"
            assert turn.sql is not None and len(turn.sql.strip()) > 0, (
                f"Expected non-empty SQL for '{question}'"
            )
            assert turn.answer is not None and len(turn.answer.strip()) > 0, (
                f"Expected non-empty narrated answer for '{question}'"
            )
            # Caveat must mention deterministic template / model capacity
            assert any("deterministic" in a.lower() or "capacity" in a.lower() for a in turn.assumptions), (
                f"Expected deterministic assumption caveat in turn.assumptions for '{question}'"
            )
        elif turn.action == "unsupported":
            unsupported_turns.append((question, turn))
            assert turn.provider == "deterministic", (
                f"Expected provider='deterministic' on unsupported turn for '{question}'"
            )
            assert turn.answer is not None and len(turn.answer.strip()) > 0, (
                f"Expected honest unsupported message for '{question}'"
            )

    print("\n" + "=" * 70)
    print(f"DETERMINISTIC FLOOR TEST RESULTS (Total Questions: {len(FOUNDER_QUESTIONS[:40])})")
    print("=" * 70)
    print(f"Action Distribution: {actions_count}")
    print(f"  - SQL matches (floor hit):     {actions_count['sql']}")
    print(f"  - Graceful unsupported:       {actions_count['unsupported']}")
    print(f"  - Error actions (must be 0):  {actions_count['error']}")
    print("=" * 70)

    # Core target: >= 24 return action="sql" and ZERO return action="error"
    assert actions_count["error"] == 0, f"Expected 0 errors, got {actions_count['error']}"
    assert actions_count["sql"] >= 24, (
        f"Expected at least 24 SQL matches under outage, got {actions_count['sql']}"
    )


def test_narrator_skips_when_providers_throttled() -> None:
    """Verify narrate_turn skips LLM when all live providers are throttled or standby."""
    wh = _warehouse()
    turn = AgentTurn(
        question="What is our win rate?",
        action="sql",
        sql="SELECT ROUND(100.0 * SUM(CASE WHEN is_won THEN 1 ELSE 0 END) / COUNT(*), 2) AS win_rate_pct FROM deals WHERE is_won OR is_dead",
        intent="Overall win rate",
        result=pd.DataFrame([{"win_rate_pct": 56.51}]),
        assumptions=["Answer generated from deterministic query template."],
    )

    # Temporarily set all providers to throttled
    states = UnifiedLLM._states
    orig_throttled = {k: v.throttled_until for k, v in states.items()}
    try:
        for k in ("vertex", "ai_studio"):
            states[k].throttled_until = 9999999999.0

        # With all live providers throttled, narrate_turn must use the fallback
        # before attempting its prose generation call.
        dead_llm = DeadLLM()
        narrative = answer_question(wh, turn.question, llm=dead_llm)
        assert narrative.answer is not None and len(narrative.answer) > 0
        assert dead_llm.generate_json_calls == 1  # planner only
        assert dead_llm.generate_calls == 0, "Narrator should skip the LLM when every live provider is unavailable"
    finally:
        for k, val in orig_throttled.items():
            states[k].throttled_until = val


def main() -> int:
    print("Running deterministic floor validation suite...")
    test_every_template_sql_executes_on_warehouse()
    print("PASS: test_every_template_sql_executes_on_warehouse (40/40 templates valid)")
    test_deterministic_floor_under_total_llm_outage()
    print("PASS: test_deterministic_floor_under_total_llm_outage")
    test_narrator_skips_when_providers_throttled()
    print("PASS: test_narrator_skips_when_providers_throttled")
    print("\nALL DETERMINISTIC FLOOR TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
