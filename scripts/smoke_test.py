#!/usr/bin/env python3
"""
Live end-to-end check: monday.com -> normalise -> DuckDB -> Gemini -> answer.

    MONDAY_API_TOKEN=... GEMINI_API_KEY=... python scripts/smoke_test.py

Runs a benchmark set of founder questions and prints the SQL alongside each
answer so the numbers can be spot-checked against the Data tab.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.agent import answer_question
from agent.leadership import compute_metrics
from agent.llm import Gemini
from agent.warehouse import build_warehouse, quality_summary

QUESTIONS = [
    "How's our pipeline looking for the renewables sector this quarter?",
    "What's our total outstanding receivable and which sector is it concentrated in?",
    "Which work orders are completed but haven't been invoiced yet?",
    "What's our win rate?",
    "When did we last collect payment from each customer?",   # must refuse: not tracked
    "How much have we contracted in mining versus renewables?",
]


def main() -> int:
    t0 = time.time()
    wh = build_warehouse()
    print(f"Loaded in {time.time()-t0:.1f}s: "
          f"{len(wh.deals)} deals, {len(wh.work_orders)} work orders\n")

    q = quality_summary(wh)
    for t, info in q["tables"].items():
        print(f"[{t}] board '{info['board']}' ({info['board_id']}) rows={info['rows']} "
              f"dropped_headers={info['dropped_header_rows']}")
        print(f"     unusable: {info['unusable_columns']}")
        if info["coercion_failures"]:
            print(f"     coercion failures: {info['coercion_failures']}")
    print()

    m = compute_metrics(wh)
    print("Deterministic metrics sanity check:")
    print(f"  open deals        {m['pipeline']['open_deals']}  "
          f"value={m['pipeline']['open_value']:,.0f} ({m['pipeline']['open_value_coverage']})")
    print(f"  win rate          {m['conversion']['win_rate']:.1%}"
          if m['conversion']['win_rate'] is not None else "  win rate n/a")
    print(f"  contracted        {m['revenue']['contracted_incl_gst']:,.0f}")
    print(f"  billed            {m['revenue']['billed_incl_gst']:,.0f}")
    print(f"  collected         {m['revenue']['collected_incl_gst']:,.0f}")
    print(f"  outstanding AR    {m['revenue']['outstanding_incl_gst']:,.0f}")
    print(f"  risks flagged     {len(m['risks'])}\n")

    llm = Gemini()
    failures = 0
    for i, question in enumerate(QUESTIONS, 1):
        print("=" * 78)
        print(f"Q{i}: {question}")
        t = time.time()
        turn = answer_question(wh, question, llm=llm)
        print(f"[{turn.action}, {time.time()-t:.1f}s]")
        if turn.sql:
            print("SQL:", " ".join(turn.sql.split())[:400])
        if turn.assumptions:
            print("Assumptions:", turn.assumptions)
        if turn.attempts:
            print(f"Self-corrected after {len(turn.attempts)} failed attempt(s)")
        if turn.action == "error":
            failures += 1
            print("ERROR:", turn.error)
        print("-" * 78)
        print(turn.answer or turn.clarify or "")
        print()

    print("=" * 78)
    print(f"{len(QUESTIONS) - failures}/{len(QUESTIONS)} questions answered without error.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
