"""Held-out semantic test suite: 40 NEW founder questions never seen or tuned against.

This file serves as a blind generalization check for the BI Agent planner.
Every expected value is derived directly from the underlying Deals and Work Orders datasets.

Usage:
    python tests/test_questions_holdout.py
    python tests/test_questions_holdout.py --only="<substring>"
    python tests/test_questions_holdout.py --no-cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import tomllib

# Load secrets
secrets_file = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"
if secrets_file.is_file():
    with open(secrets_file, "rb") as f:
        for k, v in tomllib.load(f).items():
            os.environ.setdefault(k, str(v))

from agent.agent import (
    PLANNER_SYSTEM,
    AgentTurn,
    plan_and_execute,
)
from agent.llm import Gemini
from agent.warehouse import Warehouse
from tests.test_questions import (
    _warehouse,
    _row_has,
    _row_for,
    _require_sql,
    _prompt_hash,
    _execute_cached_or_live,
)

def _close(actual: float, expected: float, tol: float = 0.02, tolerance: float = 0.02) -> bool:
    t = tol if tol != 0.02 else tolerance
    return math.isclose(float(actual), float(expected), rel_tol=1e-3, abs_tol=t)


sys.stdout.reconfigure(line_buffering=True)

HOLDOUT_CACHE_PATH = Path(__file__).resolve().parent / ".holdout_eval_cache.json"


# ----------------------------------------------------------------------
# Held-Out Ground Truth Validators (Derived directly from DuckDB/Pandas)
# ----------------------------------------------------------------------

def _v_q01(turn: AgentTurn) -> None:
    # Win rate for OWNER_001: 67 won / 73 closed = 91.78%
    frame = _require_sql(turn)
    assert any(_close(v, 91.78) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 91.78% win rate for OWNER_001; got {frame.to_dict('records')}"
    )

def _v_q02(turn: AgentTurn) -> None:
    # Which owner won most deals by count: OWNER_003 (84 won)
    frame = _require_sql(turn)
    assert any("OWNER_003" in str(v) for row in frame.itertuples(index=False) for v in row), (
        f"expected OWNER_003 as top won deal count leader; got {frame.to_dict('records')}"
    )
    assert any(_close(v, 84) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 84 won deals; got {frame.to_dict('records')}"
    )

def _v_q03(turn: AgentTurn) -> None:
    # Highest won revenue owner: OWNER_003 (85,742,855.97)
    frame = _require_sql(turn)
    assert any("OWNER_003" in str(v) for row in frame.itertuples(index=False) for v in row), (
        f"expected OWNER_003; got {frame.to_dict('records')}"
    )
    assert any(_close(v, 85742855.97, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 85,742,855.97 won value; got {frame.to_dict('records')}"
    )

def _v_q04(turn: AgentTurn) -> None:
    # OWNER_001 open pipeline: 12 deals (10 valued), sum = 43,030,642.40
    frame = _require_sql(turn)
    has_val = any(_close(v, 43030642.40, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool))
    assert has_val, f"expected 43,030,642.40 open pipeline for OWNER_001; got {frame.to_dict('records')}"

def _v_q05(turn: AgentTurn) -> None:
    # Total won deal value overall: 165 won deals, sum = 95,038,944.57
    frame = _require_sql(turn)
    assert any(_close(v, 95038944.57, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 95,038,944.57 total won deal value; got {frame.to_dict('records')}"
    )

def _v_q06(turn: AgentTurn) -> None:
    # Win rate for OWNER_002: 4 won / 18 closed = 22.22%
    frame = _require_sql(turn)
    assert any(_close(v, 22.22) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 22.22% win rate for OWNER_002; got {frame.to_dict('records')}"
    )

def _v_q07(turn: AgentTurn) -> None:
    # Open deals in Negotiations stage ('F. Negotiations'): 11 deals, sum = 152,999,600.00
    frame = _require_sql(turn)
    assert any(_close(v, 152999600.00, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 152,999,600.00 in Negotiations; got {frame.to_dict('records')}"
    )

def _v_q08(turn: AgentTurn) -> None:
    # Open deals in Proposal stage ('E. Proposal/Commercials Sent'): 19 deals, sum = 169,136,094.00
    frame = _require_sql(turn)
    assert any(_close(v, 169136094.00, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 169,136,094.00 in Proposal stage; got {frame.to_dict('records')}"
    )

def _v_q09(turn: AgentTurn) -> None:
    # Total probability-weighted open pipeline: 246,490,094.94
    frame = _require_sql(turn)
    assert any(_close(v, 246490094.94, tol=1000.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 246,490,094.94 weighted open pipeline; got {frame.to_dict('records')}"
    )

def _v_q10(turn: AgentTurn) -> None:
    # Deals On Hold: 2 deals, 0 / null recorded value
    frame = _require_sql(turn)
    assert any(_close(v, 2) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 2 deals on hold; got {frame.to_dict('records')}"
    )

def _v_q11(turn: AgentTurn) -> None:
    # Lost / Dead deal value: 127 dead deals, sum = 1,522,326,989.47
    frame = _require_sql(turn)
    assert any(_close(v, 1522326989.47, tol=5000.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 1,522,326,989.47 dead deal value; got {frame.to_dict('records')}"
    )

def _v_q12(turn: AgentTurn) -> None:
    # Average won deal value: 1,484,983.51 (64 valued won deals)
    frame = _require_sql(turn)
    assert any(_close(v, 1484983.51, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 1,484,983.51 avg won deal size; got {frame.to_dict('records')}"
    )

def _v_q13(turn: AgentTurn) -> None:
    # Total unbilled amount across work orders: 123,026,438.41
    frame = _require_sql(turn)
    assert any(_close(v, 123026438.41, tol=1000.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 123,026,438.41 unbilled WO amount; got {frame.to_dict('records')}"
    )

def _v_q14(turn: AgentTurn) -> None:
    # Total billed amount across work orders: 126,719,890.72
    frame = _require_sql(turn)
    assert any(_close(v, 126719890.72, tol=1000.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 126,719,890.72 billed WO amount; got {frame.to_dict('records')}"
    )

def _v_q15(turn: AgentTurn) -> None:
    # Collection percentage of contracted WO revenue: 36.21%
    frame = _require_sql(turn)
    assert any(_close(v, 36.21) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 36.21% collection rate; got {frame.to_dict('records')}"
    )

def _v_q16(turn: AgentTurn) -> None:
    # Customer with largest outstanding balance: WOCOMPANY_010 (10,347,682.02)
    frame = _require_sql(turn)
    assert any("WOCOMPANY_010" in str(v) for row in frame.itertuples(index=False) for v in row), (
        f"expected WOCOMPANY_010; got {frame.to_dict('records')}"
    )
    assert any(_close(v, 10347682.02, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 10,347,682.02 outstanding; got {frame.to_dict('records')}"
    )

def _v_q17(turn: AgentTurn) -> None:
    # Customer with highest total contracted WO amount: WOCOMPANY_010 (67,834,773.08)
    frame = _require_sql(turn)
    assert any("WOCOMPANY_010" in str(v) for row in frame.itertuples(index=False) for v in row), (
        f"expected WOCOMPANY_010; got {frame.to_dict('records')}"
    )
    assert any(_close(v, 67834773.08, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 67,834,773.08 contracted amount; got {frame.to_dict('records')}"
    )

def _v_q18(turn: AgentTurn) -> None:
    # Outstanding receivable in Mining sector: 13,243,899.59
    frame = _require_sql(turn)
    assert any(_close(v, 13243899.59, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 13,243,899.59 Mining receivable; got {frame.to_dict('records')}"
    )

def _v_q19(turn: AgentTurn) -> None:
    # Ongoing work orders: 25 orders, sum = 79,385,739.06
    frame = _require_sql(turn)
    assert any(_close(v, 25) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 25 ongoing work orders; got {frame.to_dict('records')}"
    )
    assert any(_close(v, 79385739.06, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 79,385,739.06 ongoing contract value; got {frame.to_dict('records')}"
    )

def _v_q20(turn: AgentTurn) -> None:
    # Total contracted amount for completed work orders: 117 orders, sum = 124,333,485.49
    frame = _require_sql(turn)
    assert any(_close(v, 124333485.49, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 124,333,485.49 completed WO amount; got {frame.to_dict('records')}"
    )

def _v_q21(turn: AgentTurn) -> None:
    # Fully paid work orders with zero outstanding: 77 work orders have outstanding_incl_gst = 0.0
    frame = _require_sql(turn)
    assert any(_close(v, 77) or _close(v, 125) or _close(v, 138599252.88, tol=1000.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 77 zero-outstanding work orders; got {frame.to_dict('records')}"
    )

def _v_q22(turn: AgentTurn) -> None:
    # Contracted WO value in Construction sector: 2 orders, sum = 2,717,003.48
    frame = _require_sql(turn)
    assert any(_close(v, 2717003.48, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 2,717,003.48 Construction WO value; got {frame.to_dict('records')}"
    )

def _v_q23(turn: AgentTurn) -> None:
    # Contracted WO value in Railways: 13 orders, sum = 70,676,960.84
    frame = _require_sql(turn)
    assert any(_close(v, 70676960.84, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 70,676,960.84 Railways WO value; got {frame.to_dict('records')}"
    )

def _v_q24(turn: AgentTurn) -> None:
    # Contracted WO value in Powerline: 6 orders, sum = 8,232,862.30
    frame = _require_sql(turn)
    assert any(_close(v, 8232862.30, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 8,232,862.30 Powerline WO value; got {frame.to_dict('records')}"
    )

def _v_q25(turn: AgentTurn) -> None:
    # Open pipeline in Railways: 13 deals, sum = 52,023,792.00
    frame = _require_sql(turn)
    assert any(_close(v, 52023792.00, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 52,023,792.00 Railways open pipeline; got {frame.to_dict('records')}"
    )

def _v_q26(turn: AgentTurn) -> None:
    # Open pipeline in Powerline: 4 deals, sum = 6,324,978.00
    frame = _require_sql(turn)
    assert any(_close(v, 6324978.00, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 6,324,978.00 Powerline open pipeline; got {frame.to_dict('records')}"
    )

def _v_q27(turn: AgentTurn) -> None:
    # Won deals total value in Renewables: 54 won deals, sum = 14,154,890.62
    frame = _require_sql(turn)
    assert any(_close(v, 14154890.62, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 14,154,890.62 Renewables won deal value; got {frame.to_dict('records')}"
    )

def _v_q28(turn: AgentTurn) -> None:
    # Won deals total value in Mining: 69 won deals, sum = 16,466,056.88
    frame = _require_sql(turn)
    assert any(_close(v, 16466056.88, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 16,466,056.88 Mining won deal value; got {frame.to_dict('records')}"
    )

def _v_q29(turn: AgentTurn) -> None:
    # Compare open pipeline between Railways and Powerline: Railways = 52,023,792.00, Powerline = 6,324,978.00
    frame = _require_sql(turn)
    has_rw = any(_close(v, 52023792.00, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool))
    has_pl = any(_close(v, 6324978.00, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool))
    assert has_rw and has_pl, f"expected both Railways (52.02M) and Powerline (6.32M); got {frame.to_dict('records')}"

def _v_q30(turn: AgentTurn) -> None:
    # Sector with largest open pipeline: Tender (531,964,610.00)
    frame = _require_sql(turn)
    assert any("Tender" in str(v) for row in frame.itertuples(index=False) for v in row), (
        f"expected Tender sector; got {frame.to_dict('records')}"
    )
    assert any(_close(v, 531964610.00, tol=5000.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 531,964,610.00 pipeline in Tender; got {frame.to_dict('records')}"
    )

def _v_q31(turn: AgentTurn) -> None:
    # Won deals matching to work orders: 24 matching deal names, total WO value = 88,958,409.28
    frame = _require_sql(turn)
    assert any(_close(v, 24) or _close(v, 88958409.28, tol=1000.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 24 won matched deals or 88,958,409.28 WO value; got {frame.to_dict('records')}"
    )

def _v_q32(turn: AgentTurn) -> None:
    # Dead deals matching to work orders: 24 matching deal names, total WO value = 139,942,012.39
    frame = _require_sql(turn)
    assert any(_close(v, 24) or _close(v, 139942012.39, tol=1000.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 24 dead matched deals or 139,942,012.39 WO value; got {frame.to_dict('records')}"
    )

def _v_q33(turn: AgentTurn) -> None:
    # Open pipeline in FY25-26 Q4: 41 deals, sum = 377,074,705.17
    frame = _require_sql(turn)
    assert any(_close(v, 377074705.17, tol=5000.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 377,074,705.17 open pipeline in FY25-26 Q4; got {frame.to_dict('records')}"
    )

def _v_q34(turn: AgentTurn) -> None:
    # Open pipeline in FY24-25 Q2: 1 deal, value = 305,850,000.00
    frame = _require_sql(turn)
    assert any(_close(v, 305850000.00, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 305,850,000.00 open pipeline in FY24-25 Q2; got {frame.to_dict('records')}"
    )

def _v_q35(turn: AgentTurn) -> None:
    # Top 5 completed work orders list: must include item_id, top 1 is 67,834,773.08
    frame = _require_sql(turn)
    cols = [str(c).lower() for c in frame.columns]
    assert "item_id" in cols, f"individual work order list must include item_id; got cols={frame.columns.tolist()}"
    assert any(_close(v, 67834773.08, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected top completed WO value 67,834,773.08; got {frame.to_dict('records')}"
    )

def _v_q36(turn: AgentTurn) -> None:
    # OWNER_002 open pipeline: 6 deals, sum = 20,179,638.00
    frame = _require_sql(turn)
    assert any(_close(v, 20179638.00, tol=500.0) for row in frame.itertuples(index=False) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)), (
        f"expected 20,179,638.00 open pipeline for OWNER_002; got {frame.to_dict('records')}"
    )

def _v_q37(turn: AgentTurn) -> None:
    # Expected billing month unusable column trap
    assert turn.action in {"unsupported", "clarify"} or (
        turn.action == "sql" and any(k in str(turn.assumptions or "").lower() for k in ["unusable", "empty", "null", "not tracked"])
    ), f"expected unusable column handling for expected_billing_month; got {turn.action}, sql={turn.sql}"

def _v_q38(turn: AgentTurn) -> None:
    # Actual collection month unusable column trap
    assert turn.action in {"unsupported", "clarify"} or (
        turn.action == "sql" and any(k in str(turn.assumptions or "").lower() for k in ["unusable", "empty", "null", "not tracked"])
    ), f"expected unusable column handling for actual_collection_month; got {turn.action}, sql={turn.sql}"

def _v_q39(turn: AgentTurn) -> None:
    # Ambiguous sector query (infrastructure consulting) -> clarify or multi-sector SQL
    assert turn.action in {"clarify", "sql"}, f"expected clarify or reasoned SQL for ambiguous domain; got {turn.action}"

def _v_q40(turn: AgentTurn) -> None:
    # Out of scope domain query (monthly payroll expense)
    assert turn.action == "unsupported", f"expected 'unsupported' action for out-of-scope payroll query; got {turn.action}"


# ----------------------------------------------------------------------
# Held-Out Questions Registration (40 Cases)
# ----------------------------------------------------------------------

@dataclass
class HoldoutCase:
    question: str
    validator: Callable[[AgentTurn], None]

HOLDOUT_CASES: list[HoldoutCase] = [
    # Group 1: Deal Owner Performance & Win Rates
    HoldoutCase("What is the win rate for OWNER_001?", _v_q01),
    HoldoutCase("Which deal owner has won the most deals by count?", _v_q02),
    HoldoutCase("Which deal owner has brought in the highest total won deal value?", _v_q03),
    HoldoutCase("How many open deals does OWNER_001 currently manage and what is their total value?", _v_q04),
    HoldoutCase("What is our total won deal value across all owners?", _v_q05),
    HoldoutCase("What is the win rate for OWNER_002?", _v_q06),
    
    # Group 2: Pipeline Stages, Probability Weighting & Deal Health
    HoldoutCase("How many open opportunities are in the Negotiations stage and what is their total value?", _v_q07),
    HoldoutCase("What is the total value of open deals currently in Proposal stage?", _v_q08),
    HoldoutCase("What is our total probability-weighted pipeline for all open deals?", _v_q09),
    HoldoutCase("How many deals are on hold and what is their recorded value?", _v_q10),
    HoldoutCase("How much potential revenue was lost in dead deals?", _v_q11),
    HoldoutCase("What is the average deal value of our won opportunities?", _v_q12),
    
    # Group 3: Accounts Receivable, Billing & Cash Collections
    HoldoutCase("What is the total unbilled amount across all our work orders?", _v_q13),
    HoldoutCase("What is the total billed amount across all work orders?", _v_q14),
    HoldoutCase("What percentage of our total contracted work order revenue has been collected so far?", _v_q15),
    HoldoutCase("Which customer has the largest outstanding balance?", _v_q16),
    HoldoutCase("Which customer has the highest total contracted work order amount?", _v_q17),
    HoldoutCase("How much outstanding receivable is owed by customers in the Mining sector?", _v_q18),
    
    # Group 4: Work Order Execution & Operations
    HoldoutCase("How many work orders are ongoing and what is their total value?", _v_q19),
    HoldoutCase("What is the total contracted amount for completed work orders?", _v_q20),
    HoldoutCase("How many work orders have been fully paid with zero outstanding balance?", _v_q21),
    HoldoutCase("How much contracted work order value sits in the Construction sector?", _v_q22),
    HoldoutCase("What is our total contracted work order value in Railways?", _v_q23),
    HoldoutCase("What is our total contracted work order value in Powerline?", _v_q24),
    
    # Group 5: Sector-Specific Pipeline & Win Rate Comparisons
    HoldoutCase("How much open pipeline do we have in Railways?", _v_q25),
    HoldoutCase("What is the open pipeline value in Powerline?", _v_q26),
    HoldoutCase("What is our total won deal value in Renewables?", _v_q27),
    HoldoutCase("What is our total won deal value in Mining?", _v_q28),
    HoldoutCase("Compare open pipeline value between Railways and Powerline.", _v_q29),
    HoldoutCase("Which sector has the largest open pipeline by total value?", _v_q30),
    
    # Group 6: Cross-Board Linkage & Timing
    HoldoutCase("For deals that we won, how many unique deal names match to work orders and what is their total work order value?", _v_q31),
    HoldoutCase("For deals marked dead, how many unique deal names match to work orders and what is their total work order amount?", _v_q32),
    HoldoutCase("How much open pipeline is scheduled to close in FY25-26 Q4?", _v_q33),
    HoldoutCase("How much open pipeline is scheduled to close in FY24-25 Q2?", _v_q34),
    HoldoutCase("List our top 5 largest completed work orders by contract value.", _v_q35),
    HoldoutCase("How many open deals does OWNER_002 manage and what is their total value?", _v_q36),
    
    # Group 7: Data Quality Traps, Clarifications & Out of Scope
    HoldoutCase("Break down work orders by expected billing month.", _v_q37),
    HoldoutCase("Show collection schedule using actual collection month.", _v_q38),
    HoldoutCase("How is our infrastructure consulting business doing?", _v_q39),
    HoldoutCase("What is the monthly payroll expense for our field operations team?", _v_q40),
]


# ----------------------------------------------------------------------
# Execution Runner
# ----------------------------------------------------------------------

def _load_cache() -> dict[str, dict]:
    if not HOLDOUT_CACHE_PATH.exists():
        return {}
    try:
        with open(HOLDOUT_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cache(cache: dict[str, dict]) -> None:
    try:
        with open(HOLDOUT_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, default=str)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Held-out evaluation suite for generalization testing.")
    parser.add_argument("--only", type=str, default="", help="Run only cases matching this pattern.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass local holdout cache.")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("FAIL: GEMINI_API_KEY is required for the held-out question suite.", file=sys.stderr)
        return 2

    wh = _warehouse()
    llm = Gemini()
    ph = _prompt_hash()
    cache = _load_cache()

    selected_cases: list[tuple[int, HoldoutCase]] = []
    for idx, case in enumerate(HOLDOUT_CASES, 1):
        if args.only and args.only.lower() not in case.question.lower():
            continue
        selected_cases.append((idx, case))

    total = len(selected_cases)
    print(f"\nRunning {total} HELD-OUT evaluation cases (prompt_hash: {ph})...\n", flush=True)

    suite_started = time.perf_counter()
    passed = 0
    failures: list[dict] = []

    for run_idx, (orig_index, case) in enumerate(selected_cases, 1):
        started = time.perf_counter()
        turn, was_cached = _execute_cached_or_live(
            wh, case.question, llm, cache, ph, use_cache=not args.no_cache
        )
        if not was_cached:
            time.sleep(0.8)
        cache_tag = "cached" if was_cached else "live"
        reason: str | None = None

        try:
            case.validator(turn)
            status = "PASS"
            passed += 1
            if case.question in cache:
                cache[case.question]["last_status"] = "PASS"
        except AssertionError as exc:
            status = "FAIL"
            reason = str(exc)
            failures.append({
                "index": orig_index,
                "question": case.question,
                "sql": turn.sql,
                "reason": reason,
                "result": None if turn.result is None else turn.result.to_dict("records"),
            })
            if case.question in cache:
                cache[case.question]["last_status"] = "FAIL"
        except Exception as exc:
            status = "ERROR"
            reason = str(exc)
            failures.append({
                "index": orig_index,
                "question": case.question,
                "sql": turn.sql,
                "reason": reason,
                "result": None,
            })

        elapsed = time.perf_counter() - started
        print(f"[{run_idx:02d}/{total:02d}] {status} Q{orig_index:02d} ({elapsed:.2f}s [{cache_tag}]) {case.question}", flush=True)
        if status in {"FAIL", "ERROR"}:
            print(f"      SQL:    {' '.join((turn.sql or '(none)').split())[:400]}", flush=True)
            print(f"      RESULT: {None if turn.result is None else turn.result.to_dict('records')[:3]}", flush=True)
            print(f"      REASON: {reason}", flush=True)

    _save_cache(cache)
    total_elapsed = time.perf_counter() - suite_started

    print("\n" + "=" * 80)
    print(f"HELD-OUT EVALUATION COMPLETE: {passed}/{total} PASSED ({total_elapsed:.1f}s total)")
    print("=" * 80)

    if failures:
        print(f"\nFAILED CASES ({len(failures)}):")
        for f in failures:
            print(f"  - Q{f['index']:02d}: {f['question']}")
            print(f"    Reason: {f['reason']}")
            print(f"    SQL:    {f['sql']}\n")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
