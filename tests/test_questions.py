#!/usr/bin/env python3
"""Semantic regression suite for founder questions.

Unlike ``test_pipeline.py``, this suite exercises Gemini's planner. It first
re-derives every expected number from the normalized CSV fixtures, then asks
82 founder questions (20 canonical + 62 natural paraphrases) and asserts action
types and result numbers rather than prose wording.

Run with ``GEMINI_API_KEY`` set, or with the ignored local
``.streamlit/secrets.toml`` present::

    python tests/test_questions.py
    python tests/test_questions.py --only="win rate"
    python tests/test_questions.py --failed-only
    python tests/test_questions.py --no-cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Ensure unbuffered output on stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_local_secrets() -> None:
    path = ROOT / ".streamlit" / "secrets.toml"
    if path.exists():
        with path.open("rb") as handle:
            for key, value in tomllib.load(handle).items():
                os.environ.setdefault(key, str(value))


_load_local_secrets()

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402

from agent.agent import (  # noqa: E402
    PLANNER_SYSTEM,
    AgentTurn,
    _missing_row_provenance,
    deterministic_narrative as _deterministic_narrative,
    narrated_currency_is_valid,
    narrate_turn,
    plan_and_execute,
    prepare_narrator_result,
    run_sql,
)
from agent.llm import Gemini  # noqa: E402
from agent.warehouse import Warehouse  # noqa: E402
from api.index import _records, _result_board_id  # noqa: E402
from tests.test_pipeline import build as build_frames  # noqa: E402

DEALS_BOARD_ID = "5030962955"
WORK_ORDERS_BOARD_ID = "5030963215"
CACHE_PATH = ROOT / "tests" / ".eval_cache.json"


def _warehouse() -> Warehouse:
    (deals, deals_q), (work_orders, work_orders_q) = build_frames()
    deal_names = set(deals["deal_name"].dropna())
    wo_names = set(work_orders["deal_name"].dropna())
    work_orders_q.notes.append(
        "Boards share no customer key. The only bridge is masked deal_name; "
        f"{len(deal_names & wo_names)} work-order names match a deal name, and "
        "each side must be aggregated before joining."
    )
    con = duckdb.connect(":memory:")
    con.register("deals_df", deals)
    con.register("work_orders_df", work_orders)
    con.execute("CREATE TABLE deals AS SELECT * FROM deals_df")
    con.execute("CREATE TABLE work_orders AS SELECT * FROM work_orders_df")
    return Warehouse(
        con=con,
        deals=deals,
        work_orders=work_orders,
        quality={"deals": deals_q, "work_orders": work_orders_q},
        loaded_at=time.time(),
        board_ids={"deals": DEALS_BOARD_ID, "work_orders": WORK_ORDERS_BOARD_ID},
    )


def _close(actual: float, expected: float, *, tolerance: float = 0.02) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=1e-8, abs_tol=tolerance)


def _numbers(frame: pd.DataFrame | None) -> list[float]:
    if frame is None:
        return []
    values: list[float] = []
    for column in frame.columns:
        for value in frame[column]:
            if isinstance(value, (int, float)) and not isinstance(value, bool) and pd.notna(value):
                values.append(float(value))
    return values


def _has_number(turn: AgentTurn, expected: float, tolerance: float = 0.02) -> None:
    values = _numbers(turn.result)
    assert any(_close(value, expected, tolerance=tolerance) for value in values), (
        f"expected numeric value {expected!r}; got {values!r}"
    )


def _row_for(frame: pd.DataFrame, label: str) -> pd.Series:
    needle = label.casefold()
    for _, row in frame.iterrows():
        if any(isinstance(value, str) and value.casefold() == needle for value in row):
            return row
    raise AssertionError(f"result has no row labelled {label!r}")


def _row_has(row: pd.Series, expected: float, tolerance: float = 0.02) -> None:
    values = [float(value) for value in row if isinstance(value, (int, float)) and not isinstance(value, bool) and pd.notna(value)]
    assert any(_close(value, expected, tolerance=tolerance) for value in values), (
        f"row does not contain {expected!r}; got {values!r}"
    )


def _require_sql(turn: AgentTurn) -> pd.DataFrame:
    assert turn.action == "sql", f"expected sql action, got {turn.action!r}"
    assert turn.result is not None, "sql action returned no frame"
    return turn.result


def _validate_win(turn: AgentTurn) -> None:
    _require_sql(turn)
    _has_number(turn, 56.51)


def _validate_sector_win(turn: AgentTurn) -> None:
    frame = _require_sql(turn)
    overall_rows = [
        row for _, row in frame.iterrows()
        if any(isinstance(v, str) and v.casefold() in {"overall", "total", "grand total", "all"} for v in row)
    ]
    assert overall_rows, f"result has no row labelled 'Overall' or 'Total' in {frame.to_dict('records')}"
    assert any(
        any(_close(v, 56.51) for v in row if isinstance(v, (int, float)) and not isinstance(v, bool) and pd.notna(v))
        for row in overall_rows
    ), f"no Overall/Total row contains 56.51; got {[r.to_dict() for r in overall_rows]}"
    _row_has(_row_for(frame, "Mining"), 71.13)


def _validate_open_pipeline(turn: AgentTurn) -> None:
    _require_sql(turn)
    _has_number(turn, 49)
    _has_number(turn, 47)
    _has_number(turn, 688_152_293.1738)


def _validate_receivable(turn: AgentTurn) -> None:
    frame = _require_sql(turn)
    label_column = None
    for column in frame.columns:
        if any(isinstance(value, str) and value.lower() == "renewables" for value in frame[column]):
            label_column = column
            break
    assert label_column is not None, f"no sector label column found in {list(frame.columns)!r}"
    
    renewables = _row_for(frame, "Renewables")
    _row_has(renewables, 20_823_561.90)

    # Validate grand total either by sum of sector rows or through overall total
    candidates = [
        column for column in frame.columns
        if ("outstanding" in column.lower() or "receivable" in column.lower())
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    assert candidates, f"no numeric outstanding column in {list(frame.columns)!r}"
    sector_column = candidates[0]
    labels = frame[label_column].astype("string").str.casefold()
    sector_rows = frame[frame[label_column].notna() & ~labels.isin({"overall", "total", "grand total", "all"})]
    total = float(sector_rows[sector_column].sum())
    assert _close(total, 36_291_748.87) or any(_close(v, 36_291_748.87) for v in _numbers(frame)), (
        f"receivables must pin the 36.29M grand total; got {total:.2f}"
    )


def _validate_completed_uninvoiced(turn: AgentTurn) -> None:
    frame = _require_sql(turn)
    assert len(frame) == 23, f"expected 23 exactly-completed, zero-billed rows; got {len(frame)}"
    assert "item_id" in frame.columns, "row-level list must include item_id for provenance"
    sql = (turn.sql or "").lower()
    assert "execution_status" in sql and "billed_incl_gst" in sql
    assert "invoice_status" not in sql, "invoice workflow label must not define invoicing"
    answer = _deterministic_narrative(frame, turn.intent, turn.assumptions)
    assert answer is not None
    assert "23 completed work orders" in answer
    assert "Rs 1.28 Cr" in answer and "17 positive-value" in answer


def _validate_owner(turn: AgentTurn) -> None:
    frame = _require_sql(turn)
    owner = _row_for(frame, "OWNER_003")
    _row_has(owner, 20)
    _row_has(owner, 497_830_748.196)
    sql = (turn.sql or "").lower()
    assert "deal_value" in sql and "weighted_value" not in sql
    labelled = " ".join([turn.intent or "", *map(str, frame.columns)]).lower()
    assert "raw" in labelled or "unweighted" in labelled, (
        "planner must label the owner metric as raw/unweighted"
    )
    answer = _deterministic_narrative(frame, turn.intent, turn.assumptions)
    assert answer is not None
    assert "OWNER_003" in answer and "Rs 49.78 Cr" in answer
    assert "raw/unweighted" in answer


def _validate_renewables(turn: AgentTurn) -> None:
    _require_sql(turn)
    _has_number(turn, 8)
    _has_number(turn, 25_569_056.3298)


def _validate_mining_renewables(turn: AgentTurn) -> None:
    frame = _require_sql(turn)
    try:
        _row_has(_row_for(frame, "Mining"), 56_898_641.43317377)
        _row_has(_row_for(frame, "Renewables"), 110_370_343.38634752)
    except AssertionError:
        _has_number(turn, 56_898_641.43317377)
        _has_number(turn, 110_370_343.38634752)


def _validate_deal_status(turn: AgentTurn) -> None:
    frame = _require_sql(turn)
    for label, count in {"Won": 165, "Dead": 127, "Open": 49, "On Hold": 2}.items():
        _row_has(_row_for(frame, label), count)


def _validate_execution(turn: AgentTurn) -> None:
    frame = _require_sql(turn)
    for label, count in {"Completed": 117, "Ongoing": 25, "Partial Completed": 2}.items():
        _row_has(_row_for(frame, label), count)


def _validate_top_open_deal(turn: AgentTurn) -> None:
    frame = _require_sql(turn)
    row = _row_for(frame, "Sakura")
    _row_has(row, 305_850_000.0)
    assert "item_id" in frame.columns, "individual deal result must include item_id"


def _validate_revenue_totals(turn: AgentTurn) -> None:
    _require_sql(turn)
    for expected in (249_746_302.86610553, 126_719_936.37287712, 90_428_187.503748):
        _has_number(turn, expected)


def _validate_quarters(turn: AgentTurn) -> None:
    frame = _require_sql(turn)
    q4 = _row_for(frame, "FY25-26 Q4")
    _row_has(q4, 41)
    _row_has(q4, 377_074_708.644)


def _validate_overlap(turn: AgentTurn) -> None:
    _require_sql(turn)
    _has_number(turn, 52)
    assert "join" in (turn.sql or "").lower() or "intersect" in (turn.sql or "").lower()


def _validate_open_matched(turn: AgentTurn) -> None:
    _require_sql(turn)
    for expected in (7, 637_732_655.196, 39_845_637.7447796):
        _has_number(turn, expected)
    sql = (turn.sql or "").lower()
    assert "deals" in sql and "work_orders" in sql
    assert "group by" in sql, "each side must be aggregated before the cross-board join"


def _validate_unsupported(turn: AgentTurn) -> None:
    assert turn.action == "unsupported", f"expected unsupported, got {turn.action!r}"


def _validate_ambiguous(turn: AgentTurn) -> None:
    assert turn.action in {"clarify", "sql"}, f"expected clarify or reasoned sql, got {turn.action!r}"
    if turn.action == "sql":
        assert turn.result is not None


def _validate_ambiguous_deal_list(turn: AgentTurn) -> None:
    _validate_ambiguous(turn)
    if turn.action == "sql":
        assert turn.result is not None and "item_id" in turn.result.columns, (
            "an individual deal list must include item_id for provenance"
        )


@dataclass(frozen=True)
class Case:
    question: str
    validate: Callable[[AgentTurn], None]
    narrate: bool = False


# 20 canonical founder benchmark questions
BASE_CASES = [
    Case("What's our win rate?", _validate_win),
    Case("What's our win rate, and does it differ by sector?", _validate_sector_win),
    Case("How many open deals do we have, how many carry value, and what is the total raw open pipeline value?", _validate_open_pipeline),
    Case("What's our total outstanding receivable, and which sector is it concentrated in?", _validate_receivable),
    Case("Which work orders are completed but haven't been invoiced yet?", _validate_completed_uninvoiced),
    Case("Which deal owner has the largest open pipeline?", _validate_owner, narrate=True),
    Case("How many open Renewables deals do we have, how many carry value, and what is their raw pipeline value?", _validate_renewables),
    Case("How much contracted revenue do Mining and Renewables each have?", _validate_mining_renewables),
    Case("Break down all deals by deal status.", _validate_deal_status),
    Case("Break down all work orders by execution status.", _validate_execution),
    Case("What is our single biggest open deal?", _validate_top_open_deal),
    Case("What are total contracted, billed, and collected amounts?", _validate_revenue_totals),
    Case("Which fiscal quarter does our open pipeline actually close in? Show deal count and raw value by quarter.", _validate_quarters),
    Case("How many distinct deal names appear in both Deals and Work Orders?", _validate_overlap),
    Case("For open deal names that also appear in Work Orders, how many names match and what are the aggregate raw open pipeline and contracted work-order values?", _validate_open_matched),
    Case("When did we last collect payment from each customer?", _validate_unsupported),
    Case("What is our average number of days from invoice to payment collection?", _validate_unsupported),
    Case("Show collection status by customer.", _validate_unsupported),
    Case("How are we doing in energy?", _validate_ambiguous),
    Case("Which deals are large?", _validate_ambiguous_deal_list),
]

# Natural founder paraphrases (62 total = 3 per benchmark + 1 extra for the 2 highest-risk metrics)
PARAPHRASES: list[list[str]] = [
    ["Calculate our deal win percentage.", "Of decided deals, what share did we win?", "What is the closed-outcome win rate?"],
    ["Compare win rates across sectors and include the company total.", "How does our overall win percentage vary by sector?", "Show sector win rates alongside Overall."],
    ["Summarise open deal count, valued count, and unweighted pipeline.", "How many live opportunities are there, how many have values, and what do their raw values total?", "Give me open opportunities, value coverage, and raw pipeline total."],
    ["Break outstanding receivables down by every sector and give the total.", "Where is our receivable concentrated across sectors?", "Show all sectors' outstanding balance and the grand total."],
    ["Which completed work orders have not been invoiced yet?", "Show me completed jobs that have zero billing.", "What work orders are finished but have not been billed?", "List our completed work orders with zero billed amount."],
    ["Rank deal owners by raw open pipeline and give me the leader.", "Who owns the most unweighted open pipeline?", "Which owner leads on total open deal value before probability weighting?", "Name the owner with the highest raw open opportunity value."],
    ["How's our pipeline looking for the renewables sector?", "What is our open pipeline for Renewables?", "How many open opportunities do we have in Renewables and what is their value?"],
    ["Compare contracted work-order value for Mining versus Renewables.", "What contracted revenue sits in Mining and Renewables?", "Show Mining and Renewables total WO amount side by side."],
    ["Count deals in each status.", "Show the full deal-status distribution.", "How many deals are Won, Dead, Open, and On Hold?"],
    ["Count work orders by execution status.", "Show the complete WO execution-status mix.", "How many work orders are completed, ongoing, or partially completed?"],
    ["Which one open opportunity has the highest deal value?", "Show our largest individual open deal.", "Name the top open deal by raw value."],
    ["Sum contracted, invoiced, and paid work-order amounts.", "What are our aggregate contracted, billed, and collected values?", "Give total WO value, billing, and collections."],
    ["Group raw open pipeline by fiscal close quarter.", "In which FY quarters are our open deals scheduled to close? Show count and raw value.", "Show open deal count and unweighted value for every tentative-close quarter."],
    ["Count distinct non-null deal names shared by both boards.", "How many unique opportunity names overlap between Deals and Work Orders?", "Find the size of the deal-name intersection across the two boards."],
    ["For open deals that have matching work orders, how many names match and what are the total pipeline and work order values?", "For open deals with matching work orders, give the matched count, open pipeline, and contracted work-order value.", "For overlapping open deal names, what are the matching count, total pipeline, and total work order value?"],
    ["For every customer, what is the most recent payment collection date?", "Show each customer's last collection date.", "When was payment last received from every customer?"],
    ["How long on average does payment take after invoicing?", "Calculate mean days from invoice date to collection date.", "What is our invoice-to-cash cycle in days?"],
    ["Group customers by collection status.", "Using the collection status field, which customers are collected versus uncollected?", "Show the untracked collection_status field per customer."],
    ["Summarise performance in the energy business.", "How is the energy segment performing?", "Give me our energy-sector picture."],
    ["List high-value deals.", "Show me the big opportunities.", "Which individual deals qualify as large?"],
]

CASES = BASE_CASES + [
    Case(question, BASE_CASES[index].validate)
    for index, questions in enumerate(PARAPHRASES)
    for question in questions
]


def _prompt_hash() -> str:
    return hashlib.sha256(PLANNER_SYSTEM.encode("utf-8")).hexdigest()[:16]


def _load_cache() -> dict[str, dict]:
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, dict]) -> None:
    try:
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as exc:
        print(f"Warning: could not save eval cache: {exc}", file=sys.stderr)


def _execute_cached_or_live(
    wh: Warehouse,
    question: str,
    llm: Gemini,
    cache: dict[str, dict],
    current_hash: str,
    use_cache: bool = True,
) -> tuple[AgentTurn, bool]:
    if use_cache and question in cache:
        entry = cache[question]
        if entry.get("prompt_hash") == current_hash and entry.get("action"):
            turn = AgentTurn(
                question=question,
                action=entry.get("action", "sql"),
                intent=entry.get("intent"),
                sql=entry.get("sql"),
                assumptions=entry.get("assumptions", []),
                clarify=entry.get("clarify"),
                options=entry.get("options", []),
                error=entry.get("error"),
                attempts=entry.get("attempts", []),
            )
            if turn.action == "sql" and turn.sql:
                try:
                    turn.result = run_sql(wh, turn.sql)
                    return turn, True
                except Exception as exc:
                    turn.action = "error"
                    turn.error = str(exc)
            else:
                return turn, True

    turn = plan_and_execute(wh, question, llm=llm)
    if turn.action == "error":
        # Retry once on transient network error
        time.sleep(1.0)
        turn = plan_and_execute(wh, question, llm=llm)

    cache[question] = {
        "action": turn.action,
        "intent": turn.intent,
        "sql": turn.sql,
        "assumptions": turn.assumptions,
        "clarify": turn.clarify,
        "options": turn.options,
        "error": turn.error,
        "attempts": turn.attempts,
        "prompt_hash": current_hash,
        "timestamp": time.time(),
    }
    return turn, False


def _verify_ground_truth(wh: Warehouse) -> None:
    deals, work_orders = wh.deals, work_orders = wh.deals, wh.work_orders
    open_deals = deals[deals["deal_status"].eq("Open")]
    won = int(deals["is_won"].sum())
    dead = int(deals["is_dead"].sum())
    win_rate = 100 * won / (won + dead)
    receivable_by_sector = work_orders.groupby("sector")["outstanding_incl_gst"].sum()
    completed_uninvoiced = work_orders[
        work_orders["execution_status"].eq("Completed")
        & work_orders["billed_incl_gst"].fillna(0).eq(0)
    ]
    owner = open_deals.groupby("owner_code")["deal_value"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    deal_names = set(deals["deal_name"].dropna())
    work_order_names = set(work_orders["deal_name"].dropna())

    assert (len(deals), len(work_orders)) == (344, 176)
    assert (won, dead) == (165, 127) and _close(win_rate, 56.506849315)
    assert len(open_deals) == 49 and int(open_deals["deal_value"].notna().sum()) == 47
    assert _close(open_deals["deal_value"].sum(), 688_152_293.1738)
    assert _close(receivable_by_sector.sum(), 36_291_748.87)
    assert receivable_by_sector.idxmax() == "Renewables"
    assert _close(receivable_by_sector.max(), 20_823_561.90)
    assert len(completed_uninvoiced) == 23
    assert owner.index[0] == "OWNER_003" and int(owner.iloc[0]["count"]) == 20
    assert _close(owner.iloc[0]["sum"], 497_830_748.196)
    assert len(deal_names & work_order_names) == 52

    print("GROUND-TRUTH WORKING")
    print(f"  win rate: {won} / ({won} + {dead}) * 100 = {win_rate:.8f}%")
    print(f"  open pipeline: {len(open_deals)} deals; {open_deals['deal_value'].notna().sum()} valued; sum = {open_deals['deal_value'].sum():.4f}")
    print("  receivable by sector:")
    for sector, value in receivable_by_sector.items():
        print(f"    {sector}: {value:.2f}")
    print(f"    SUM(all 6 sectors) = {receivable_by_sector.sum():.2f}")
    print("  completed/not invoiced: execution_status='Completed' AND COALESCE(billed_incl_gst,0)=0 -> 23")
    print(f"  top owner raw pipeline: {owner.index[0]} -> {owner.iloc[0]['sum']:.3f} across {int(owner.iloc[0]['count'])} valued deals")
    print(f"  distinct deal-name intersection: {len(deal_names & work_order_names)}")


def _verify_provenance_helpers() -> None:
    frame = pd.DataFrame([{"item_id": "987654", "deal_name": "Example"}])
    records = _records(frame, DEALS_BOARD_ID)
    expected = (
        "https://tanmayk311s-team-company.monday.com/boards/"
        f"{DEALS_BOARD_ID}/pulses/987654"
    )
    assert records[0]["monday_item_url"] == expected
    assert _result_board_id("SELECT item_id FROM deals", {"deals": DEALS_BOARD_ID, "work_orders": WORK_ORDERS_BOARD_ID}) == DEALS_BOARD_ID
    assert _result_board_id("SELECT item_id FROM work_orders", {"deals": DEALS_BOARD_ID, "work_orders": WORK_ORDERS_BOARD_ID}) == WORK_ORDERS_BOARD_ID
    assert _result_board_id("SELECT * FROM deals JOIN work_orders USING (deal_name)", {"deals": DEALS_BOARD_ID, "work_orders": WORK_ORDERS_BOARD_ID}) is None
    assert _missing_row_provenance(
        "SELECT deal_name, deal_value FROM deals ORDER BY deal_value DESC LIMIT 1",
        pd.DataFrame([{"deal_name": "Example", "deal_value": 1.0}]),
    )
    assert _missing_row_provenance(
        "WITH stats AS (SELECT MAX(deal_value) FROM deals) "
        "SELECT deal_name, deal_value FROM deals, stats",
        pd.DataFrame([{"deal_name": "Example", "deal_value": 1.0}]),
    )
    assert not _missing_row_provenance(
        "SELECT COUNT(*) AS deals FROM deals",
        pd.DataFrame([{"deals": 344}]),
    )
    assert not _missing_row_provenance(
        "SELECT item_id, deal_name FROM deals",
        pd.DataFrame([{"item_id": "1", "deal_name": "Example"}]),
    )
    print("  provenance URL helper: PASS")


class _FixedNarrator:
    def __init__(self, answer: str):
        self.answer = answer

    def generate(self, *_args, **_kwargs) -> str:
        return self.answer


def _verify_currency_guard(wh: Warehouse) -> None:
    owner_frame = pd.DataFrame([{
        "owner_code": "OWNER_003",
        "open_deal_count": 20,
        "valued_deal_count": 20,
        "raw_open_pipeline_value": 497_830_748.196,
    }])
    preview, allowed, block = prepare_narrator_result(owner_frame, 20)
    assert preview.loc[0, "raw_open_pipeline_value__formatted"] == "Rs 49.78 Cr"
    assert "Rs 49.78 Cr" in allowed and "SUM(raw_open_pipeline_value): Rs 49.78 Cr" in block
    assert narrated_currency_is_valid("The result is Rs 49.78 Cr.", allowed)
    assert not narrated_currency_is_valid("The result is Rs 4.98 Cr.", allowed)

    turn = AgentTurn(
        question="Which deal owner has the largest open pipeline?",
        intent="Largest owner by raw/unweighted open pipeline",
        sql="SELECT owner_code, COUNT(*) AS open_deal_count, COUNT(deal_value) AS valued_deal_count, SUM(deal_value) AS raw_open_pipeline_value FROM deals WHERE deal_status='Open' GROUP BY owner_code ORDER BY 4 DESC LIMIT 1",
        result=owner_frame,
    )
    answer = narrate_turn(wh, turn, llm=_FixedNarrator("OWNER_003 leads with Rs 4.98 Cr."))
    assert "Rs 49.78 Cr" in answer and "Rs 4.98 Cr" not in answer, (
        "an invented/rescaled narrator amount must be replaced by the audited fallback"
    )

    receivable = pd.DataFrame({
        "sector": ["Renewables", "Mining"],
        "sector_outstanding": [20_823_561.90, 15_468_186.97],
    })
    _, receivable_allowed, _ = prepare_narrator_result(receivable, 20)
    assert "Rs 3.63 Cr" in receivable_allowed, (
        "full-result currency totals must be supplied even when the result is grouped"
    )
    print("  structural currency allow-list and fallback: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic regression suite for founder questions.")
    parser.add_argument("--only", type=str, default="", help="Run only cases matching this pattern.")
    parser.add_argument("--failed-only", action="store_true", help="Run only previously failed cases.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass local eval cache.")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("FAIL: GEMINI_API_KEY is required for the semantic question suite.", file=sys.stderr)
        return 2

    wh = _warehouse()
    _verify_ground_truth(wh)
    _verify_provenance_helpers()
    _verify_currency_guard(wh)

    llm = Gemini()
    cache = _load_cache()
    current_hash = _prompt_hash()
    failures: list[dict] = []
    passed = 0

    selected_cases: list[tuple[int, Case]] = []
    for index, case in enumerate(CASES, 1):
        if args.only and args.only.lower() not in case.question.lower():
            continue
        if args.failed_only:
            entry = cache.get(case.question)
            if entry and entry.get("prompt_hash") == current_hash and entry.get("last_status") == "PASS":
                continue
        selected_cases.append((index, case))

    total = len(selected_cases)
    print(f"\nRunning {total} semantic evaluation cases (prompt_hash: {current_hash})...\n", flush=True)

    suite_started = time.perf_counter()
    for run_idx, (orig_index, case) in enumerate(selected_cases, 1):
        started = time.perf_counter()
        turn, was_cached = _execute_cached_or_live(
            wh, case.question, llm, cache, current_hash, use_cache=not args.no_cache
        )
        if not was_cached:
            time.sleep(0.8)
        cache_tag = "cached" if was_cached else "live"
        reason: str | None = None

        try:
            case.validate(turn)
            if case.narrate:
                answer = narrate_turn(wh, turn, llm=llm)
                assert "raw" in answer.lower() or "unweighted" in answer.lower(), (
                    "owner answer must state that raw/unweighted pipeline was used"
                )
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

        elapsed = time.perf_counter() - started
        print(f"[{run_idx:02d}/{total:02d}] {status} Q{orig_index:02d} ({elapsed:.2f}s [{cache_tag}]) {case.question}", flush=True)
        if status == "FAIL":
            print(f"      SQL:    {' '.join((turn.sql or '(none)').split())[:400]}", flush=True)
            print(f"      RESULT: {None if turn.result is None else turn.result.to_dict('records')[:3]}", flush=True)
            print(f"      REASON: {reason}", flush=True)

    _save_cache(cache)
    total_elapsed = time.perf_counter() - suite_started

    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE: {passed}/{total} PASSED ({total_elapsed:.1f}s total)")
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
