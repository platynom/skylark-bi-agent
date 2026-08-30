#!/usr/bin/env python3
"""Semantic regression suite for founder questions.

Unlike ``test_pipeline.py``, this suite exercises Gemini's planner. It first
re-derives every expected number from the normalized CSV fixtures, then asks 20
questions and asserts action types and result numbers rather than prose wording.

Run with ``GEMINI_API_KEY`` set, or with the ignored local
``.streamlit/secrets.toml`` present::

    python tests/test_questions.py
"""
from __future__ import annotations

import math
import os
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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

from agent.agent import AgentTurn, _missing_row_provenance, narrate_turn, plan_and_execute  # noqa: E402
from agent.llm import Gemini  # noqa: E402
from agent.warehouse import Warehouse  # noqa: E402
from api.index import _deterministic_narrative, _records, _result_board_id  # noqa: E402
from tests.test_pipeline import build as build_frames  # noqa: E402

DEALS_BOARD_ID = "5030962955"
WORK_ORDERS_BOARD_ID = "5030963215"


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
    _row_has(_row_for(frame, "Overall"), 56.51)
    _row_has(_row_for(frame, "Mining"), 71.13)


def _validate_open_pipeline(turn: AgentTurn) -> None:
    _require_sql(turn)
    _has_number(turn, 49)
    _has_number(turn, 47)
    _has_number(turn, 688_152_293.1738)


def _validate_receivable(turn: AgentTurn) -> None:
    frame = _require_sql(turn)
    assert len(frame) == 6, f"expected all 6 sector rows, got {len(frame)}"
    candidates = [
        column for column in frame.columns
        if "outstanding" in column.lower() and pd.api.types.is_numeric_dtype(frame[column])
    ]
    assert candidates, f"no numeric outstanding column in {list(frame.columns)!r}"
    sector_column = next(
        (column for column in candidates if "sector" in column.lower()), candidates[0]
    )
    total = float(frame[sector_column].sum())
    assert _close(total, 36_291_748.87), (
        f"sum across sector rows must pin the grand total; got {total:.2f}"
    )
    renewables = _row_for(frame, "Renewables")
    _row_has(renewables, 20_823_561.90)
    assert _close(float(frame[sector_column].max()), float(renewables[sector_column]))


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
    _has_number(turn, 7)
    _has_number(turn, 25_569_056.3298)


def _validate_mining_renewables(turn: AgentTurn) -> None:
    frame = _require_sql(turn)
    _row_has(_row_for(frame, "Mining"), 56_898_641.43317377)
    _row_has(_row_for(frame, "Renewables"), 110_370_343.38634752)


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


CASES = [
    Case("What's our win rate?", _validate_win),
    Case("What's our win rate, and does it differ by sector?", _validate_sector_win),
    Case("How many open deals do we have, how many carry value, and what is the total raw open pipeline value?", _validate_open_pipeline),
    Case("What's our total outstanding receivable and which sector is it concentrated in? Show every sector.", _validate_receivable),
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


def _verify_ground_truth(wh: Warehouse) -> None:
    deals, work_orders = wh.deals, wh.work_orders
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


def main() -> int:
    if not os.environ.get("GEMINI_API_KEY"):
        print("FAIL: GEMINI_API_KEY is required for the semantic question suite.")
        return 2

    wh = _warehouse()
    _verify_ground_truth(wh)
    _verify_provenance_helpers()
    llm = Gemini()
    failures: list[str] = []

    for index, case in enumerate(CASES, 1):
        started = time.perf_counter()
        turn = plan_and_execute(wh, case.question, llm=llm)
        try:
            case.validate(turn)
            if case.narrate:
                answer = narrate_turn(wh, turn, llm=llm)
                assert "raw" in answer.lower() or "unweighted" in answer.lower(), (
                    "owner answer must state that raw/unweighted pipeline was used"
                )
            status = "PASS"
        except AssertionError as exc:
            status = "FAIL"
            failures.append(f"Q{index}: {exc}")
        elapsed = time.perf_counter() - started
        print(f"{status} Q{index:02d} [{turn.action}, {elapsed:.1f}s] {case.question}")
        print("     SQL:", " ".join((turn.sql or "(none)").split())[:700])
        if status == "FAIL":
            print("     RESULT:", None if turn.result is None else turn.result.to_dict("records"))
            print("     REASON:", failures[-1])

    if failures:
        print("\nSEMANTIC EVAL FAILURES")
        for failure in failures:
            print(" -", failure)
        return 1
    print(f"\nALL {len(CASES)} FOUNDER-QUESTION EVALS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
