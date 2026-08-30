"""
Query orchestration: natural language -> SQL -> result -> narrated insight.

Two model calls per answer:
  1. PLAN   - decide between asking a clarifying question and writing SQL.
  2. NARRATE- turn the result set into a founder-readable answer with caveats.

Splitting them keeps the SQL step cheap and deterministic-ish (temperature 0)
while allowing a warmer temperature for prose, and it means a failed SQL attempt
never costs a narration call.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import config
from .llm import Gemini, LLMError
from .warehouse import UnsafeQuery, Warehouse, run_sql, schema_document

PLANNER_SYSTEM = """You are the query planner for a business-intelligence agent used by
the founders of Skylark Drones, a drone services company. You translate business
questions into DuckDB SQL over two tables sourced live from monday.com boards.

RULES
1. Output JSON only, matching one of the shapes below.
2. Write DuckDB SQL. One statement. SELECT or WITH only. Never write, never DDL.
3. Only reference columns that appear in the schema. Do not invent columns.
4. A column marked UNUSABLE has no data at all. Never aggregate it. If the question
   depends on one, return action "unsupported" and say which field is not tracked.
5. NULL is not zero. Use explicit filters (WHERE x IS NOT NULL) when the question is
   about populated records, and prefer COUNT(col) alongside COUNT(*) so coverage is visible.
6. The two tables have NO shared customer key. deals.client_code (COMPANY*) and
   work_orders.customer_code (WOCOMPANY*) are different masking schemes and must never be
   joined. The only bridge is deal_name, which is NOT unique. If a cross-table join is
   required, aggregate each side to one row per deal_name in a CTE before joining, and
   record this in "assumptions".
7. Fiscal year is Indian convention (April-March). Pre-computed helper columns exist:
   *_fy (e.g. 'FY25-26') and *_fq (e.g. 'FY25-26 Q1'). Prefer them over date arithmetic.
8. "Pipeline" means deals where deal_status = 'Open'. "Won"/"Dead" are closed outcomes.
   Sector values are in the sector column; match them case-insensitively with ILIKE
   because founders use loose names (e.g. "energy" plausibly means Renewables and/or
   Powerline -- if the mapping is genuinely ambiguous, ask).
   Win rate ALWAYS means won / (won + dead): exclude Open and On Hold deals from the
   denominator. The is_won and is_dead columns are booleans; both are False (not NULL)
   for Open and On Hold deals, so `is_won IS NOT NULL AND is_dead IS NOT NULL` does NOT
   select closed deals. Filter closed outcomes with `is_won OR is_dead` (or filter
   deal_status to Won/Dead) before calculating the overall or per-sector win rate.
   Return win rates as percentage points rounded to two decimal places (e.g. 56.51),
   not as a 0-to-1 fraction, so the overall figure is reported consistently.
9. Ask a clarifying question ONLY when the answer would materially change depending on
   the interpretation AND you cannot state a reasonable default. Prefer answering with
   an explicit stated assumption over interrupting the user. Never ask more than one
   question at a time.
10. Return at most 200 rows. Aggregate rather than dumping raw rows unless the user
    explicitly asks for a list.
11. Apply LIMIT 1 ONLY when the question specifically asks for a single individual record or single named entity (e.g. "which deal owner has the largest X", "our single biggest deal", "which customer owes the most"). DO NOT apply LIMIT 1 when the question asks about concentration, distribution, breakdown, ranking, mix, comparison, or share across categories/sectors (e.g. "which sector is receivable concentrated in" requires a full breakdown by sector so the relative concentration and total are both visible).

OUTPUT SHAPES
{"action":"sql","intent":"<one line: what you are computing>","sql":"<query>",
 "assumptions":["<any interpretation you had to choose>"]}
{"action":"clarify","question":"<one question>","options":["<option>","<option>"]}
{"action":"unsupported","message":"<why this cannot be answered from these boards>"}
"""

NARRATOR_SYSTEM = """You are a business-intelligence analyst briefing the founders of
Skylark Drones. You are given a question, the SQL that was run against live monday.com
data, and the result table.

Write a direct answer, then the insight behind it. Rules:
- Lead with the number or the finding. No preamble, no restating the question.
- Amounts are Indian Rupees and are masked/scaled; format them as INR with lakh/crore
  framing when large (e.g. Rs 1.24 Cr). Never invent currency conversions.
- Add context: comparison, concentration, what is driving the number, what it implies.
  A bare number is a failed answer.
- If the DATA CAVEATS section is non-empty and a caveat materially affects this answer,
  state it in one short line under a "Caveat:" label. Do not repeat caveats that do not
  bear on this question.
- If the result set is empty, say so plainly and suggest the most likely reason
  (filter too narrow, or the field is sparsely populated).
- Never state a total as authoritative when the driving column has low coverage; say
  "across the N records that have a value" instead.
- Markdown. Short. No headers unless there are 3+ distinct sections. No emoji.
"""


@dataclass
class AgentTurn:
    question: str
    action: str = "sql"
    intent: str | None = None
    sql: str | None = None
    assumptions: list[str] = field(default_factory=list)
    result: pd.DataFrame | None = None
    answer: str | None = None
    clarify: str | None = None
    options: list[str] = field(default_factory=list)
    error: str | None = None
    attempts: list[dict[str, str]] = field(default_factory=list)


def _relevant_caveats(wh: Warehouse, sql: str | None) -> list[str]:
    """Only surface caveats touching columns this query actually used."""
    out: list[str] = []
    sql_l = (sql or "").lower()
    for table in ("deals", "work_orders"):
        q = wh.quality[table]
        for c in q.columns:
            if c.field.lower() in sql_l:
                if c.filled == 0:
                    out.append(f"{table}.{c.field} is never populated on the board.")
                elif c.fill_rate < 0.85:
                    out.append(
                        f"{table}.{c.field} is populated for only {c.filled} of "
                        f"{c.total} records ({c.fill_rate:.0%})."
                    )
                if c.coercion_failures:
                    out.append(
                        f"{c.coercion_failures} value(s) in {table}.{c.field} could not be "
                        f"parsed into {c.dtype} and are treated as missing."
                    )
        if q.dropped_header_rows:
            out.append(
                f"{q.dropped_header_rows} repeated header row(s) were removed from "
                f"{table} before analysis."
            )
        if "join" in sql_l and table == "work_orders":
            out.extend(n for n in q.notes if "no customer key" in n.lower())
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _history_block(history: list[dict[str, str]] | None, limit: int = 6) -> str:
    if not history:
        return "(none)"
    lines = []
    for h in history[-limit:]:
        role = "Founder" if h.get("role") == "user" else "Agent"
        lines.append(f"{role}: {h.get('content','')[:400]}")
    return "\n".join(lines)


def plan_and_execute(wh: Warehouse, question: str, llm: Gemini | None = None,
                     history: list[dict[str, str]] | None = None) -> AgentTurn:
    llm = llm or Gemini()
    turn = AgentTurn(question=question)
    schema = schema_document(wh)
    today = dt.date.today()

    context = (
        f"TODAY: {today.isoformat()} (fiscal year {'FY' + str(today.year)[2:] + '-' + str(today.year + 1)[2:] if today.month >= 4 else 'FY' + str(today.year - 1)[2:] + '-' + str(today.year)[2:]})\n\n"
        f"SCHEMA (read off the live monday boards):\n{schema}\n"
        f"CONVERSATION SO FAR:\n{_history_block(history)}\n\n"
        f"QUESTION: {question}"
    )

    try:
        plan = llm.generate_json(PLANNER_SYSTEM, context, temperature=0.0)
    except LLMError as exc:
        turn.action, turn.error = "error", str(exc)
        return turn

    action = str(plan.get("action", "sql")).lower()
    turn.action = action
    turn.assumptions = [str(a) for a in (plan.get("assumptions") or [])]

    if action == "clarify":
        turn.clarify = plan.get("question") or "Could you narrow that down?"
        turn.options = [str(o) for o in (plan.get("options") or [])]
        return turn
    if action == "unsupported":
        turn.answer = plan.get("message") or "That cannot be answered from these two boards."
        return turn

    sql = (plan.get("sql") or "").strip()
    turn.intent = plan.get("intent")
    df: pd.DataFrame | None = None
    last_error: str | None = None

    for attempt in range(config.MAX_SQL_RETRIES + 1):
        if not sql:
            last_error = "Planner returned no SQL."
            break
        try:
            df = run_sql(wh, sql)
            turn.sql = sql
            break
        except (UnsafeQuery, Exception) as exc:  # duckdb raises many exception types
            last_error = str(exc)
            turn.attempts.append({"sql": sql, "error": last_error})
            if attempt >= config.MAX_SQL_RETRIES:
                break
            repair = (
                f"{context}\n\nThe SQL you produced failed.\nSQL:\n{sql}\n\n"
                f"ERROR:\n{last_error}\n\nReturn corrected JSON. Only use columns "
                f"present in the schema above."
            )
            try:
                plan = llm.generate_json(PLANNER_SYSTEM, repair, temperature=0.0)
            except LLMError as exc2:
                last_error = str(exc2)
                break
            sql = (plan.get("sql") or "").strip()

    if df is None:
        turn.action = "error"
        turn.error = (
            f"I could not build a valid query for that after "
            f"{len(turn.attempts)} attempt(s). Last error: {last_error}"
        )
        return turn

    turn.result = df
    return turn


def narrate_turn(wh: Warehouse, turn: AgentTurn, llm: Gemini | None = None) -> str:
    if turn.action != "sql" or turn.result is None:
        return turn.answer or ""
    llm = llm or Gemini()
    df = turn.result
    caveats = _relevant_caveats(wh, turn.sql)

    preview = df.head(config.MAX_ROWS_TO_LLM)
    truncated = len(df) > len(preview)
    table_md = preview.to_markdown(index=False) if not preview.empty else "(no rows returned)"

    narrate_input = (
        f"QUESTION: {turn.question}\n\n"
        f"WHAT WAS COMPUTED: {turn.intent}\n\n"
        f"SQL:\n{turn.sql}\n\n"
        f"RESULT ({len(df)} row(s)"
        f"{f', showing first {len(preview)}' if truncated else ''}):\n{table_md}\n\n"
        f"INTERPRETATION ASSUMPTIONS MADE: {turn.assumptions or 'none'}\n\n"
        f"DATA CAVEATS:\n" + ("\n".join(f"- {c}" for c in caveats) if caveats else "(none)")
    )

    try:
        return llm.generate(NARRATOR_SYSTEM, narrate_input, temperature=0.3, max_tokens=1200)
    except LLMError as exc:
        return (
            f"Query ran successfully but the narration step failed ({exc}). "
            f"Raw result is shown below."
        )


def answer_question(wh: Warehouse, question: str, llm: Gemini | None = None,
                    history: list[dict[str, str]] | None = None) -> AgentTurn:
    llm = llm or Gemini()
    turn = plan_and_execute(wh, question, llm=llm, history=history)
    if turn.action == "sql" and turn.result is not None and not turn.answer:
        turn.answer = narrate_turn(wh, turn, llm=llm)
    return turn
