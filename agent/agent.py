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
import logging
import math
import numbers
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import config
from .llm import Gemini, LLMError, UnifiedLLM
from .templates import TEMPLATE_MATCH_THRESHOLD, match_template
from .warehouse import UnsafeQuery, Warehouse, run_sql, schema_document

LOGGER = logging.getLogger("skylark.agent")

PLANNER_SYSTEM = """You are the query planner for a business-intelligence agent used by
the founders of Skylark Drones, a drone services company. You translate business
questions into DuckDB SQL over two tables sourced live from monday.com boards.

RULES
1. Output JSON only, matching one of the shapes below.
2. Write DuckDB SQL. One statement. SELECT or WITH only. Never write, never DDL.
   When combining result sets with UNION ALL, if ordering, either ORDER BY column name / position
   (e.g. `ORDER BY 1`) or wrap the entire union in an outer query: `SELECT * FROM (SELECT ... UNION ALL SELECT ...) ORDER BY ...`.
3. Only reference columns that appear in the schema. Do not invent columns.
4. A column marked UNUSABLE has 0% fill rate and no data at all (such as collection_status,
   deal_type, priority, etc.). Never query, group by, or aggregate an unusable column. If a
   question specifically asks for or depends on an unusable column (e.g. "group by collection status",
   "by deal type", "by priority", "using the collection status field"), you MUST return action:
   "unsupported" and state that the field is not populated / not tracked on the monday board.
5. NULL is not zero. Use explicit filters (WHERE x IS NOT NULL) when the question is
   about populated records, and prefer COUNT(col) alongside COUNT(*) so coverage is visible.
   Never add unrequested filters on metadata columns (such as deal_name IS NOT NULL or
   owner_code IS NOT NULL) when computing population-level metrics like win rates, total
   deal counts, or overall distributions, as filtering on non-metric metadata shrinks the true population.
6. The two tables have NO shared customer key. deals.client_code (COMPANY*) and
   work_orders.customer_code (WOCOMPANY*) are different masking schemes and must never be
   joined. The only bridge is deal_name, which is NOT unique. If a cross-table join between
   deals and work_orders is required, you MUST aggregate each side to exactly ONE row per
   deal_name using `GROUP BY deal_name` in a CTE before joining (e.g.
   `WITH d AS (SELECT deal_name, SUM(deal_value) AS pipeline FROM deals WHERE deal_status='Open' AND deal_name IS NOT NULL GROUP BY deal_name), w AS (SELECT deal_name, SUM(amount_incl_gst) AS wo_value FROM work_orders WHERE deal_name IS NOT NULL GROUP BY deal_name) SELECT COUNT(d.deal_name), SUM(d.pipeline), SUM(w.wo_value) FROM d JOIN w USING (deal_name)`).
   Joining unaggregated rows causes Cartesian explosion and multiplies financial metrics.
   Always record this aggregation assumption in 'assumptions'. For single-table queries,
   never add `WHERE deal_name IS NOT NULL` unless explicitly asked for named records.
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
   deal_status to Won/Dead) before calculating the overall or per-sector win rate. Do
   not add deal_name, sector, owner, or other population filters to an overall win-rate
   denominator unless the founder explicitly requested that slice.
   Return win rates as percentage points rounded to two decimal places (e.g. 56.51),
   not as a 0-to-1 fraction, so the overall figure is reported consistently.
   When a win-rate question asks for an overall/company total alongside sectors,
   include a row labelled exactly 'Overall' for the grand total, and label unassigned/null sectors
   as 'Unassigned' or 'Unknown' (never use COALESCE(sector, 'Overall') on individual rows
   because it creates duplicate 'Overall' rows and collides with the grand total). To combine
   the sector breakdown and Overall row in DuckDB, wrap the UNION ALL in an outer query:
   `SELECT * FROM (SELECT sector, won, dead, win_rate FROM sector_stats UNION ALL SELECT 'Overall', won, dead, win_rate FROM overall_stats) ORDER BY CASE WHEN sector='Overall' THEN 0 ELSE 1 END, sector`.
   The Overall row must use every Won/Dead deal, including records whose sector is NULL; do not
   let a sector-population filter change the company-wide 56.51% denominator.
   Unless the founder explicitly asks for a weighted or probability-adjusted pipeline,
   "pipeline value" means the raw/unweighted SUM(deal_value) over Open deals. For an
   owner pipeline ranking, return exactly owner_code, COUNT(*) AS open_deal_count,
   COUNT(deal_value) AS valued_deal_count, and SUM(deal_value) AS
   raw_open_pipeline_value, ordered descending with LIMIT 1; label the intent/result as
   raw or unweighted so the answer states which definition was used. Use weighted_value
   only when the question explicitly asks for a weighted/probability-adjusted view.
   "Completed but not invoiced" means execution_status = 'Completed' exactly AND
   COALESCE(billed_incl_gst, 0) = 0. Exclude 'Partial Completed'. billed_incl_gst is the
   financial evidence of invoicing; do not substitute invoice_status, latest_invoice_no,
   wo_status_billed, or a fuzzy ILIKE match for this metric. A row-level result for this
   metric must include item_id, execution_status, billed_incl_gst, and amount_incl_gst so
   its pinned count and financial exposure can be verified and narrated consistently.
   "Fully paid with zero outstanding balance" means billed_incl_gst > 0 AND
   outstanding_incl_gst = 0. A zero-outstanding work order that was never billed is not
   fully paid; do not count it for this metric.
9. Ask a clarifying question ONLY when the answer would materially change depending on
   the interpretation AND you cannot state a reasonable default. Prefer answering with
   an explicit stated assumption over interrupting the user. Never ask more than one
   question at a time.
10. Return at most 200 rows. Aggregate rather than dumping raw rows unless the user
    explicitly asks for a list.
11. Apply LIMIT 1 ONLY when the question specifically asks for a single individual record or single named entity (e.g. "which deal owner has the largest X", "our single biggest deal", "which customer owes the most"). DO NOT apply LIMIT 1 when the question asks about concentration, distribution, breakdown, ranking, mix, comparison, or share across categories/sectors (e.g. "which sector is receivable concentrated in" requires a full breakdown by sector so the relative concentration and total are both visible).
12. PROVENANCE IS NON-NEGOTIABLE: for EVERY query returning individual deal, work order, or
    opportunity records (including questions like "show large deals", "list big opportunities",
    "which deals are large", "which work orders are completed", "our single biggest open deal",
    "name the top open deal by raw value", "show our largest individual open deal"), the SELECT list
    MUST include `item_id` so each row links directly to its monday.com pulse. A row-level deal or
    work-order SELECT without `item_id` is invalid. Aggregated/grouped rows (with GROUP BY) do not
    represent a single source item and therefore do not need it.
13. CATEGORICAL VALUES & EXACT MATCHING: When filtering by categorical fields (deal_stage,
    deal_status, sector, execution_status, owner_code, customer_code), copy string literals
    VERBATIM from the schema document (including exact spacing, punctuation, and casing, e.g.
    `deal_stage = 'E. Proposal/Commercials Sent'`). As a safety rule, prefer `ILIKE '%keyword%'`
    (such as `deal_stage ILIKE '%Proposal%'` or `sector ILIKE '%Renewables%'`) over exact `=`
    when matching stages, sectors, or descriptions unless copying the exact value verbatim.

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
- Currency values have server-formatted companion fields and a separate allow-list.
  Copy those INR strings VERBATIM. Never calculate, rescale, round, or emit an `Rs ...`
  value that is not present in the server-provided currency strings.
- Add context: comparison, concentration, what is driving the number, what it implies.
  A bare number is a failed answer.
- If the DATA CAVEATS section is non-empty and a caveat materially affects this answer,
  state it in one short line under a "Caveat:" label. Do not repeat caveats that do not
  bear on this question.
- If the result set is empty, say so plainly and suggest the most likely reason
  (filter too narrow, or the field is sparsely populated).
- Never state a total as authoritative when the driving column has low coverage; say
  "across the N records that have a value" instead.
- When the result distinguishes raw/unweighted pipeline from weighted pipeline, explicitly
  state which definition was used.
- Markdown. Short. No headers unless there are 3+ distinct sections. No emoji.
"""

_CURRENCY_HINTS = (
    "amount", "value", "receivable", "outstanding", "billed", "collected",
    "pipeline", "revenue", "contracted", "unbilled", "price",
)
_NON_CURRENCY_HINTS = ("count", "rate", "pct", "percent", "probability", "qty", "id")
_RS_TOKEN_RE = re.compile(
    r"\bRs\s+-?[\d,]+(?:\.\d+)?(?:\s+(?:Cr|Lakh|L|K))?\b",
    flags=re.IGNORECASE,
)


def _is_currency_column(column: str) -> bool:
    name = column.lower()
    return (
        any(hint in name for hint in _CURRENCY_HINTS)
        and not any(hint in name for hint in _NON_CURRENCY_HINTS)
        and not name.endswith("__formatted")
    )


def format_inr(value: Any) -> str:
    """Format one raw rupee value deterministically for all narration paths."""
    amount = float(value)
    sign = "-" if amount < 0 else ""
    absolute = abs(amount)
    if absolute >= 1e7:
        return f"Rs {sign}{absolute / 1e7:.2f} Cr"
    if absolute >= 1e5:
        return f"Rs {sign}{absolute / 1e5:.2f} L"
    if absolute >= 1e3:
        return f"Rs {sign}{absolute / 1e3:.2f} K"
    return f"Rs {sign}{absolute:,.2f}"


def format_result_value(column: str, value: Any) -> str:
    if value is None or (isinstance(value, numbers.Number) and not math.isfinite(float(value))):
        return "NULL"
    if isinstance(value, numbers.Number) and not isinstance(value, bool):
        numeric = float(value)
        if _is_currency_column(column):
            return format_inr(numeric)
        name = column.lower()
        if any(hint in name for hint in ("rate", "pct", "percent", "probability")):
            return f"{numeric * 100:.1f}%" if 0 <= numeric <= 1 else f"{numeric:.1f}%"
        if numeric.is_integer():
            return f"{int(numeric):,}"
        return f"{numeric:,.2f}"
    return str(value)


def prepare_narrator_result(
    frame: pd.DataFrame, limit: int,
) -> tuple[pd.DataFrame, set[str], str]:
    """Attach formatted currency companions and return the only INR strings allowed."""
    preview = frame.head(limit).copy()
    allowed: set[str] = set()
    supplied: list[str] = []
    for column in frame.columns:
        if not _is_currency_column(str(column)) or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        formatted_column = f"{column}__formatted"
        preview[formatted_column] = preview[column].map(
            lambda value: format_inr(value) if pd.notna(value) else None
        )
        row_values = [format_inr(value) for value in preview[column] if pd.notna(value)]
        allowed.update(row_values)
        supplied.extend(f"{formatted_column}: {value}" for value in dict.fromkeys(row_values))
        populated = frame[column].dropna()
        if not populated.empty:
            total = format_inr(populated.sum())
            allowed.add(total)
            supplied.append(f"SUM({column}): {total}")
    currency_block = "\n".join(f"- {entry}" for entry in supplied) if supplied else "(none)"
    return preview, allowed, currency_block


def narrated_currency_is_valid(answer: str, allowed: set[str]) -> bool:
    """Reject hallucinated or rescaled rupee strings before they reach a founder."""
    emitted = {match.group(0) for match in _RS_TOKEN_RE.finditer(answer)}
    return all(value in allowed for value in emitted)


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
    provider: str = "none"
    model: str | None = None
    provider_chain_attempted: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


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


def _missing_row_provenance(sql: str, result: pd.DataFrame) -> bool:
    """Reject individual-record queries that cannot link back to monday.com.

    Aggregates, category lists, and cross-board results do not correspond to one
    source item. A plain SELECT from exactly one source table does.
    """
    if "item_id" in result.columns:
        return False
    sql_l = sql.lower()
    uses_deals = bool(re.search(r"\bdeals\b", sql_l))
    uses_work_orders = bool(re.search(r"\bwork_orders\b", sql_l))
    if uses_deals == uses_work_orders:  # neither table, or both tables
        return False
    # Inspect the outer/final projection, not an aggregate inside an earlier CTE.
    select_part = sql_l.rsplit("select", 1)[-1].split("from", 1)[0]
    aggregate = bool(re.search(r"\b(count|sum|avg|min|max|median|percentile_\w*)\s*\(", select_part))
    category_result = "group by" in sql_l or "select distinct" in sql_l
    set_result = " intersect " in sql_l or " union " in sql_l
    return not (aggregate or category_result or set_result)


def deterministic_narrative(
    df: pd.DataFrame | None, intent: str | None, assumptions: list[str],
) -> str | None:
    """Generate an audited summary for result shapes with pinned founder metrics."""
    if df is None or df.empty:
        return None

    owner_columns = {
        "owner_code", "open_deal_count", "valued_deal_count",
        "raw_open_pipeline_value",
    }
    if owner_columns <= set(df.columns):
        top = df.sort_values("raw_open_pipeline_value", ascending=False).iloc[0]
        text = (
            f"**{top['owner_code']}** has the largest open pipeline: "
            f"**{format_inr(top['raw_open_pipeline_value'])}** "
            f"across **{int(top['open_deal_count'])} open deals**, of which "
            f"**{int(top['valued_deal_count'])} carry a value**. This is the "
            "**raw/unweighted** pipeline; probability weighting was not used."
        )
        if assumptions:
            text += f"\n\n*Assumptions: {'; '.join(assumptions)}*"
        return text

    uninvoiced_columns = {
        "item_id", "execution_status", "billed_incl_gst", "amount_incl_gst",
    }
    if uninvoiced_columns <= set(df.columns):
        completed = df["execution_status"].eq("Completed").all()
        zero_billed = df["billed_incl_gst"].fillna(0).eq(0).all()
        if completed and zero_billed:
            positive = df["amount_incl_gst"].fillna(0).gt(0)
            contracted = float(df.loc[positive, "amount_incl_gst"].sum())
            return (
                f"**{len(df)} completed work orders have no recorded billing**, "
                f"representing **{format_inr(contracted)}** of contracted value "
                f"across **{int(positive.sum())} positive-value work orders**. This "
                "uses exactly `execution_status = 'Completed'` and "
                "`COALESCE(billed_incl_gst, 0) = 0`; partially completed work "
                "orders and invoice workflow labels are excluded."
            )

    if len(df) == 1 and len(df.columns) <= 3:
        columns = list(df.columns)
        if len(columns) == 1:
            value = format_result_value(columns[0], df.iloc[0, 0])
            description = (intent or columns[0].replace("_", " ")).strip().rstrip(".")
            text = f"**{value}** ({description})."
        elif len(columns) == 2:
            first, second = columns
            text = (
                f"**{df.iloc[0, 0]}**: "
                f"**{format_result_value(second, df.iloc[0, 1])}** "
                f"({second.replace('_', ' ')})."
            )
        else:
            first, second, third = columns
            text = (
                f"**{df.iloc[0, 0]}**: {second.replace('_', ' ')} = "
                f"**{format_result_value(second, df.iloc[0, 1])}**, "
                f"{third.replace('_', ' ')} = "
                f"**{format_result_value(third, df.iloc[0, 2])}**."
            )
        if assumptions:
            text += f"\n\n*Assumptions: {'; '.join(assumptions)}*"
        return text
    return None


def deterministic_narration_fallback(
    frame: pd.DataFrame, intent: str | None, assumptions: list[str],
) -> str:
    """Return a safe answer when Gemini fails or emits an unapproved INR value."""
    pinned = deterministic_narrative(frame, intent, assumptions)
    if pinned:
        return pinned
    if frame.empty:
        return "No matching records were returned. The selected filter may be too narrow."
    totals: list[str] = []
    for column in frame.columns:
        if _is_currency_column(str(column)) and pd.api.types.is_numeric_dtype(frame[column]):
            populated = frame[column].dropna()
            if not populated.empty:
                totals.append(f"{column.replace('_', ' ')}: **{format_inr(populated.sum())}**")
    summary = f"The query returned **{len(frame)} row(s)**."
    if totals:
        summary += " Server-verified currency totals are " + "; ".join(totals) + "."
    summary += " The result rows and SQL below are the authoritative detail."
    if assumptions:
        summary += f"\n\n*Assumptions: {'; '.join(assumptions)}*"
    return summary


def _handle_deterministic_floor(
    wh: Warehouse, turn: AgentTurn, question: str, llm: Gemini | None = None
) -> AgentTurn:
    """Fallback handler when live LLM capacity across all providers is exhausted.

    A threshold of 0.35 ensures that queries with at least two matching core domain
    keywords trigger the appropriate precomputed template while avoiding false positives
    on ambiguous or out-of-scope queries.
    """
    turn.provider = "deterministic"
    turn.model = None
    turn.provider_chain_attempted = getattr(llm, "last_chain", []) if llm else []
    turn.latency_ms = getattr(llm, "last_latency_ms", 0.0) if llm else 0.0

    template, score = match_template(question)
    if template and score >= TEMPLATE_MATCH_THRESHOLD:
        try:
            df = run_sql(wh, template.sql)
            turn.action = "sql"
            turn.sql = template.sql
            turn.intent = template.intent
            turn.result = df
            turn.assumptions = [
                "Answer generated from deterministic query template because live model capacity was exhausted."
            ]
            return turn
        except Exception as sql_exc:
            LOGGER.warning("Deterministic template execution failed: %s", sql_exc)

    turn.action = "unsupported"
    turn.answer = (
        "Live AI model capacity is currently exhausted across all providers, and this "
        "question does not match a standard precomputed template. Please retry in a moment."
    )
    return turn


def plan_and_execute(
    wh: Warehouse,
    question: str,
    llm: Gemini | None = None,
    history: list[dict[str, str]] | None = None,
    force_provider: str | None = None,
) -> AgentTurn:
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
        plan = llm.generate_json(PLANNER_SYSTEM, context, temperature=0.0, force_provider=force_provider)
        turn.provider = getattr(llm, "last_provider", "vertex")
        turn.model = getattr(llm, "last_model", None)
        turn.provider_chain_attempted = getattr(llm, "last_chain", [])
        turn.latency_ms = getattr(llm, "last_latency_ms", 0.0)
    except LLMError as exc:
        LOGGER.warning("Planner LLM call failed across all providers (%s); falling back to deterministic floor", exc)
        return _handle_deterministic_floor(wh, turn, question, llm=llm)

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
            if _missing_row_provenance(sql, df):
                raise UnsafeQuery(
                    "Row-level results must SELECT item_id so every row can link "
                    "to its monday.com source record. Add item_id to the SELECT list."
                )
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
                plan = llm.generate_json(PLANNER_SYSTEM, repair, temperature=0.0, force_provider=force_provider)
            except LLMError as exc2:
                LOGGER.warning("SQL repair LLM call failed across all providers (%s); falling back to deterministic floor", exc2)
                return _handle_deterministic_floor(wh, turn, question, llm=llm)
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


def narrate_turn(
    wh: Warehouse,
    turn: AgentTurn,
    llm: Gemini | None = None,
    force_provider: str | None = None,
) -> str:
    if turn.action != "sql" or turn.result is None:
        return turn.answer or ""
    llm = llm or Gemini()
    df = turn.result

    # If all live LLM providers are throttled or on standby, skip the LLM call directly
    provider_status = UnifiedLLM.get_provider_status()
    live_providers = [
        info for name, info in provider_status.items()
        if name != "deterministic"
    ]
    if live_providers and all(info.get("status") in {"throttled", "standby"} for info in live_providers):
        LOGGER.info("All live LLM providers are throttled/standby; returning deterministic narration directly.")
        return deterministic_narration_fallback(df, turn.intent, turn.assumptions)

    caveats = _relevant_caveats(wh, turn.sql)

    preview, allowed_currency, currency_block = prepare_narrator_result(
        df, config.MAX_ROWS_TO_LLM
    )
    truncated = len(df) > len(preview)
    table_md = preview.to_markdown(index=False) if not preview.empty else "(no rows returned)"

    narrate_input = (
        f"QUESTION: {turn.question}\n\n"
        f"WHAT WAS COMPUTED: {turn.intent}\n\n"
        f"SQL:\n{turn.sql}\n\n"
        f"RESULT ({len(df)} row(s)"
        f"{f', showing first {len(preview)}' if truncated else ''}):\n{table_md}\n\n"
        f"INTERPRETATION ASSUMPTIONS MADE: {turn.assumptions or 'none'}\n\n"
        f"SERVER-PROVIDED CURRENCY STRINGS (copy verbatim; no others allowed):\n"
        f"{currency_block}\n\n"
        f"DATA CAVEATS:\n" + ("\n".join(f"- {c}" for c in caveats) if caveats else "(none)")
    )

    try:
        answer = llm.generate(
            NARRATOR_SYSTEM,
            narrate_input,
            temperature=0.3,
            max_tokens=1200,
            force_provider=force_provider,
        )
        if narrated_currency_is_valid(answer, allowed_currency):
            return answer
        LOGGER.warning("Narrator emitted an INR value outside the server-provided allow-list")
        return deterministic_narration_fallback(df, turn.intent, turn.assumptions)
    except LLMError as exc:
        LOGGER.warning("Narration failed; using deterministic fallback: %s", exc)
        return deterministic_narration_fallback(df, turn.intent, turn.assumptions)


def answer_question(
    wh: Warehouse,
    question: str,
    llm: Gemini | None = None,
    history: list[dict[str, str]] | None = None,
    force_provider: str | None = None,
) -> AgentTurn:
    llm = llm or Gemini()
    turn = plan_and_execute(wh, question, llm=llm, history=history, force_provider=force_provider)
    if turn.action == "sql" and turn.result is not None and not turn.answer:
        turn.answer = narrate_turn(wh, turn, llm=llm, force_provider=force_provider)
    return turn
