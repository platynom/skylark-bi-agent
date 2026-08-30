"""FastAPI adapter for the Skylark BI agent with two-phase execution and semantic caching."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent import config
from agent.agent import (
    NARRATOR_SYSTEM,
    _relevant_caveats,
    deterministic_narration_fallback,
    deterministic_narrative as _deterministic_narrative,
    narrated_currency_is_valid,
    plan_and_execute,
    prepare_narrator_result,
)
from agent.leadership import generate_update
from agent.llm import Gemini, LLMError
from agent.warehouse import quality_summary
from api._warehouse_cache import (
    WarehouseLoad,
    get_cached_question,
    get_warehouse,
    normalize_question,
    set_cached_question,
)

LOGGER = logging.getLogger("skylark.api")
app = FastAPI(title="Skylark BI Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[HistoryItem] = Field(default_factory=list, max_length=20)


class NarrateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    intent: str | None = None
    sql: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    history: list[HistoryItem] = Field(default_factory=list, max_length=20)


class LeadershipRequest(BaseModel):
    focus: str | None = Field(default=None, max_length=1000)


class LeadershipGemini(Gemini):
    """Gemini client allowing narrative token budget override via LEADERSHIP_MAX_TOKENS."""
    def generate(self, system: str, user: str, *, temperature: float = 0.1, max_tokens: int = 2048, json_mode: bool = False) -> str:
        env_tokens = os.environ.get("LEADERSHIP_MAX_TOKENS")
        if env_tokens and env_tokens.strip().isdigit():
            max_tokens = int(env_tokens.strip())
        return super().generate(system, user, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)


def _clean_nan(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    return obj


MONDAY_ACCOUNT_URL = "https://tanmayk311s-team-company.monday.com"


def _records(frame: pd.DataFrame | None, board_id: str | None = None) -> list[dict[str, Any]]:
    if frame is None:
        return []
    records = json.loads(frame.to_json(orient="records", date_format="iso"))
    if board_id and "item_id" in frame.columns:
        for row in records:
            item_id = row.get("item_id")
            if item_id not in (None, ""):
                row["monday_item_url"] = (
                    f"{MONDAY_ACCOUNT_URL}/boards/{board_id}/pulses/{item_id}"
                )
    return records


def _result_board_id(sql: str | None, board_ids: dict[str, str]) -> str | None:
    """Resolve provenance only for a result drawn from one source table."""
    sql_l = (sql or "").lower()
    uses_deals = bool(re.search(r"\bdeals\b", sql_l))
    uses_work_orders = bool(re.search(r"\bwork_orders\b", sql_l))
    if uses_deals and not uses_work_orders:
        return board_ids.get("deals")
    if uses_work_orders and not uses_deals:
        return board_ids.get("work_orders")
    return None


def _meta(load: WarehouseLoad) -> dict[str, Any]:
    return {
        "cache": load.cache_status,
        "data_age_seconds": round(load.warehouse.age_seconds, 3),
        "fetched_at": load.fetched_at.isoformat(),
        "warning": load.warning,
    }


def _safe_error(exc: Exception, status: int = 503) -> JSONResponse:
    LOGGER.exception("API request failed: %s", type(exc).__name__)
    return JSONResponse(
        status_code=status,
        content={"error": {"code": "service_unavailable", "message": "The live business data service is temporarily unavailable."}},
    )


@app.get("/api/health")
def health(force: bool = Query(default=False)):
    started = time.perf_counter()
    try:
        load = get_warehouse(force_refresh=force)
        wh = load.warehouse
        return {
            "ok": True,
            "boards": wh.board_ids,
            "row_counts": {"deals": len(wh.deals), "work_orders": len(wh.work_orders)},
            "quality": quality_summary(wh),
            **_meta(load),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    except Exception as exc:  # noqa: BLE001
        return _safe_error(exc)


@app.post("/api/ask")
def ask(request: AskRequest):
    started = time.perf_counter()
    try:
        norm_q = normalize_question(request.question)
        if not request.history:
            cached_resp = get_cached_question(norm_q)
            if cached_resp is not None:
                return {
                    **cached_resp,
                    "cache": "question_cache",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                }

        load = get_warehouse()
        llm = Gemini()
        turn = plan_and_execute(
            load.warehouse,
            request.question.strip(),
            llm=llm,
            history=[item.model_dump() for item in request.history],
        )

        caveats = _relevant_caveats(load.warehouse, turn.sql) if turn.sql else []

        # Check for deterministic template on simple results
        if turn.action == "sql" and turn.result is not None and not turn.answer:
            deterministic = _deterministic_narrative(turn.result, turn.intent, turn.assumptions)
            if deterministic:
                turn.answer = deterministic

        result_rows = _records(
            turn.result,
            _result_board_id(turn.sql, load.warehouse.board_ids),
        )
        result_columns = [] if turn.result is None else list(turn.result.columns)
        if result_rows and "monday_item_url" in result_rows[0]:
            result_columns.append("monday_item_url")

        response_data = {
            "action": turn.action,
            "intent": turn.intent,
            "sql": turn.sql,
            "answer": turn.answer,
            "assumptions": turn.assumptions,
            "rows": result_rows,
            "rowcount": 0 if turn.result is None else len(turn.result),
            "columns": result_columns,
            "caveats": caveats,
            "attempts": turn.attempts,
            "clarify": turn.clarify,
            "options": turn.options,
            "error": turn.error,
            "model": llm._working_model,
            **_meta(load),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }

        if not request.history and turn.action in ("sql", "unsupported") and turn.answer:
            set_cached_question(norm_q, response_data)

        return response_data
    except Exception as exc:  # noqa: BLE001
        return _safe_error(exc)


@app.post("/api/narrate")
def narrate(request: NarrateRequest):
    started = time.perf_counter()
    try:
        llm = Gemini()

        df = pd.DataFrame.from_records(request.rows)
        preview, allowed_currency, currency_block = prepare_narrator_result(
            df, config.MAX_ROWS_TO_LLM
        )
        truncated = len(df) > len(preview)
        table_md = preview.to_markdown(index=False) if not preview.empty else "(no rows returned)"

        narrate_input = (
            f"QUESTION: {request.question}\n\n"
            f"WHAT WAS COMPUTED: {request.intent}\n\n"
            f"SQL:\n{request.sql}\n\n"
            f"RESULT ({len(df)} row(s)"
            f"{f', showing first {len(preview)}' if truncated else ''}):\n{table_md}\n\n"
            f"INTERPRETATION ASSUMPTIONS MADE: {request.assumptions or 'none'}\n\n"
            f"SERVER-PROVIDED CURRENCY STRINGS (copy verbatim; no others allowed):\n"
            f"{currency_block}\n\n"
            f"DATA CAVEATS:\n" + ("\n".join(f"- {c}" for c in request.caveats) if request.caveats else "(none)")
        )

        try:
            answer = llm.generate(NARRATOR_SYSTEM, narrate_input, temperature=0.3, max_tokens=1200)
            if not narrated_currency_is_valid(answer, allowed_currency):
                LOGGER.warning(
                    "Narrator emitted an INR value outside the server-provided allow-list"
                )
                answer = deterministic_narration_fallback(
                    df, request.intent, request.assumptions
                )
        except LLMError as exc:
            LOGGER.warning("Narration failed; using deterministic fallback: %s", exc)
            answer = deterministic_narration_fallback(
                df, request.intent, request.assumptions
            )

        # Update question cache with full response including narrative prose
        if not request.history and request.sql:
            norm_q = normalize_question(request.question)
            full_cached_resp = {
                "action": "sql",
                "intent": request.intent,
                "sql": request.sql,
                "answer": answer,
                "assumptions": request.assumptions,
                "rows": request.rows,
                "rowcount": len(request.rows),
                "columns": list(request.rows[0].keys()) if request.rows else [],
                "caveats": request.caveats,
                "attempts": [],
                "clarify": None,
                "options": [],
                "error": None,
                "model": llm._working_model,
            }
            set_cached_question(norm_q, full_cached_resp)

        return {
            "answer": answer,
            "model": llm._working_model,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    except Exception as exc:  # noqa: BLE001
        return _safe_error(exc)


@app.post("/api/leadership")
def leadership(request: LeadershipRequest):
    started = time.perf_counter()
    try:
        load = get_warehouse()
        llm = LeadershipGemini()
        narrative, metrics = generate_update(load.warehouse, llm=llm, focus=request.focus)
        return {
            "narrative": narrative, "metrics": _clean_nan(metrics), "model": llm._working_model,
            **_meta(load),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    except Exception as exc:  # noqa: BLE001
        return _safe_error(exc)


@app.get("/api/data")
def data(table: Literal["deals", "work_orders"] = "deals", limit: int = Query(100, ge=1, le=500)):
    try:
        load = get_warehouse()
        frame = load.warehouse.deals if table == "deals" else load.warehouse.work_orders
        rows = _records(frame.head(limit), load.warehouse.board_ids[table])
        columns = list(frame.columns)
        if "item_id" in frame.columns:
            columns.append("monday_item_url")
        return {"table": table, "rows": rows, "rowcount": len(frame), "columns": columns, **_meta(load)}
    except Exception as exc:  # noqa: BLE001
        return _safe_error(exc)


@app.post("/api/refresh")
def refresh():
    try:
        load = get_warehouse(force_refresh=True)
        return {
            "ok": True, "boards": load.warehouse.board_ids,
            "row_counts": {"deals": len(load.warehouse.deals), "work_orders": len(load.warehouse.work_orders)},
            **_meta(load),
        }
    except Exception as exc:  # noqa: BLE001
        return _safe_error(exc)
