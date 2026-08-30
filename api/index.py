"""FastAPI adapter for the unchanged Skylark BI agent."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent.agent import answer_question
from agent.leadership import generate_update
from agent.llm import Gemini
from agent.warehouse import quality_summary
from api._warehouse_cache import WarehouseLoad, get_warehouse

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


class LeadershipRequest(BaseModel):
    focus: str | None = Field(default=None, max_length=1000)


def _records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso"))


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
        load = get_warehouse()
        llm = Gemini()
        turn = answer_question(
            load.warehouse,
            request.question.strip(),
            llm=llm,
            history=[item.model_dump() for item in request.history],
        )
        return {
            "action": turn.action, "intent": turn.intent, "sql": turn.sql,
            "answer": turn.answer, "assumptions": turn.assumptions,
            "rows": _records(turn.result),
            "rowcount": 0 if turn.result is None else len(turn.result),
            "attempts": turn.attempts, "clarify": turn.clarify,
            "options": turn.options, "error": turn.error,
            "model": llm._working_model,
            **_meta(load),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    except Exception as exc:  # noqa: BLE001
        return _safe_error(exc)


@app.post("/api/leadership")
def leadership(request: LeadershipRequest):
    started = time.perf_counter()
    try:
        load = get_warehouse()
        llm = Gemini()
        narrative, metrics = generate_update(load.warehouse, llm=llm, focus=request.focus)
        return {
            "narrative": narrative, "metrics": metrics, "model": llm._working_model,
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
        return {"table": table, "rows": _records(frame.head(limit)), "rowcount": len(frame), "columns": list(frame.columns), **_meta(load)}
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
