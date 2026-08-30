"""External normalized-warehouse cache for stateless Vercel functions.

Supabase is accessed only through its HTTPS PostgREST API. If it is absent,
paused, stale, or unavailable, callers still receive a live monday.com warehouse.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import duckdb
import pandas as pd
import requests

from agent.normalize import ColumnQuality, TableQuality
from agent.warehouse import Warehouse, build_warehouse, quality_summary, schema_document

LOGGER = logging.getLogger("skylark.warehouse_cache")
CACHE_KEY = "skylark:warehouse:v2"
CACHE_TTL_SECONDS = 1800
REQUEST_TIMEOUT_SECONDS = 8


@dataclass
class WarehouseLoad:
    warehouse: Warehouse
    cache_status: str
    fetched_at: datetime
    warning: str | None = None


def _supabase_settings() -> tuple[str | None, str | None]:
    url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/") or None
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip() or None
    return url, key


def _headers(key: str) -> dict[str, str]:
    # Works with legacy service_role JWTs and current sb_secret keys. The key is
    # never imported by or returned to the Next.js client.
    #
    # BOTH headers are required. `apikey` gets the request past Supabase's gateway,
    # but PostgREST derives the Postgres ROLE from the JWT in `Authorization`.
    # Sending `apikey` alone runs the query as `anon`, which RLS on warehouse_cache
    # blocks -- reads come back empty (indistinguishable from a cache miss) and
    # writes fail. With `Authorization: Bearer <service_role>`, PostgREST assumes the
    # service_role and bypasses RLS as intended.
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _table_url(url: str) -> str:
    return f"{url}/rest/v1/warehouse_cache"


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _df_to_parquet_b64(df: pd.DataFrame) -> str:
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
        path = tf.name.replace("\\", "/")
    try:
        con = duckdb.connect(":memory:")
        con.register("df_view", df)
        con.execute(f"COPY df_view TO '{path}' (FORMAT PARQUET)")
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("ascii")
    finally:
        if os.path.exists(path):
            os.remove(path)


def _parquet_b64_to_df(b64_str: str, dtypes: dict[str, str] | None = None) -> pd.DataFrame:
    raw = base64.b64decode(b64_str.encode("ascii"))
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
        tf.write(raw)
        path = tf.name.replace("\\", "/")
    try:
        con = duckdb.connect(":memory:")
        df = con.execute(f"SELECT * FROM read_parquet('{path}')").df()
        if dtypes:
            for col, dtype in dtypes.items():
                if col in df:
                    if "datetime" in dtype:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    elif dtype.startswith(("float", "int", "Int", "UInt")):
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                        try:
                            df[col] = df[col].astype(dtype)
                        except (TypeError, ValueError):
                            pass
                    elif dtype in ("bool", "boolean"):
                        df[col] = df[col].astype("boolean" if df[col].isna().any() else "bool")
                    elif dtype.startswith("string"):
                        df[col] = df[col].astype("string")
        return df
    finally:
        if os.path.exists(path):
            os.remove(path)


def _frame_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _serialize(wh: Warehouse) -> dict[str, Any]:
    return {
        "version": 2,
        "parquet": {
            "deals": _df_to_parquet_b64(wh.deals),
            "work_orders": _df_to_parquet_b64(wh.work_orders),
        },
        "dtypes": {
            "deals": {column: str(dtype) for column, dtype in wh.deals.dtypes.items()},
            "work_orders": {column: str(dtype) for column, dtype in wh.work_orders.dtypes.items()},
        },
        "schema_document": schema_document(wh),
        "quality_summary": quality_summary(wh),
        "quality_detail": {name: asdict(profile) for name, profile in wh.quality.items()},
        "board_ids": wh.board_ids,
    }


def _restore_frame(records: list[dict[str, Any]], dtypes: dict[str, str]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    for column in dtypes:
        if column not in frame:
            frame[column] = pd.Series(dtype="object")
    frame = frame[list(dtypes)]
    for column, dtype in dtypes.items():
        if "datetime" in dtype:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        elif dtype.startswith(("float", "int", "Int", "UInt")):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            try:
                frame[column] = frame[column].astype(dtype)
            except (TypeError, ValueError):
                pass
        elif dtype in ("bool", "boolean"):
            frame[column] = frame[column].astype("boolean" if frame[column].isna().any() else "bool")
        elif dtype.startswith("string"):
            frame[column] = frame[column].astype("string")
    return frame


def _restore_quality(raw: dict[str, Any]) -> dict[str, TableQuality]:
    restored: dict[str, TableQuality] = {}
    for table, profile in raw.items():
        values = dict(profile)
        values["columns"] = [ColumnQuality(**column) for column in values.get("columns", [])]
        restored[table] = TableQuality(**values)
    return restored


def _deserialize(payload: dict[str, Any], fetched_at: datetime) -> Warehouse:
    version = payload.get("version")
    if version == 2 and "parquet" in payload:
        dtypes = payload["dtypes"]
        deals = _parquet_b64_to_df(payload["parquet"]["deals"], dtypes["deals"])
        work_orders = _parquet_b64_to_df(payload["parquet"]["work_orders"], dtypes["work_orders"])
    elif version == 1 and "frames" in payload:
        frames, dtypes = payload["frames"], payload["dtypes"]
        deals = _restore_frame(frames["deals"], dtypes["deals"])
        work_orders = _restore_frame(frames["work_orders"], dtypes["work_orders"])
    else:
        raise ValueError("Unsupported warehouse cache version")

    con = duckdb.connect(":memory:")
    con.register("deals_df", deals)
    con.register("work_orders_df", work_orders)
    con.execute("CREATE TABLE deals AS SELECT * FROM deals_df")
    con.execute("CREATE TABLE work_orders AS SELECT * FROM work_orders_df")
    return Warehouse(
        con=con,
        deals=deals,
        work_orders=work_orders,
        quality=_restore_quality(payload["quality_detail"]),
        loaded_at=fetched_at.timestamp(),
        board_ids={str(k): str(v) for k, v in payload["board_ids"].items()},
        _schema_doc=payload.get("schema_document"),
    )


def _read_cache(url: str, key: str) -> tuple[dict[str, Any], datetime] | None:
    response = requests.get(
        _table_url(url),
        headers=_headers(key),
        params={"key": f"eq.{CACHE_KEY}", "select": "payload,fetched_at", "limit": "1"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    rows = response.json()
    if not rows:
        return None
    fetched_at = _parse_timestamp(rows[0]["fetched_at"])
    if (datetime.now(timezone.utc) - fetched_at).total_seconds() > CACHE_TTL_SECONDS:
        return None
    return rows[0]["payload"], fetched_at


def _write_cache(url: str, key: str, wh: Warehouse) -> datetime:
    fetched_at = datetime.now(timezone.utc)
    response = requests.post(
        _table_url(url),
        headers={**_headers(key), "Prefer": "resolution=merge-duplicates,return=minimal"},
        params={"on_conflict": "key"},
        json={"key": CACHE_KEY, "payload": _serialize(wh), "fetched_at": fetched_at.isoformat()},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return fetched_at


def get_warehouse(*, force_refresh: bool = False) -> WarehouseLoad:
    """Return normalized data, preferring a fresh Supabase cache record."""
    url, key = _supabase_settings()
    warning: str | None = None
    if not url or not key:
        warning = (
            "Supabase warehouse cache is not configured; falling back to a live "
            "monday.com fetch for this request."
        )
        LOGGER.warning(warning)
    elif not force_refresh:
        try:
            cached = _read_cache(url, key)
            if cached is not None:
                payload, fetched_at = cached
                return WarehouseLoad(_deserialize(payload, fetched_at), "hit", fetched_at)
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            warning = (
                "Supabase warehouse cache is unavailable or invalid; falling back "
                "to a live monday.com fetch for this request."
            )
            LOGGER.warning("%s (%s)", warning, type(exc).__name__)

    wh = build_warehouse()
    fetched_at = datetime.fromtimestamp(wh.loaded_at, timezone.utc)
    status = "bypass" if not url or not key else "miss"
    if url and key:
        try:
            fetched_at = _write_cache(url, key, wh)
            wh.loaded_at = fetched_at.timestamp()
        except requests.RequestException as exc:
            warning = "Live data loaded, but the Supabase cache write failed."
            LOGGER.warning("%s (%s)", warning, type(exc).__name__)
    return WarehouseLoad(wh, status, fetched_at, warning)


QUESTION_CACHE_KEY = "skylark:qcache:v1"
QUESTION_CACHE_TTL_SECONDS = 900
_LOCAL_QCACHE: dict[str, tuple[dict[str, Any], float]] = {}


def normalize_question(q: str) -> str:
    import re
    cleaned = re.sub(r"[^\w\s]", " ", q.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def get_cached_question(norm_q: str) -> dict[str, Any] | None:
    now = time.time()
    if norm_q in _LOCAL_QCACHE:
        val, ts = _LOCAL_QCACHE[norm_q]
        if now - ts <= QUESTION_CACHE_TTL_SECONDS:
            return dict(val)
        _LOCAL_QCACHE.pop(norm_q, None)

    url, key = _supabase_settings()
    if not url or not key:
        return None

    try:
        response = requests.get(
            _table_url(url),
            headers=_headers(key),
            params={"key": f"eq.{QUESTION_CACHE_KEY}", "select": "payload,fetched_at", "limit": "1"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code == 200:
            rows = response.json()
            if rows:
                payload = rows[0].get("payload") or {}
                entry = payload.get(norm_q)
                if entry and isinstance(entry, dict):
                    cached_at_str = entry.get("cached_at")
                    if cached_at_str:
                        cached_at = datetime.fromisoformat(cached_at_str.replace("Z", "+00:00"))
                        if (datetime.now(timezone.utc) - cached_at).total_seconds() <= QUESTION_CACHE_TTL_SECONDS:
                            data = entry.get("data")
                            if data:
                                _LOCAL_QCACHE[norm_q] = (data, now)
                                return dict(data)
    except Exception as exc:
        LOGGER.debug("Question cache read error: %s", exc)
    return None


def set_cached_question(norm_q: str, response_data: dict[str, Any]) -> None:
    import time
    now = time.time()
    _LOCAL_QCACHE[norm_q] = (dict(response_data), now)

    url, key = _supabase_settings()
    if not url or not key:
        return

    try:
        current_payload: dict[str, Any] = {}
        read_resp = requests.get(
            _table_url(url),
            headers=_headers(key),
            params={"key": f"eq.{QUESTION_CACHE_KEY}", "select": "payload", "limit": "1"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if read_resp.status_code == 200:
            rows = read_resp.json()
            if rows and isinstance(rows[0].get("payload"), dict):
                current_payload = rows[0]["payload"]

        cutoff = datetime.now(timezone.utc).timestamp() - QUESTION_CACHE_TTL_SECONDS
        cleaned_payload = {}
        for k, v in current_payload.items():
            if isinstance(v, dict) and "cached_at" in v:
                try:
                    ts = datetime.fromisoformat(v["cached_at"].replace("Z", "+00:00")).timestamp()
                    if ts >= cutoff:
                        cleaned_payload[k] = v
                except Exception:
                    pass

        cleaned_payload[norm_q] = {
            "data": response_data,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }

        requests.post(
            _table_url(url),
            headers={**_headers(key), "Prefer": "resolution=merge-duplicates,return=minimal"},
            params={"on_conflict": "key"},
            json={
                "key": QUESTION_CACHE_KEY,
                "payload": cleaned_payload,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        LOGGER.debug("Question cache write error: %s", exc)
