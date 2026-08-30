"""
In-memory DuckDB warehouse plus the schema document handed to the LLM.

Why DuckDB and not a fixed set of Python "tools": founder questions are
open-ended ("pipeline for energy this quarter", "which accounts are stuck in
billing"). A fixed toolset answers only the questions we anticipated; SQL over a
documented schema answers the ones we did not, and the generated SQL is shown to
the user, which makes every number auditable. The cost is prompt-injection-shaped
risk, mitigated by a read-only statement guard.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import duckdb
import pandas as pd

from . import config
from .monday_client import MondayClient, MondayError
from .normalize import TableQuality, normalize_board

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|copy|export|install|load|"
    r"pragma|set|call|replace)\b",
    re.IGNORECASE,
)


class UnsafeQuery(RuntimeError):
    pass


@dataclass
class Warehouse:
    con: duckdb.DuckDBPyConnection
    deals: pd.DataFrame
    work_orders: pd.DataFrame
    quality: dict[str, TableQuality]
    loaded_at: float
    board_ids: dict[str, str]

    @property
    def age_seconds(self) -> float:
        return time.time() - self.loaded_at


def build_warehouse(client: MondayClient | None = None) -> Warehouse:
    """Fetch both boards live from monday.com and load them into DuckDB."""
    client = client or MondayClient()

    deals_id = client.resolve_board_id(config.DEALS_BOARD_ID, config.DEALS_BOARD_NAME)
    wo_id = client.resolve_board_id(config.WORK_ORDERS_BOARD_ID, config.WORK_ORDERS_BOARD_NAME)

    deals_board = client.fetch_board(deals_id)
    wo_board = client.fetch_board(wo_id)

    deals_df, deals_q = normalize_board(deals_board, "deals")
    wo_df, wo_q = normalize_board(wo_board, "work_orders")

    # Cross-board join reality check, surfaced as a caveat rather than assumed away.
    if "deal_name" in deals_df and "deal_name" in wo_df:
        d_names = set(deals_df["deal_name"].dropna())
        w_names = set(wo_df["deal_name"].dropna())
        overlap = d_names & w_names
        dupes = int(deals_df["deal_name"].duplicated().sum())
        wo_q.notes.append(
            f"Boards share no customer key (deals use COMPANY*, work orders use "
            f"WOCOMPANY*). The only bridge is the masked deal name: "
            f"{len(overlap)} of {len(w_names)} work-order names match a deal name. "
            f"Deal names repeat ({dupes} duplicate rows), so any join is "
            f"many-to-many and will inflate sums unless aggregated per side first."
        )

    con = duckdb.connect(":memory:")
    con.register("deals_df", deals_df)
    con.register("work_orders_df", wo_df)
    con.execute("CREATE TABLE deals AS SELECT * FROM deals_df")
    con.execute("CREATE TABLE work_orders AS SELECT * FROM work_orders_df")

    return Warehouse(
        con=con, deals=deals_df, work_orders=wo_df,
        quality={"deals": deals_q, "work_orders": wo_q},
        loaded_at=time.time(), board_ids={"deals": deals_id, "work_orders": wo_id},
    )


def run_sql(wh: Warehouse, sql: str, limit: int = 500) -> pd.DataFrame:
    """Execute a single read-only SELECT. Raises UnsafeQuery on anything else."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeQuery("Empty query.")
    if ";" in cleaned:
        raise UnsafeQuery("Only a single statement is allowed.")
    head = cleaned.lstrip("( \n\t").lower()
    if not (head.startswith("select") or head.startswith("with")):
        raise UnsafeQuery("Only SELECT / WITH queries are allowed.")
    if _FORBIDDEN.search(cleaned):
        raise UnsafeQuery("Query contains a write or configuration keyword; refused.")

    df = wh.con.execute(cleaned).fetchdf()
    return df.head(limit)


# --------------------------------------------------------------------------- #
# schema document for the prompt
# --------------------------------------------------------------------------- #

_FIELD_NOTES = {
    "deal_value": "Deal size in INR (masked/scaled). Sparsely populated -- always report coverage.",
    "weighted_value": "deal_value * probability_weight. NULL when either input is missing.",
    "probability_weight": "High=0.75, Medium=0.45, Low=0.20. Our assumption, not a monday field.",
    "deal_stage_order": "Numeric rank parsed from the 'A.'/'B.' stage prefix; higher = later stage.",
    "amount_incl_gst": "Total work-order value including GST -- the contracted revenue.",
    "billed_incl_gst": "Invoiced to date, including GST.",
    "collected_incl_gst": "Cash received, including GST.",
    "unbilled_incl_gst": "amount_incl_gst - billed_incl_gst (derived).",
    "outstanding_incl_gst": "billed_incl_gst - collected_incl_gst (derived); the AR exposure.",
    "receivable": "Board's own 'Amount Receivable' field. May disagree with outstanding_incl_gst.",
    "qty_po_value": "Numeric part of the PO quantity.",
    "qty_po_unit": "Unit parsed out of the same cell (HA, KM, ...). NULL means the cell had no unit.",
}


def schema_document(wh: Warehouse) -> str:
    """A compact, data-derived schema + quality briefing. Nothing here is hardcoded
    from the CSVs -- distinct values and fill rates are read off the live boards."""
    lines: list[str] = []
    for table, df in (("deals", wh.deals), ("work_orders", wh.work_orders)):
        q = wh.quality[table]
        lines.append(f"TABLE {table}  ({q.rows_kept} rows, from monday board "
                     f"'{q.board_name}' id={q.board_id})")
        qmap = {c.field: c for c in q.columns}
        for col in df.columns:
            if col == "item_id":
                continue
            dtype = str(df[col].dtype)
            simple = ("DATE" if "datetime" in dtype else
                      "DOUBLE" if dtype.startswith(("float", "int")) else
                      "BOOLEAN" if dtype == "bool" else "TEXT")
            cq = qmap.get(col)
            filled = int(df[col].notna().sum())
            pct = round(100 * filled / max(len(df), 1))
            bits = [f"  - {col} {simple}  [filled {pct}%]"]
            if filled == 0:
                bits.append("UNUSABLE: never populated on the board -- answer 'not tracked', never 0")
            elif cq and cq.sample_values:
                bits.append("values: " + " | ".join(cq.sample_values))
            elif df[col].dtype == object and df[col].nunique(dropna=True) <= 25:
                vals = [str(v) for v in df[col].dropna().unique()[:12]]
                if vals:
                    bits.append("values: " + " | ".join(vals))
            if col in _FIELD_NOTES:
                bits.append("note: " + _FIELD_NOTES[col])
            lines.append("  ".join(bits))
        for n in q.notes:
            lines.append(f"  ! {n}")
        lines.append("")
    return "\n".join(lines)


def quality_summary(wh: Warehouse) -> dict:
    """Machine-readable summary for the UI panel and for answer caveats."""
    out: dict = {"tables": {}}
    for table in ("deals", "work_orders"):
        q = wh.quality[table]
        out["tables"][table] = {
            "board": q.board_name,
            "board_id": q.board_id,
            "rows": q.rows_kept,
            "dropped_header_rows": q.dropped_header_rows,
            "unusable_columns": [c.field for c in q.unusable_columns()],
            "low_coverage": {c.field: round(c.fill_rate, 3) for c in q.low_coverage()},
            "coercion_failures": {
                c.field: c.coercion_failures for c in q.columns if c.coercion_failures
            },
            "notes": q.notes,
        }
    return out
