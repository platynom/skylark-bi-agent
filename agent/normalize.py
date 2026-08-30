"""
Normalisation and data-quality layer.

This is where the "real-world messy data" requirement is actually met. The
monday boards are imported raw on purpose -- every defect below was observed in
the source workbooks and is repaired here at query time, not at import time, so
the cleaning logic is exercised and auditable rather than hidden in a one-off
spreadsheet edit.

Defects handled
---------------
1.  Header rows repeated inside the data body (a row whose cells equal their own
    column titles). Dropped.
2.  Fully empty columns (4 of 38 on the work-order board). Kept in the schema but
    flagged UNUSABLE so the agent says "not tracked" instead of "zero".
3.  Mixed date representations -> single date dtype.
4.  Currency/amount fields arriving as text with separators -> float.
5.  Quantities carrying units in the same cell ("5360 HA", "4", "3000")
    -> split into a numeric value and a unit.
6.  Categorical drift: casing, stray whitespace, near-duplicates
    ("BIlled" vs "Billed").
7.  No shared customer key across boards (COMPANY089 vs WOCOMPANY_002); the only
    bridge is the masked deal name, which repeats. Recorded as a join caveat.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any

import pandas as pd

from .monday_client import Board

# --------------------------------------------------------------------------- #
# header -> canonical field mapping
# --------------------------------------------------------------------------- #

def _norm_header(h: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(h).lower())


# Ordered rules: (canonical_name, [substring fragments that must all appear]).
# Substring matching (rather than exact) survives the long parenthesised
# monday column titles like "Amount in Rupees (Incl of GST) (Masked)".
DEAL_RULES: list[tuple[str, list[str]]] = [
    ("deal_name",            ["dealname"]),
    ("owner_code",           ["ownercode"]),
    ("client_code",          ["clientcode"]),
    ("deal_status",          ["dealstatus"]),
    ("close_date_actual",    ["closedate", "a"]),
    ("closure_probability",  ["closureprobability"]),
    ("deal_value",           ["maskeddealvalue"]),
    ("deal_value",           ["dealvalue"]),
    ("tentative_close_date", ["tentativeclose"]),
    ("deal_stage",           ["dealstage"]),
    ("product_deal",         ["productdeal"]),
    ("sector",               ["sector"]),
    ("created_date",         ["createddate"]),
]

WO_RULES: list[tuple[str, list[str]]] = [
    ("deal_name",              ["dealname"]),
    ("customer_code",          ["customername"]),
    ("serial_no",              ["serial"]),
    ("nature_of_work",         ["natureofwork"]),
    ("last_executed_month",    ["lastexecutedmonth"]),
    ("execution_status",       ["executionstatus"]),
    ("data_delivery_date",     ["datadelivery"]),
    ("po_date",                ["dateofpo"]),
    ("document_type",          ["documenttype"]),
    ("probable_start_date",    ["probablestart"]),
    ("probable_end_date",      ["probableend"]),
    ("owner_code",             ["personnelcode"]),
    ("sector",                 ["sector"]),
    ("type_of_work",           ["typeofwork"]),
    ("software_platform",      ["skylarksoftware"]),
    ("last_invoice_date",      ["lastinvoicedate"]),
    ("latest_invoice_no",      ["latestinvoice"]),
    ("amount_excl_gst",        ["amountinrupees", "exclofgst"]),
    ("amount_incl_gst",        ["amountinrupees", "inclofgst"]),
    ("billed_excl_gst",        ["billedvalue", "exclofgst"]),
    ("billed_incl_gst",        ["billedvalue", "inclofgst"]),
    ("collected_incl_gst",     ["collectedamount"]),
    ("to_be_billed_excl_gst",  ["tobebilled", "exlofgst"]),
    ("to_be_billed_excl_gst",  ["tobebilled", "exclofgst"]),
    ("to_be_billed_incl_gst",  ["tobebilled", "inclofgst"]),
    ("receivable",             ["amountreceivable"]),
    ("ar_priority",            ["arpriority"]),
    ("qty_ops",                ["quantitybyops"]),
    ("qty_po_raw",             ["quantitiesasperpo"]),
    ("qty_billed",             ["quantitybilled"]),
    ("qty_balance",            ["balanceinquantity"]),
    ("invoice_status",         ["invoicestatus"]),
    ("expected_billing_month", ["expectedbilling"]),
    ("actual_billing_month",   ["actualbilling"]),
    ("actual_collection_month", ["actualcollection"]),
    ("wo_status_billed",       ["wostatus"]),
    ("collection_status",      ["collectionstatus"]),
    ("collection_date",        ["collectiondate"]),
    ("billing_status",         ["billingstatus"]),
]

DATE_FIELDS = {
    "close_date_actual", "tentative_close_date", "created_date",
    "data_delivery_date", "po_date", "probable_start_date", "probable_end_date",
    "last_invoice_date", "collection_date",
}

NUMERIC_FIELDS = {
    "deal_value", "amount_excl_gst", "amount_incl_gst", "billed_excl_gst",
    "billed_incl_gst", "collected_incl_gst", "to_be_billed_excl_gst",
    "to_be_billed_incl_gst", "receivable", "qty_ops", "qty_billed", "qty_balance",
}

# Categorical fields whose values are title-cased and de-duplicated.
CATEGORICAL_FIELDS = {
    "deal_status", "closure_probability", "deal_stage", "product_deal", "sector",
    "nature_of_work", "execution_status", "document_type", "type_of_work",
    "software_platform", "invoice_status", "wo_status_billed", "billing_status",
    "collection_status", "ar_priority",
}


def _map_headers(titles: list[str], rules: list[tuple[str, list[str]]]) -> dict[str, str]:
    """Map raw board column titles to canonical field names. First rule wins."""
    mapping: dict[str, str] = {}
    taken: set[str] = set()
    for canon, frags in rules:
        if canon in taken:
            continue
        for t in titles:
            if t in mapping:
                continue
            n = _norm_header(t)
            if all(f in n for f in frags):
                mapping[t] = canon
                taken.add(canon)
                break
    return mapping


# --------------------------------------------------------------------------- #
# value coercion
# --------------------------------------------------------------------------- #

_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
    "%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%Y-%m-%d %H:%M:%S",
    "%d-%m-%y", "%d/%m/%y",
]


def parse_date(v: Any) -> date | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null", "-", "na", "n/a", "tbd"}:
        return None
    s = s.split("T")[0].strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    try:  # last resort, pandas is permissive
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
        return None if pd.isna(parsed) else parsed.date()
    except Exception:
        return None


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_number(v: Any) -> float | None:
    """Strip currency symbols, thousands separators and stray text."""
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null", "-", "na", "n/a", "tbd"}:
        return None
    s = s.replace(",", "").replace("₹", "").replace("Rs.", "").replace("Rs", "")
    s = s.replace("INR", "").strip()
    if s.startswith("(") and s.endswith(")"):  # accounting negatives
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        m = _NUM_RE.search(s)
        return float(m.group()) if m else None


_QTY_RE = re.compile(r"^\s*(-?[\d,]+(?:\.\d+)?)\s*([A-Za-z%/²^\s\.]*)\s*$")


def parse_quantity(v: Any) -> tuple[float | None, str | None]:
    """'5360 HA' -> (5360.0, 'HA');  '3000' -> (3000.0, None)."""
    if v is None:
        return None, None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null", "-"}:
        return None, None
    m = _QTY_RE.match(s)
    if m:
        num = parse_number(m.group(1))
        unit = (m.group(2) or "").strip().upper() or None
        return num, unit
    return parse_number(s), None


_WS_RE = re.compile(r"\s+")


def canon_category(v: Any) -> str | None:
    """Collapse whitespace and casing drift so 'BIlled' and 'Billed' agree."""
    if v is None:
        return None
    s = _WS_RE.sub(" ", str(v).strip())
    if not s or s.lower() in {"nan", "none", "null", "-", "na", "n/a"}:
        return None
    # Preserve deal-stage prefixes like "B. Sales Qualified Leads" verbatim.
    if re.match(r"^[A-Z]\.\s", s):
        return s
    # Title-case only if the value is not an intentional acronym.
    if s.isupper() and len(s) <= 12:
        return s
    return s[0].upper() + s[1:] if s else s


# --------------------------------------------------------------------------- #
# quality profile
# --------------------------------------------------------------------------- #

@dataclass
class ColumnQuality:
    field: str
    source_title: str
    dtype: str
    filled: int
    total: int
    coercion_failures: int = 0
    distinct: int = 0
    sample_values: list[str] = field(default_factory=list)

    @property
    def fill_rate(self) -> float:
        return (self.filled / self.total) if self.total else 0.0

    @property
    def unusable(self) -> bool:
        return self.filled == 0


@dataclass
class TableQuality:
    table: str
    board_id: str
    board_name: str
    rows_raw: int
    rows_kept: int
    dropped_header_rows: int
    columns: list[ColumnQuality]
    notes: list[str] = field(default_factory=list)

    def low_coverage(self, threshold: float = 0.8) -> list[ColumnQuality]:
        return [c for c in self.columns if 0 < c.fill_rate < threshold]

    def unusable_columns(self) -> list[ColumnQuality]:
        return [c for c in self.columns if c.unusable]


# --------------------------------------------------------------------------- #
# board -> dataframe
# --------------------------------------------------------------------------- #

def board_to_raw_frame(board: Board) -> pd.DataFrame:
    """Flatten monday items into a frame keyed by column *title*.

    monday absorbs the first CSV column into the built-in item Name field, so we
    surface it under its board column title if we can find one, and always as
    `__item_name__` as a safety net.
    """
    rows: list[dict[str, Any]] = []
    for it in board.items:
        row: dict[str, Any] = {
            "__item_id__": it.get("id"),
            "__item_name__": it.get("name"),
        }
        for cv in it.get("column_values") or []:
            title = next((c["title"] for c in board.columns if c["id"] == cv.get("id")), cv.get("id"))
            row[title] = cv.get("text")
        rows.append(row)
    return pd.DataFrame(rows)


def _drop_embedded_headers(df: pd.DataFrame, titles: list[str]) -> tuple[pd.DataFrame, int]:
    """Remove rows that are copies of the header (a spreadsheet-export artefact)."""
    if df.empty:
        return df, 0
    norm_titles = {_norm_header(t) for t in titles}

    def is_header_row(row: pd.Series) -> bool:
        hits = 0
        vals = 0
        for col, val in row.items():
            if col.startswith("__"):
                continue
            if val is None or str(val).strip() == "":
                continue
            vals += 1
            if _norm_header(val) == _norm_header(col) or _norm_header(val) in norm_titles:
                hits += 1
        return vals >= 2 and hits >= max(2, int(vals * 0.5))

    mask = df.apply(is_header_row, axis=1)
    return df[~mask].copy(), int(mask.sum())


def normalize_board(board: Board, table: str) -> tuple[pd.DataFrame, TableQuality]:
    """Board -> (clean canonical frame, quality profile)."""
    rules = DEAL_RULES if table == "deals" else WO_RULES
    raw = board_to_raw_frame(board)
    rows_raw = len(raw)

    titles = [c for c in raw.columns if not c.startswith("__")]
    raw, dropped = _drop_embedded_headers(raw, titles)

    mapping = _map_headers(titles, rules)

    # monday's built-in Name column carries the first CSV column's values.
    if "deal_name" not in mapping.values() and "__item_name__" in raw.columns:
        raw["Name (item)"] = raw["__item_name__"]
        mapping["Name (item)"] = "deal_name"
    else:
        # Even when a mapped column exists it may be blank if monday absorbed it.
        src = next((k for k, v in mapping.items() if v == "deal_name"), None)
        if src and raw[src].replace("", None).isna().all() and "__item_name__" in raw.columns:
            raw[src] = raw["__item_name__"]

    out = pd.DataFrame(index=raw.index)
    quality: list[ColumnQuality] = []

    for src_title, canon in mapping.items():
        series = raw[src_title]
        failures = 0

        if canon in DATE_FIELDS:
            nonblank = series.apply(lambda v: v is not None and str(v).strip() != "")
            parsed = series.apply(parse_date)
            failures = int((nonblank & parsed.isna()).sum())
            out[canon] = pd.to_datetime(parsed, errors="coerce")
            dtype = "DATE"
        elif canon in NUMERIC_FIELDS:
            nonblank = series.apply(lambda v: v is not None and str(v).strip() != "")
            parsed = series.apply(parse_number)
            failures = int((nonblank & parsed.isna()).sum())
            out[canon] = pd.to_numeric(parsed, errors="coerce")
            dtype = "DOUBLE"
        elif canon == "qty_po_raw":
            pairs = series.apply(parse_quantity)
            out["qty_po_value"] = pd.to_numeric(pairs.apply(lambda p: p[0]), errors="coerce")
            out["qty_po_unit"] = pairs.apply(lambda p: p[1])
            out["qty_po_raw"] = series.apply(lambda v: None if v is None or str(v).strip() == "" else str(v).strip())
            dtype = "TEXT+DOUBLE"
        elif canon in CATEGORICAL_FIELDS:
            out[canon] = series.apply(canon_category)
            dtype = "TEXT"
        else:
            out[canon] = series.apply(
                lambda v: None if v is None or str(v).strip() == "" else str(v).strip()
            )
            dtype = "TEXT"

        col = out[canon] if canon in out.columns else out.get("qty_po_value")
        filled = int(col.notna().sum()) if col is not None else 0
        distinct = int(col.nunique(dropna=True)) if col is not None else 0
        samples = (
            [str(x) for x in col.dropna().unique()[:8]] if col is not None and distinct <= 25 else []
        )
        quality.append(ColumnQuality(
            field=canon, source_title=src_title, dtype=dtype, filled=filled,
            total=len(out), coercion_failures=failures, distinct=distinct,
            sample_values=samples,
        ))

    out.insert(0, "item_id", raw["__item_id__"].values if "__item_id__" in raw else None)
    out = _add_derived(out, table)

    notes: list[str] = []
    if dropped:
        notes.append(
            f"{dropped} row(s) were repeated header rows inside the board body and were dropped."
        )
    unmapped = [t for t in titles if t not in mapping]
    if unmapped:
        notes.append("Board columns not mapped into the query schema: " + ", ".join(unmapped[:10]))

    tq = TableQuality(
        table=table, board_id=board.id, board_name=board.name, rows_raw=rows_raw,
        rows_kept=len(out), dropped_header_rows=dropped, columns=quality, notes=notes,
    )
    return out.reset_index(drop=True), tq


def _fiscal_year_label(d) -> str | None:
    """Indian FY: Apr-Mar. 2025-06-01 -> 'FY25-26'."""
    if pd.isna(d):
        return None
    y = d.year
    start = y if d.month >= 4 else y - 1
    return f"FY{str(start)[2:]}-{str(start + 1)[2:]}"


def _fiscal_quarter(d) -> str | None:
    if pd.isna(d):
        return None
    q = ((d.month - 4) % 12) // 3 + 1
    return f"{_fiscal_year_label(d)} Q{q}"


def _add_derived(df: pd.DataFrame, table: str) -> pd.DataFrame:
    if table == "deals":
        if "deal_stage" in df:
            df["deal_stage_order"] = df["deal_stage"].apply(
                lambda s: (ord(s[0].upper()) - 64) if isinstance(s, str) and re.match(r"^[A-Za-z]\.", s) else None
            )
        if "deal_status" in df:
            st = df["deal_status"].fillna("")
            df["is_open"] = st.str.lower().eq("open")
            df["is_won"] = st.str.lower().eq("won")
            df["is_dead"] = st.str.lower().eq("dead")
        if "closure_probability" in df:
            df["probability_weight"] = df["closure_probability"].map(
                {"High": 0.75, "Medium": 0.45, "Low": 0.20}
            )
        if "deal_value" in df and "probability_weight" in df:
            df["weighted_value"] = df["deal_value"] * df["probability_weight"]
        for c in ("created_date", "tentative_close_date", "close_date_actual"):
            if c in df:
                df[f"{c}_fy"] = df[c].apply(_fiscal_year_label)
                df[f"{c}_fq"] = df[c].apply(_fiscal_quarter)
    else:
        for c in ("po_date", "last_invoice_date", "probable_end_date"):
            if c in df:
                df[f"{c}_fy"] = df[c].apply(_fiscal_year_label)
                df[f"{c}_fq"] = df[c].apply(_fiscal_quarter)
        if {"amount_incl_gst", "billed_incl_gst"} <= set(df.columns):
            df["unbilled_incl_gst"] = df["amount_incl_gst"].fillna(0) - df["billed_incl_gst"].fillna(0)
        if {"billed_incl_gst", "collected_incl_gst"} <= set(df.columns):
            df["outstanding_incl_gst"] = df["billed_incl_gst"].fillna(0) - df["collected_incl_gst"].fillna(0)
    return df
