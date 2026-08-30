"""Offline validation of normalisation + warehouse against the source fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import pandas as pd

from agent.normalize import (canon_category, normalize_board, parse_date,
                             parse_number, parse_quantity)
from tests.mock_monday import board_from_csv

FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def build():
    d = normalize_board(board_from_csv(FIX / "deals.csv", "111", "Deals"), "deals")
    w = normalize_board(board_from_csv(FIX / "work_orders.csv", "222", "Work Orders"), "work_orders")
    return d, w


def main() -> int:
    fails = []

    def check(name, cond, extra=""):
        print(("PASS  " if cond else "FAIL  ") + name + (f"   {extra}" if extra else ""))
        if not cond:
            fails.append(name)

    # ---- unit coercion ----
    check("parse_number strips separators", parse_number("1,23,456.50") == 123456.50)
    check("parse_number handles rupee symbol", parse_number("Rs 5,000") == 5000.0)
    check("parse_number rejects junk", parse_number("n/a") is None)
    check("parse_date iso", str(parse_date("2025-06-01")) == "2025-06-01")
    check("parse_date dd-mm-yyyy", str(parse_date("01-06-2025")) == "2025-06-01")
    check("parse_date junk -> None", parse_date("TBD") is None)
    check("parse_quantity with unit", parse_quantity("5360 HA") == (5360.0, "HA"))
    check("parse_quantity bare", parse_quantity("3000") == (3000.0, None))
    check("canon_category preserves stage prefix",
          canon_category("B. Sales Qualified Leads") == "B. Sales Qualified Leads")
    check("canon_category fixes casing drift", canon_category("  billed ") == "Billed")

    # ---- normalisation ----
    (dd, dq), (wd, wq) = build()
    check("deals rows loaded", len(dd) > 300, f"{len(dd)} rows")
    check("work_orders rows loaded", len(wd) > 150, f"{len(wd)} rows")
    check("embedded header rows dropped from deals", dq.dropped_header_rows >= 1,
          f"{dq.dropped_header_rows} dropped")
    check("deal_name recovered from monday item name",
          dd["deal_name"].notna().sum() > 300, f"{dd['deal_name'].notna().sum()} filled")
    check("wo deal_name recovered", wd["deal_name"].notna().sum() > 150)

    check("deal_value is numeric", pd.api.types.is_numeric_dtype(dd["deal_value"]))
    check("created_date is datetime", pd.api.types.is_datetime64_any_dtype(dd["created_date"]))
    check("no 'Deal Status' junk value survives",
          "Deal Status" not in set(dd["deal_status"].dropna()))
    check("deal_status values sane",
          set(dd["deal_status"].dropna()) <= {"Won", "Dead", "Open", "On Hold"},
          str(sorted(set(dd["deal_status"].dropna()))))

    unusable = [c.field for c in wq.unusable_columns()]
    check("empty columns flagged unusable", len(unusable) >= 3, str(unusable))
    check("expected_billing_month flagged", "expected_billing_month" in unusable)

    check("qty split into value+unit", "qty_po_value" in wd and "qty_po_unit" in wd)
    check("qty units detected", wd["qty_po_unit"].notna().sum() > 0,
          str(wd["qty_po_unit"].dropna().unique()[:6]))

    check("fiscal quarter derived", dd["created_date_fq"].notna().sum() > 300,
          str(dd["created_date_fq"].dropna().unique()[:3]))
    check("weighted_value derived", "weighted_value" in dd)
    check("outstanding derived", "outstanding_incl_gst" in wd)

    # ---- warehouse SQL ----
    con = duckdb.connect(":memory:")
    con.register("d", dd); con.register("w", wd)
    con.execute("CREATE TABLE deals AS SELECT * FROM d")
    con.execute("CREATE TABLE work_orders AS SELECT * FROM w")

    open_v = con.execute(
        "SELECT count(*) n, count(deal_value) cov, sum(deal_value) v "
        "FROM deals WHERE deal_status='Open'").fetchone()
    check("open pipeline queryable", open_v[0] > 0, f"open={open_v[0]} valued={open_v[1]} sum={open_v[2]}")

    ar = con.execute("SELECT sum(billed_incl_gst)-sum(collected_incl_gst) FROM work_orders").fetchone()[0]
    check("AR computable", ar is not None, f"outstanding={ar:,.0f}")

    sec = con.execute(
        "SELECT sector, count(*) n FROM deals WHERE deal_status='Open' "
        "GROUP BY 1 ORDER BY 2 DESC").fetchall()
    check("sector breakdown works", len(sec) > 2, str(sec[:4]))

    # ---- ground truth cross-check against raw CSV ----
    raw = pd.read_csv(FIX / "deals.csv")
    raw = raw[raw["Deal Status"] != "Deal Status"]
    raw_open = (raw["Deal Status"] == "Open").sum()
    check("open-deal count matches raw CSV", int(open_v[0]) == int(raw_open),
          f"agent={open_v[0]} raw={raw_open}")

    raw_val = pd.to_numeric(raw.loc[raw["Deal Status"] == "Open", "Masked Deal value"],
                            errors="coerce").sum()
    check("open-deal value matches raw CSV", abs(float(open_v[2] or 0) - float(raw_val)) < 1,
          f"agent={open_v[2]:,.0f} raw={raw_val:,.0f}")

    print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
