"""
Leadership update generator.

INTERPRETATION OF THE OPTIONAL REQUIREMENT
------------------------------------------
"Help prepare data for leadership updates" is read as: produce the recurring
board/investor-update block that someone currently rebuilds by hand every month --
pipeline, conversion, revenue realisation, cash exposure, execution risk -- with
the data-quality footnotes that make it defensible.

Two design choices worth defending:
  * Every number here is computed in pandas, NOT by the LLM. A leadership update
    that contains a hallucinated figure is worse than no update. The model only
    writes the narrative around numbers it was handed.
  * The pack always ships a "what this update cannot tell you" section, driven by
    the live quality profile. Silent omissions are how bad board decks happen.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from .llm import Gemini, LLMError
from .warehouse import Warehouse


def _inr(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    x = float(x)
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1e7:
        return f"{sign}Rs {a/1e7:,.2f} Cr"
    if a >= 1e5:
        return f"{sign}Rs {a/1e5:,.2f} L"
    return f"{sign}Rs {a:,.0f}"


def _cov(series: pd.Series) -> str:
    n, t = int(series.notna().sum()), len(series)
    return f"{n}/{t} records ({n/t:.0%})" if t else "0 records"


def compute_metrics(wh: Warehouse) -> dict[str, Any]:
    d, w = wh.deals, wh.work_orders
    m: dict[str, Any] = {"generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                         "data_age_seconds": round(wh.age_seconds)}

    # ---------------- pipeline ----------------
    if "deal_status" in d:
        opend = d[d["deal_status"].str.lower().eq("open")] if d["deal_status"].notna().any() else d.iloc[0:0]
        won = d[d["deal_status"].str.lower().eq("won")]
        dead = d[d["deal_status"].str.lower().eq("dead")]
    else:
        opend = won = dead = d.iloc[0:0]

    m["pipeline"] = {
        "open_deals": len(opend),
        "open_value": float(opend["deal_value"].sum()) if "deal_value" in opend else None,
        "open_value_coverage": _cov(opend["deal_value"]) if "deal_value" in opend else "n/a",
        "weighted_value": float(opend["weighted_value"].sum()) if "weighted_value" in opend else None,
        "weighted_coverage": _cov(opend["weighted_value"]) if "weighted_value" in opend else "n/a",
    }
    if "deal_stage" in opend and len(opend):
        by_stage = (opend.groupby("deal_stage", dropna=False)
                    .agg(deals=("deal_stage", "size"), value=("deal_value", "sum"))
                    .sort_index())
        m["pipeline"]["by_stage"] = by_stage.reset_index().to_dict("records")
    if "sector" in opend and len(opend):
        by_sec = (opend.groupby("sector", dropna=False)
                  .agg(deals=("sector", "size"),
                       value=("deal_value", "sum"),
                       with_value=("deal_value", "count"))
                  .sort_values("value", ascending=False))
        m["pipeline"]["by_sector"] = by_sec.reset_index().to_dict("records")
    if "deal_value" in opend and len(opend):
        top = opend.nlargest(5, "deal_value")[
            [c for c in ("deal_name", "client_code", "sector", "deal_stage",
                         "deal_value", "closure_probability") if c in opend]
        ]
        m["pipeline"]["top_deals"] = top.to_dict("records")

    # ---------------- conversion ----------------
    closed = len(won) + len(dead)
    m["conversion"] = {
        "won": len(won), "dead": len(dead), "on_hold": int(len(d) - len(opend) - closed),
        "win_rate": (len(won) / closed) if closed else None,
        "won_value": float(won["deal_value"].sum()) if "deal_value" in won else None,
        "won_value_coverage": _cov(won["deal_value"]) if "deal_value" in won else "n/a",
    }

    # ---------------- revenue & cash ----------------
    def s(col: str) -> float | None:
        return float(w[col].sum()) if col in w else None

    contracted = s("amount_incl_gst")
    billed = s("billed_incl_gst")
    collected = s("collected_incl_gst")
    m["revenue"] = {
        "work_orders": len(w),
        "contracted_incl_gst": contracted,
        "billed_incl_gst": billed,
        "collected_incl_gst": collected,
        "unbilled_incl_gst": (contracted - billed) if None not in (contracted, billed) else None,
        "outstanding_incl_gst": (billed - collected) if None not in (billed, collected) else None,
        "billing_ratio": (billed / contracted) if contracted else None,
        "collection_ratio": (collected / billed) if billed else None,
        "collected_coverage": _cov(w["collected_incl_gst"]) if "collected_incl_gst" in w else "n/a",
    }
    if "sector" in w:
        sec = (w.groupby("sector", dropna=False)
               .agg(work_orders=("sector", "size"),
                    contracted=("amount_incl_gst", "sum"),
                    billed=("billed_incl_gst", "sum"),
                    collected=("collected_incl_gst", "sum"))
               .sort_values("contracted", ascending=False))
        m["revenue"]["by_sector"] = sec.reset_index().to_dict("records")

    # ---------------- execution ----------------
    if "execution_status" in w:
        m["execution"] = {
            "by_status": w["execution_status"].fillna("(blank)").value_counts().to_dict()
        }
        done = w["execution_status"].fillna("").str.lower().eq("completed")
        if "billed_incl_gst" in w:
            unbilled_done = w[done & (w["billed_incl_gst"].fillna(0) <= 0)]
            m["execution"]["completed_but_unbilled"] = {
                "count": len(unbilled_done),
                "value": float(unbilled_done["amount_incl_gst"].sum()) if "amount_incl_gst" in unbilled_done else None,
            }

    # ---------------- risk register ----------------
    risks: list[dict[str, Any]] = []
    if "billing_status" in w:
        stuck = w[w["billing_status"].fillna("").str.lower().isin(["stuck", "update required"])]
        if len(stuck):
            risks.append({
                "risk": "Billing blocked or unreconciled",
                "count": len(stuck),
                "value": float(stuck["amount_incl_gst"].sum()) if "amount_incl_gst" in stuck else None,
                "detail": stuck["billing_status"].value_counts().to_dict(),
            })
    if "ar_priority" in w:
        pri = w[w["ar_priority"].notna()]
        if len(pri):
            risks.append({
                "risk": "AR priority accounts",
                "count": len(pri),
                "value": float(pri["outstanding_incl_gst"].sum()) if "outstanding_incl_gst" in pri else None,
            })
    if "tentative_close_date" in d and len(opend):
        slipped = opend[opend["tentative_close_date"] < pd.Timestamp(dt.date.today())]
        if len(slipped):
            risks.append({
                "risk": "Open deals past their tentative close date",
                "count": len(slipped),
                "value": float(slipped["deal_value"].sum()) if "deal_value" in slipped else None,
            })
    if "deal_value" in opend and len(opend):
        missing_val = int(opend["deal_value"].isna().sum())
        if missing_val:
            risks.append({
                "risk": "Open deals with no value recorded (pipeline is understated)",
                "count": missing_val, "value": None,
            })
    m["risks"] = risks

    # ---------------- what this cannot tell you ----------------
    limits: list[str] = []
    for table in ("deals", "work_orders"):
        q = wh.quality[table]
        for c in q.unusable_columns():
            limits.append(f"{table}.{c.field} is never populated -- not tracked on the board.")
        for c in q.low_coverage(0.6):
            limits.append(f"{table}.{c.field} covers only {c.fill_rate:.0%} of records.")
        limits.extend(q.notes)
    m["limitations"] = limits
    return m


NARRATIVE_SYSTEM = """You are drafting the business section of a monthly leadership update
for the founders of Skylark Drones, a drone services company.

You are given a metrics object that has ALREADY been computed from live monday.com data.
Every number you write must come from that object verbatim. Do not compute new figures,
do not estimate, do not extrapolate. If a value is null, say it is not tracked.

Structure:
**Headline** - two or three sentences a founder could read aloud.
**Pipeline** - size, weighted view, stage concentration, sector mix.
**Revenue and cash** - contracted vs billed vs collected, and what the gap means.
**Execution** - delivery status and anything completed but not invoiced.
**Risks to flag** - the risk register, ranked by rupee exposure.
**What this update cannot tell you** - the limitations list, in plain language.

Rules: Indian rupee formatting with lakh/crore. Interpret, do not just list. Name the
specific driver behind each number. No emoji, no filler, no congratulation. Markdown.
"""


def generate_update(wh: Warehouse, llm: Gemini | None = None,
                    focus: str | None = None) -> tuple[str, dict[str, Any]]:
    metrics = compute_metrics(wh)
    llm = llm or Gemini()

    def fmt(o: Any, depth: int = 0) -> str:
        pad = "  " * depth
        if isinstance(o, dict):
            return "\n".join(f"{pad}{k}: {fmt(v, depth + 1)}" if isinstance(v, (dict, list))
                             else f"{pad}{k}: {_fmt_scalar(k, v)}" for k, v in o.items())
        if isinstance(o, list):
            return "\n" + "\n".join(f"{pad}- {fmt(i, depth + 1).strip()}" for i in o[:12])
        return str(o)

    def _fmt_scalar(k: str, v: Any) -> str:
        if v is None:
            return "not tracked"
        if isinstance(v, float) and any(t in k for t in ("value", "incl_gst", "contracted",
                                                         "billed", "collected", "outstanding")):
            return _inr(v)
        if isinstance(v, float) and "ratio" in k or (isinstance(v, float) and k == "win_rate"):
            return f"{v:.1%}"
        return str(v)

    payload = fmt(metrics)
    if focus:
        payload += f"\n\nFOUNDER ASKED THE UPDATE TO EMPHASISE: {focus}"

    try:
        narrative = llm.generate(NARRATIVE_SYSTEM, payload, temperature=0.3, max_tokens=2600)
    except LLMError as exc:
        narrative = f"_Narrative generation failed ({exc}). The computed metrics are shown below._"
    return narrative, metrics
