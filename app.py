"""
Skylark BI Agent -- Streamlit front end.

Run:  streamlit run app.py
Secrets required: MONDAY_API_TOKEN, GEMINI_API_KEY (see README).
"""
from __future__ import annotations

import traceback

import pandas as pd
import streamlit as st

from agent import config
from agent.agent import answer_question
from agent.leadership import compute_metrics, generate_update
from agent.llm import Gemini, LLMError
from agent.monday_client import MondayClient, MondayError
from agent.warehouse import build_warehouse, quality_summary

st.set_page_config(page_title="Skylark BI Agent", page_icon="::", layout="wide")

SAMPLE_QUESTIONS = [
    "How's our pipeline looking for the renewables sector?",
    "What's our total outstanding receivable, and which sector is it concentrated in?",
    "Which work orders are completed but haven't been invoiced yet?",
    "What's our win rate, and does it differ by sector?",
    "Which deal owner has the largest open pipeline?",
    "How much revenue have we contracted in mining vs renewables?",
    "Which quarter does our open pipeline actually close in?",
]


# --------------------------------------------------------------------------- #
# resources
# --------------------------------------------------------------------------- #
@st.cache_resource(ttl=config.CACHE_TTL_SECONDS, show_spinner=False)
def load_warehouse(_bust: int = 0):
    """Live fetch from monday.com. Cached briefly so a conversation doesn't
    re-pull 500+ items on every turn, but never longer than CACHE_TTL_SECONDS
    so answers stay current."""
    return build_warehouse()


@st.cache_resource(show_spinner=False)
def get_llm():
    return Gemini()


def _fatal(title: str, detail: str, hint: str = "") -> None:
    st.error(f"**{title}**\n\n{detail}" + (f"\n\n{hint}" if hint else ""))
    st.stop()


# --------------------------------------------------------------------------- #
# boot
# --------------------------------------------------------------------------- #
st.title("Skylark BI Agent")
st.caption(
    "Founder-level business intelligence over live monday.com boards. "
    "Every answer shows the SQL it ran and the data-quality caveats behind it."
)

if not config.MONDAY_API_TOKEN:
    _fatal("monday.com token missing", "`MONDAY_API_TOKEN` is not configured.",
           "Add it under Streamlit Cloud -> Settings -> Secrets, or export it locally.")
if not config.GEMINI_API_KEY:
    _fatal("Gemini key missing", "`GEMINI_API_KEY` is not configured.",
           "Get one at https://aistudio.google.com/apikey and add it to secrets.")

if "bust" not in st.session_state:
    st.session_state.bust = 0
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

try:
    with st.spinner("Reading live data from monday.com..."):
        wh = load_warehouse(st.session_state.bust)
except MondayError as exc:
    _fatal("Cannot reach monday.com", str(exc),
           "Verify the API token and that both boards exist and are visible to it.")
except Exception as exc:  # noqa: BLE001
    _fatal("Failed to load board data", f"{type(exc).__name__}: {exc}",
           f"```\n{traceback.format_exc()[-1200:]}\n```")

try:
    llm = get_llm()
except LLMError as exc:
    _fatal("Gemini unavailable", str(exc))


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.subheader("Connection")
    q = quality_summary(wh)
    for t in ("deals", "work_orders"):
        info = q["tables"][t]
        st.markdown(
            f"**{info['board']}** &nbsp;`{info['board_id']}`  \n"
            f"{info['rows']} rows loaded"
            + (f"  \n:orange[{info['dropped_header_rows']} header row(s) dropped]"
               if info["dropped_header_rows"] else "")
        )
    st.caption(f"Data fetched {int(wh.age_seconds)}s ago "
               f"(auto-refresh every {config.CACHE_TTL_SECONDS}s)")
    if st.button("Refresh from monday.com", use_container_width=True):
        st.session_state.bust += 1
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.subheader("Data quality")
    for t in ("deals", "work_orders"):
        info = q["tables"][t]
        with st.expander(f"{t} ({info['rows']} rows)"):
            if info["unusable_columns"]:
                st.markdown("**Never populated** (agent will say *not tracked*, not zero):")
                st.markdown("\n".join(f"- `{c}`" for c in info["unusable_columns"]))
            if info["low_coverage"]:
                st.markdown("**Partial coverage:**")
                st.markdown("\n".join(
                    f"- `{k}` — {v:.0%}" for k, v in
                    sorted(info["low_coverage"].items(), key=lambda kv: kv[1])[:12]
                ))
            if info["coercion_failures"]:
                st.markdown("**Values that failed type coercion:**")
                st.markdown("\n".join(f"- `{k}` — {v}" for k, v in info["coercion_failures"].items()))
            for n in info["notes"]:
                st.info(n)

    st.divider()
    st.caption("Read-only. This app never writes to monday.com.")


# --------------------------------------------------------------------------- #
# tabs
# --------------------------------------------------------------------------- #
tab_chat, tab_update, tab_data = st.tabs(["Ask", "Leadership update", "Data"])

with tab_chat:
    if not st.session_state.messages:
        st.markdown("**Try one of these:**")
        cols = st.columns(2)
        for i, s in enumerate(SAMPLE_QUESTIONS):
            if cols[i % 2].button(s, key=f"s{i}", use_container_width=True):
                st.session_state.pending = s
                st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            meta = msg.get("meta") or {}
            if meta.get("assumptions"):
                st.caption("Assumptions: " + "; ".join(meta["assumptions"]))
            if meta.get("sql"):
                with st.expander("SQL executed"):
                    st.code(meta["sql"], language="sql")
                    if meta.get("attempts"):
                        st.caption(f"{len(meta['attempts'])} earlier attempt(s) self-corrected:")
                        for a in meta["attempts"]:
                            st.code(a["sql"], language="sql")
                            st.caption(a["error"][:300])
            if meta.get("rows") is not None:
                with st.expander(f"Result data ({meta['rowcount']} rows)"):
                    st.dataframe(pd.DataFrame(meta["rows"]), use_container_width=True)

    typed = st.chat_input("Ask about pipeline, revenue, sectors, collections, execution...")
    question = typed or st.session_state.pending
    st.session_state.pending = None

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history = [{"role": m["role"], "content": m["content"]}
                           for m in st.session_state.messages[:-1]]
                turn = answer_question(wh, question, llm=llm, history=history)

            if turn.action == "clarify":
                body = f"{turn.clarify}"
                if turn.options:
                    body += "\n\n" + "\n".join(f"- {o}" for o in turn.options)
                st.markdown(body)
                st.session_state.messages.append({"role": "assistant", "content": body})
            elif turn.action == "error":
                body = f":red[{turn.error}]"
                st.markdown(body)
                st.session_state.messages.append({"role": "assistant", "content": body})
            else:
                body = turn.answer or "(no answer produced)"
                st.markdown(body)
                if turn.assumptions:
                    st.caption("Assumptions: " + "; ".join(turn.assumptions))
                if turn.sql:
                    with st.expander("SQL executed"):
                        st.code(turn.sql, language="sql")
                if turn.result is not None and not turn.result.empty:
                    with st.expander(f"Result data ({len(turn.result)} rows)"):
                        st.dataframe(turn.result, use_container_width=True)
                st.session_state.messages.append({
                    "role": "assistant", "content": body,
                    "meta": {
                        "sql": turn.sql,
                        "assumptions": turn.assumptions,
                        "attempts": turn.attempts,
                        "rows": turn.result.head(200).to_dict("records") if turn.result is not None else None,
                        "rowcount": len(turn.result) if turn.result is not None else 0,
                    },
                })

with tab_update:
    st.markdown(
        "Generates the recurring block a founder would otherwise rebuild by hand: "
        "pipeline, conversion, revenue realisation, cash exposure, execution risk, "
        "and an explicit list of what the data cannot support. "
        "**Every figure is computed in pandas, not by the model** — the model only writes prose "
        "around numbers it was handed."
    )
    focus = st.text_input("Optional emphasis (e.g. 'board is worried about collections')")
    if st.button("Generate leadership update", type="primary"):
        with st.spinner("Computing metrics and drafting..."):
            narrative, metrics = generate_update(wh, llm=llm, focus=focus or None)
        st.markdown(narrative)
        st.download_button("Download as Markdown", narrative,
                           file_name="skylark_leadership_update.md", mime="text/markdown")
        with st.expander("Underlying computed metrics (source of every number above)"):
            st.json(metrics)

with tab_data:
    st.markdown("Normalised tables exactly as the SQL engine sees them.")
    which = st.radio("Table", ["deals", "work_orders"], horizontal=True)
    df = wh.deals if which == "deals" else wh.work_orders
    st.caption(f"{len(df)} rows x {len(df.columns)} columns")
    st.dataframe(df, use_container_width=True, height=520)
