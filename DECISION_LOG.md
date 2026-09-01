# Decision Log — Skylark BI Agent

## 1. Key assumptions

**Boards are imported raw, on purpose.** The CSVs went into monday.com untouched —
including the two repeated header rows sitting inside the data body. Cleaning at import
time would have hidden the exact capability the brief asks for. All repair happens at
query time in `agent/normalize.py`, where it is testable and visible to the user.

**The two boards cannot be joined on customer.** Deals use `COMPANY089`; work orders use
`WOCOMPANY_002`. These are different masking schemes over (probably) the same customers,
but nothing in the data proves the mapping. The only bridge is the masked deal name, and
it is not unique — 155 distinct names across 344 deal rows. The planner is instructed to
aggregate each side to one row per `deal_name` before joining, and to declare the join as
an assumption. **Assuming a customer-level join here would silently double-count revenue,**
which is the single most dangerous error available in this dataset.

**Closure probability is categorical, so weights are ours.** `High/Medium/Low` were mapped
to 0.75 / 0.45 / 0.20 to make a weighted pipeline possible. This is our number, not
Skylark's; it is labelled as such in the schema and in every answer that uses it. Only
88 of 346 deals carry a probability at all.

**Fiscal year is Indian (April–March).** "This quarter" resolves to a fiscal quarter, not
a calendar one. Pre-computed `*_fy` / `*_fq` columns exist so the model never does date
arithmetic by hand.

**Empty is not zero.** Four work-order columns (`Expected Billing Month`,
`Actual Collection Month`, `Collection status`, `Collection Date`) have no values at all.
They stay in the schema, flagged `UNUSABLE`, and the agent is instructed to answer
"not tracked" rather than returning 0. A confident zero is worse than an admitted gap.

**Amounts are masked/scaled.** Formatted as INR with lakh/crore framing; no conversions.

## 2. Trade-offs

**LLM-generated SQL over a fixed toolset.** A fixed set of Python analytics functions is
safer and more predictable, but only answers the questions we anticipated. Founder
questions are open-ended by definition. We load the normalised frames into in-memory
DuckDB and let the model write SQL against a schema document that is *generated from the
live boards at runtime* — column names, fill rates and distinct values are read off the
data, not hardcoded. Cost: the model can write wrong SQL. Mitigations: a read-only
statement guard that rejects anything but a single `SELECT`/`WITH`, a self-correction loop
that feeds the DuckDB error back for up to two retries, and the executed SQL displayed with
every answer so the user can audit any number.

**Two model calls per answer (plan, then narrate) instead of one.** Costs latency. Buys a
temperature-0 SQL step separate from a warmer prose step, and means a failed SQL attempt
never spends a narration call.

**Leadership-update numbers are computed in pandas, never by the model.** The model
receives a finished metrics object and writes prose around it. A hallucinated figure in a
board deck is unrecoverable; slightly stiffer prose is not.

**Shared warehouse and question caches instead of per-turn fetch.** The brief forbids
hardcoding CSV data, not caching. Re-pulling 520 items on every serverless invocation burns
monday's complexity budget and adds seconds to each answer. Vercel therefore uses a
30-minute Supabase snapshot containing Parquet dataframes plus schema/quality metadata, and
a separate 15-minute normalized-question cache for verified responses. Either cache can fail
open to the live pipeline; Streamlit retains its 300-second in-process TTL and manual refresh.

**Gemini over explicit HTTP transports, no vendor SDK.** The Google GenAI Python packages
have shipped several incompatible interfaces; a hosted demo that breaks because a transitive
pin moved is a bad trade for two convenience methods. `requests` plus documented Vertex and
AI Studio contracts keeps authentication, timeouts, provider identity, and failover visible.

**Streamlit on Streamlit Community Cloud.** A React/FastAPI split would demo better for a
full-stack role. With a five-hour budget, infrastructure time is time not spent on the data
layer — which is where this brief actually concentrates its difficulty. Free, public,
secrets-managed, zero deploy config.

## 3. How I interpreted "leadership updates"

As the recurring board/investor block someone currently rebuilds by hand each month:
pipeline size and weighted view, stage and sector concentration, win rate, contracted vs
billed vs collected, cash exposure, execution status, and a rupee-ranked risk register
(billing stuck, AR priority accounts, open deals past their tentative close date, open
deals with no value recorded).

The section I consider load-bearing is the last one: **"What this update cannot tell you."**
It is generated from the live quality profile, not written by hand. Collection dates aren't
tracked, so no DSO or ageing bucket is possible; deal value covers 48% of rows, so pipeline
totals are floors, not totals. Silent omission is how bad board decks get made, and an
agent that quietly returns a number for a column with no data is worse than one that
refuses. An emphasis box lets a founder steer the draft ("the board is worried about
collections") without letting them steer the arithmetic.

## 4. What I would do differently with more time

1. **Resolve the customer-key ambiguity properly** — fuzzy-match `COMPANY*` to `WOCOMPANY*`
   via deal name, sector and amount overlap, score the mapping, and expose it as a
   reviewable crosswalk table rather than refusing the join.
2. **Automate evals in CI.** The repository now has a canonical/paraphrase suite and a
   separately scored held-out suite; the next step is running both against candidate prompts
   before every production merge and retaining score history.
3. **Charts.** Numbers plus a stage funnel and an AR ageing bar would land better in a
   leadership context than a table.
4. **Push the read-only guarantee into the token**, not just the client. The code refuses
   mutations, but a scoped monday token would make it structural.
5. **Keyless Vertex authentication** — use Vercel OIDC/workload identity federation so the
   primary leg can authenticate without a service-account key.

## 5. AI tools used

Claude (Opus) for architecture, implementation and this document. All design decisions,
the data-defect inventory, and the test strategy were reviewed and are defensible line by
line. Gemini 3.5 Flash Lite is the runtime model inside the product itself.

## 6. Provider redundancy and model consistency

The production LLM chain is deliberately **Vertex AI `gemini-3.5-flash-lite` on the
global endpoint → Gemini AI Studio `gemini-3.5-flash-lite` on v1beta → deterministic
templates**. Vertex and AI Studio use separate quota pools, so a project quota event or
provider outage does not remove both LLM legs. Both legs intentionally run the same model:
failover therefore changes transport and quota pool, not answer behaviour. This keeps the
validated planner semantics stable regardless of which provider serves a request.

No additional Gemini model is placed in the automatic chain. In particular,
`gemini-2.5-flash` was ruled out as a primary because its measured p95 approached the
20-second request budget and it had materially more genuine quality failures. A longer
same-project Vertex model cascade would add apparent legs without adding independent
failure domains.

The deterministic template floor is intentionally part of the product contract, not just
an emergency error string. Planning and execution can succeed even when narration quota is
exhausted; discarding those verified rows and telling the user that the whole query failed
would be both slower and less truthful. Templates cover common scalar, ranked, grouped, and
list results, preserve assumptions, and use server-formatted currency. They provide useful
first content during the two-phase flow and a complete answer if every LLM leg fails.

Vertex is configured in production but not yet authenticated. The organization policy
`constraints/iam.disableServiceAccountKeyCreation` prevents the service-account key path,
so the observed healthy chain is currently `vertex (error) → ai_studio`, followed by
circuit-breaker skips of Vertex. Keyless OIDC federation is deferred work; this expected
transport failure is not a reason to revert otherwise correct application code.
