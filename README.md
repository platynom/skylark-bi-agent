# Skylark BI Agent

A conversational business-intelligence agent that answers founder-level questions over two
live monday.com boards — **Deals** (sales pipeline) and **Work Orders** (project execution
and billing).

> *"How's our pipeline looking for the renewables sector this quarter?"*
> → interprets the question, writes SQL against the live boards, runs it, and answers with
> context **plus the data-quality caveats that qualify the number.**

**Live app:** _<paste your Streamlit URL here>_

---

## Why this problem is mostly a data problem

The boards are real-world messy. Every defect below was found by profiling the source data
and is repaired at query time in `agent/normalize.py`:

| Defect | Where | Handling |
|---|---|---|
| Header rows repeated inside the data body | Deals (2 rows) | Detected structurally and dropped |
| 4 columns with zero values | Work Orders | Flagged `UNUSABLE`; agent answers *"not tracked"*, never `0` |
| `Masked Deal value` populated on 48% of rows | Deals (165/346) | Every total carries an explicit coverage caveat |
| Quantities with units in the same cell (`"5360 HA"`, `"3000"`) | Work Orders | Split into `qty_po_value` + `qty_po_unit` |
| Mixed date representations | Both | Coerced through a format ladder to a single `DATE` type |
| Casing/whitespace drift (`"BIlled"` vs `"Billed"`) | Work Orders | Canonicalised, stage prefixes (`"B. …"`) preserved |
| Three overlapping status fields | Work Orders | All exposed; the agent picks and states which it used |
| **No shared customer key** (`COMPANY089` vs `WOCOMPANY_002`) | Both | Cross-board joins forced through `deal_name`, aggregated per side first, flagged as an assumption |

The boards are imported **raw on purpose**. Cleaning during import would hide the exact
capability the assignment grades.

---

## Architecture

```
Streamlit UI  (app.py)
  │  chat · leadership update · normalised data browser
  ▼
Agent orchestration  (agent/agent.py)
  │  1. PLAN     Gemini → JSON: {sql | clarify | unsupported}, temp 0
  │  2. EXECUTE  DuckDB, read-only guard, self-correcting retry ×2
  │  3. NARRATE  Gemini → prose + only the caveats this query touched
  ▼
Warehouse  (agent/warehouse.py)
  │  in-memory DuckDB · runtime-generated schema document · SQL safety guard
  ▼
Normalisation  (agent/normalize.py)
  │  header→canonical mapping · type coercion · unit parsing · quality profiling
  ▼
monday.com client  (agent/monday_client.py)
     GraphQL v2 · cursor pagination · backoff on 429/complexity · mutation-refusing
```

**Read-only by construction.** `MondayClient._post` refuses to transmit any document
containing `mutation`, and `run_sql` rejects anything that is not a single `SELECT`/`WITH`.

**Nothing about the CSVs is hardcoded.** The schema handed to the model — column names,
fill rates, distinct values, join caveats — is generated at runtime from whatever the
boards currently contain. Rename a sector on the board and the next refresh reflects it.

### Key design decisions

- **Generated SQL over a fixed toolset** — a fixed toolset answers only anticipated
  questions. Guarded by a read-only validator, a retry loop, and full SQL transparency in
  the UI.
- **Leadership-update figures computed in pandas, never by the model** — the LLM writes
  prose around numbers it was handed. A hallucinated board-deck figure is unrecoverable.
- **300s TTL cache** — the brief bans hardcoded CSV data, not caching. Data age and a
  manual refresh are always visible in the sidebar.

Full rationale and trade-offs: **[DECISION_LOG.md](DECISION_LOG.md)**.

---

## monday.com setup

1. Create a monday.com account (free trial is enough).
2. **+ Add → Import data → Excel/CSV**, upload `monday_import/deals.csv`,
   name the board **`Deals`**.
3. Repeat with `monday_import/work_orders.csv`, name it **`Work Orders`**.
   Accept monday's auto-detected column types — the agent reads raw text and normalises it
   itself, so board-side typing does not matter.
4. **Avatar (bottom-left) → Developers → My Access Tokens → Show** and copy the token.

   *Or skip steps 2–3 entirely* and provision both boards from the CSVs in one command:

   ```bash
   python scripts/setup_boards.py --token <MONDAY_API_TOKEN>
   ```

   This is setup tooling, kept separate from the agent so the agent's own
   read-only guarantee stays unqualified. It creates every column as `text` on
   purpose: monday's importer coerces on ingest and silently discards values it
   cannot parse, which would quietly repair the very defects the agent is meant to
   handle. Importing as text is lossless; typing happens in `agent/normalize.py`,
   where coercion failures are counted and shown to the user.
5. Board IDs are optional. If omitted the agent resolves boards by name; to pin them, take
   the number from the board URL: `monday.com/boards/`**`1234567890`**.

## Configuration

| Key | Required | Purpose |
|---|---|---|
| `MONDAY_API_TOKEN` | yes | monday.com API v2 token |
| `GEMINI_API_KEY` | yes | https://aistudio.google.com/apikey (free tier is sufficient) |
| `DEALS_BOARD_ID` | no | Pin the deals board; otherwise resolved by name |
| `WORK_ORDERS_BOARD_ID` | no | Pin the work-order board |
| `CACHE_TTL_SECONDS` | no | Default `300` |
| `GEMINI_MODEL` | no | Default `gemini-2.5-flash` |

## Run locally

```bash
git clone <this repo> && cd skylark-bi-agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
#   ...fill in MONDAY_API_TOKEN and GEMINI_API_KEY

streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. https://share.streamlit.io → **New app** → pick the repo, main file `app.py`.
3. **Advanced settings → Secrets** → paste the contents of your `secrets.toml`.
4. Deploy. `.streamlit/secrets.toml` is gitignored and never leaves your machine.

## Tests

```bash
python tests/test_pipeline.py                    # offline, no credentials needed
MONDAY_API_TOKEN=... GEMINI_API_KEY=... \
  python scripts/smoke_test.py                   # live end-to-end benchmark
```

31 checks covering value coercion, header-row removal, unusable-column detection, unit
splitting, derived fiscal columns, and SQL execution — ending with a **ground-truth
reconciliation** that compares the agent's open-pipeline count and value against a direct
pandas sum over the raw CSV. They must match to the rupee.

`tests/mock_monday.py` reproduces monday's response shape offline (including the way monday
absorbs the first CSV column into the item `name` field) so the data layer is testable
without a token. It is a fixture only — the application always reads from the live API.

## Known limitations

- `COMPANY*` ↔ `WOCOMPANY*` is unresolved; cross-board analysis goes through the
  non-unique masked deal name and is caveated on every answer that uses it.
- Collection dates are not tracked on the board, so DSO and AR ageing are impossible.
- Probability weights (0.75 / 0.45 / 0.20) are our assumption, not Skylark's.
- Two LLM calls per question — fine for a demo, wrong for a team of ten asking all day.

## Project layout

```
app.py                     Streamlit UI: chat, leadership update, data browser
agent/config.py            Secrets and tuning (Streamlit secrets → env → default)
agent/monday_client.py     GraphQL v2 client, pagination, backoff, read-only guard
agent/normalize.py         Cleaning, type coercion, quality profiling
agent/warehouse.py         DuckDB load, runtime schema document, SQL safety guard
agent/llm.py               Gemini REST wrapper with model fallback
agent/agent.py             Plan → execute → narrate orchestration
agent/leadership.py        Deterministic metrics + narrative generation
scripts/setup_boards.py    One-off board provisioner (setup only, not the agent)
scripts/smoke_test.py      Live end-to-end benchmark across 6 founder questions
tests/                     Offline fixture and validation suite
monday_import/             CSVs to import into monday.com
```
