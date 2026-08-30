# Skylark BI Agent — Vercel & Supabase Port Handoff Notes

## Branch: codex/vercel-port

This document details the exact state, verification metrics, deployment requirements, and handoff instructions for deploying the Next.js 15 + FastAPI port of the Skylark BI Agent to Vercel with Supabase caching.

---

## 1. Verified Locally (With Exact Numbers)

All verification was conducted against live monday.com and Gemini APIs with SUPABASE_URL deliberately unset to test the fallback path, as well as offline test suites.

### a) Offline Test Suite
- Command: python tests/test_pipeline.py
- Result: **31 / 31 checks passed** (100% pass rate).
- Zero modifications were made to gent/*.py.

### b) Next.js / TypeScript Build
- Command: 
pm run build
- Result: **Compiled successfully** (exit code 0), 4/4 static pages generated without type or lint errors.

### c) API Endpoints (pi/index.py)
1. **GET /api/health**
   - Deals count: **344 rows** (2 embedded header rows dropped)
   - Work orders count: **176 rows**
   - Boards: Deals 5030962955, Work Orders 5030963215
   - Unusable columns: exactly 4 (['expected_billing_month', 'actual_collection_month', 'collection_status', 'collection_date'])
   - Cache status: ypass
   - Warning emitted: "Supabase warehouse cache is not configured; falling back to a live monday.com fetch for this request."
   - Status code: 200 OK

2. **POST /api/ask — "What's our total outstanding receivable and which sector is it concentrated in?"**
   - Total outstanding: **Rs 3.63 Cr** (36,291,749 across all sectors)
   - Concentration: **Renewables** (Rs 2.08 Cr, ~57.4% of total outstanding)
   - Result rows returned: **6 rows**
   - Status code: 200 OK

3. **POST /api/ask — "When did we last collect payment from each customer?"**
   - Action: **unsupported** (Correct behavior; never returns an empty table)
   - Answer: "The 'collection_date' field in the work_orders table is UNUSABLE (never populated), and there is no other field tracking payment collection dates."
   - Status code: 200 OK

4. **POST /api/leadership**
   - Open pipeline: **Rs 68.82 Cr across 49 deals** (47 carrying value)
   - Weighted value: **Rs 24.65 Cr**
   - Top deal: Sakura (COMPANY197, Tender, Rs 30.59 Cr)
   - Wall-clock duration: **13.69 seconds** (well within Vercel's 60s maxDuration and below the 45s threshold)
   - Configurable override: LEADERSHIP_MAX_TOKENS environment variable supported via LeadershipGemini in pi/index.py if needed.
   - Status code: 200 OK

5. **GET /api/data**
   - Deals: 344 rows, Work Orders: 176 rows.

---

## 2. Information Density & UI Alignment

The Next.js UI (pp/page.tsx) matches the full information density of pp.py:
- **Sidebar**: Connection status, board IDs, rows loaded, dropped header rows count, data age, cache status pill, refresh button, collapsible data quality diagnostics (unusable_columns, low_coverage, coercion_failures, and dataset notes), read-only disclaimer.
- **Ask Tab**: Benchmark sample question buttons, SQL transparency expander showing executed SQL + self-correction attempt history, structured data table with row counts, assumptions line, model badge, unsupported action indicator, and latency timer.
- **Leadership Tab**: Optional emphasis prompt, narrative markdown rendering, "Download as Markdown" button (skylark_leadership_update.md), and raw computed metrics JSON inspector.
- **Data Tab**: Toggle between normalized deals and work_orders tables with column schemas, row counts, and scrollable data grid.

---

## 3. Items That CANNOT Be Verified Without Vercel/Supabase

1. **Supabase Live PostgREST Round-Trip**:
   - Table warehouse_cache read/write cannot be tested until deployed with SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.
2. **Vercel Serverless Function Routing**:
   - ercel.json rewrites /api/(.*) to /api/index. In Vercel's Python runtime, the incoming path /api/... is passed to the ASGI scope, matching FastAPI's @app.get("/api/health"), etc. While verified in FastAPI locally, edge deployment behavior requires a live Vercel environment.
3. **Vercel Cron Execution**:
   - Cron "schedule": "0 6 * * *" targeting /api/health to prevent Supabase free project idling requires Vercel production deployment.

---

## 4. Environment Variables Required for Deployment

Set these in **Vercel Project Settings -> Environment Variables**:

| Variable | Description | Example / Value |
|---|---|---|
| MONDAY_API_TOKEN | monday.com API token (read-only) | RAW token, 229 chars, NO "Bearer" prefix â€” a Bearer prefix returns HTTP 401 |
| GEMINI_API_KEY | Google Gemini API key | AIza... |
| DEALS_BOARD_ID | Deals board ID | 5030962955 |
| WORK_ORDERS_BOARD_ID | Work Orders board ID | 5030963215 |
| GEMINI_MODEL | Primary model | gemini-3.1-flash-lite |
| SUPABASE_URL | Supabase project HTTPS URL | https://<project-ref>.supabase.co |
| SUPABASE_SERVICE_ROLE_KEY | Supabase service_role secret key | <service_role_key> |
| LEADERSHIP_MAX_TOKENS | *(Optional)* Narrative token budget | 2600 |

---

## 5. Deployment Instructions

### Step 1: Create Supabase Cache Table
In the Supabase Dashboard -> **SQL Editor**, run [supabase/schema.sql](supabase/schema.sql):
`sql
create table if not exists warehouse_cache (
  key text primary key,
  payload jsonb not null,
  fetched_at timestamptz not null default now()
);
`

### Step 2: Deploy to Vercel
From the repository root on branch codex/vercel-port:
`ash
# Pull env vars or configure them in dashboard
vercel --prod
`

### Step 3: Verify Deployment
1. Test Health:
   `ash
   curl -s https://<your-vercel-domain>/api/health
   `
   - First call: "cache": "miss" (fetches monday.com live in ~10-15s, writes to Supabase).
   - Second call (within 300s): "cache": "hit" (instant response < 500ms from Supabase).
2. Open https://<your-vercel-domain> in browser and test:
   - Ask benchmark questions.
   - Click "Generate leadership update".
   - View Data tab.

---

## 6. Technical Notes & Reasoning

- **Vercel Rewrites**: ercel.json configures "rewrites": [{ "source": "/api/(.*)", "destination": "/api/index" }]. Because pi/index.py routes are declared with explicit /api/... paths (e.g. @app.get("/api/health")), FastAPI matches the full incoming path routed by Vercel.
- **NaN JSON Sanitation**: In pi/index.py, _clean_nan() recursively transforms loat('nan') / 
umpy.nan to None so that compute_metrics results serialize to strict RFC 8259 JSON 
ull without throwing ValueError in FastAPI's response serializer.
- **Safety & Integrity**: Zero files in gent/ and pp.py were modified. The live Streamlit app on main remains untouched.