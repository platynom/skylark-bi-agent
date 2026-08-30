"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";

const QUESTIONS = [
  "How's our pipeline looking for the renewables sector?",
  "What's our total outstanding receivable, and which sector is it concentrated in?",
  "Which work orders are completed but haven't been invoiced yet?",
  "What's our win rate, and does it differ by sector?",
  "Which deal owner has the largest open pipeline?",
  "How much revenue have we contracted in mining vs renewables?",
  "Which quarter does our open pipeline actually close in?"
];

type Tab = "ask" | "leadership" | "data";
type Row = Record<string, unknown>;
type TableQuality = {
  board: string; board_id: string; rows: number; dropped_header_rows: number;
  unusable_columns: string[]; low_coverage: Record<string, number>;
  coercion_failures?: Record<string, number>; notes?: string[];
};
type Health = {
  ok: boolean; boards: Record<string, string>; row_counts: Record<string, number>;
  quality: { tables: Record<string, TableQuality> }; cache: string;
  data_age_seconds: number; warning?: string | null;
};
type AskResult = {
  action: string; intent?: string | null; sql?: string | null; answer?: string | null;
  assumptions: string[]; rows: Row[]; rowcount: number;
  attempts: { sql: string; error: string }[]; clarify?: string | null;
  options?: string[]; error?: string | null; model?: string | null;
  cache: string; latency_ms: number; warning?: string | null;
};
type Message = { role: "user" | "assistant"; content: string; result?: AskResult };
type DataResponse = { table: string; rows: Row[]; rowcount: number; columns: string[]; cache: string };

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    cache: "no-store"
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body?.error?.message || `Request failed (${response.status})`);
  return body as T;
}

function StatusPill({ value }: { value: string }) {
  const good = value === "hit";
  return <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest ${good ? "bg-mint text-forest" : "bg-amber/15 text-amber"}`}>{value}</span>;
}

function DataTable({ rows, columns }: { rows: Row[]; columns?: string[] }) {
  const keys = columns || (rows[0] ? Object.keys(rows[0]) : []);
  if (!rows.length) return <p className="py-5 text-sm text-slate-500">No rows returned.</p>;
  return (
    <div className="max-h-[32rem] overflow-auto rounded-xl border border-forest/10 bg-white">
      <table className="min-w-full whitespace-nowrap text-left text-xs">
        <thead className="sticky top-0 z-10 bg-forest text-white">
          <tr>{keys.map((key) => <th key={key} className="px-3 py-2.5 font-semibold">{key}</th>)}</tr>
        </thead>
        <tbody>{rows.map((row, index) => (
          <tr key={index} className="border-b border-forest/5 odd:bg-paper/60 hover:bg-mint/40">
            {keys.map((key) => <td key={key} className="max-w-72 overflow-hidden text-ellipsis px-3 py-2">{row[key] == null ? <span className="text-slate-300">NULL</span> : String(row[key])}</td>)}
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

function Sidebar({ health, loading, refresh }: { health: Health | null; loading: boolean; refresh: () => void }) {
  const dropped = health ? Object.values(health.quality.tables).reduce((n, table) => n + table.dropped_header_rows, 0) : 0;
  return (
    <aside className="border-b border-forest/10 bg-forest px-5 py-6 text-white lg:fixed lg:inset-y-0 lg:w-[19rem] lg:overflow-y-auto lg:border-b-0 lg:border-r">
      <div className="mb-7 flex items-center gap-3">
        <div className="grid h-10 w-10 place-items-center rounded-xl bg-mint font-black text-forest">S</div>
        <div><p className="font-bold tracking-tight">Skylark BI</p><p className="text-xs text-white/60">Founder intelligence console</p></div>
      </div>
      <section className="mb-7">
        <p className="mb-3 text-[10px] font-bold uppercase tracking-[.22em] text-moss">Live connection</p>
        {health ? Object.entries(health.quality.tables).map(([name, table]) => (
          <div key={name} className="mb-3 rounded-xl border border-white/10 bg-white/[.06] p-3">
            <div className="flex items-center justify-between"><strong className="text-sm">{table.board}</strong><span className="text-xs text-mint">{table.rows} rows</span></div>
            <code className="mt-1 block text-[10px] text-white/45">{table.board_id}</code>
          </div>
        )) : <p className="text-sm text-white/50">{loading ? "Reading live boards…" : "Connection unavailable"}</p>}
        {health && <div className="mt-3 flex items-center justify-between text-xs text-white/60"><span>{Math.round(health.data_age_seconds)}s old</span><StatusPill value={health.cache} /></div>}
        <button onClick={refresh} disabled={loading} className="mt-4 w-full rounded-xl bg-mint px-3 py-2.5 text-sm font-bold text-forest transition hover:bg-white disabled:opacity-50">{loading ? "Refreshing…" : "Refresh from monday.com"}</button>
      </section>
      {health && <section>
        <p className="mb-3 text-[10px] font-bold uppercase tracking-[.22em] text-moss">Data quality</p>
        <div className="mb-3 rounded-xl border border-white/10 p-3 text-xs"><strong>{dropped} header rows dropped</strong><p className="mt-1 text-white/55">Removed before analysis.</p></div>
        {Object.entries(health.quality.tables).map(([name, table]) => <details key={name} className="mb-2 rounded-xl border border-white/10 bg-white/[.04] p-3 text-xs">
          <summary className="cursor-pointer font-semibold capitalize">{name.replace("_", " ")}</summary>
          <p className="mt-3 font-semibold text-moss">Never populated</p>
          <p className="mt-1 break-words text-white/60">{table.unusable_columns.join(", ") || "None"}</p>
          <p className="mt-3 font-semibold text-moss">Partial coverage</p>
          <div className="mt-1 space-y-1 text-white/60">{Object.entries(table.low_coverage).sort((a, b) => a[1] - b[1]).slice(0, 12).map(([key, value]) => <div key={key} className="flex justify-between gap-2"><span className="truncate">{key}</span><span>{Math.round(value * 100)}%</span></div>)}</div>
          {table.coercion_failures && Object.keys(table.coercion_failures).length > 0 && <>
            <p className="mt-3 font-semibold text-moss">Failed type coercion</p>
            <div className="mt-1 space-y-1 text-white/60">{Object.entries(table.coercion_failures).map(([key, count]) => <div key={key} className="flex justify-between gap-2"><span className="truncate">{key}</span><span>{count}</span></div>)}</div>
          </>}
          {table.notes && table.notes.length > 0 && <>
            <p className="mt-3 font-semibold text-moss">Data notes</p>
            <div className="mt-1 space-y-1.5 text-white/60">{table.notes.map((note, idx) => <p key={idx} className="leading-snug">{note}</p>)}</div>
          </>}
        </details>)}
      </section>}
      <p className="mt-8 border-t border-white/10 pt-4 text-[11px] leading-relaxed text-white/40">Read-only. Every answer is generated from normalized monday.com data and exposes its SQL.</p>
    </aside>
  );
}

export default function Home() {
  const [tab, setTab] = useState<Tab>("ask");
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState("");
  const [loadingHealth, setLoadingHealth] = useState(true);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [focus, setFocus] = useState("");
  const [leadership, setLeadership] = useState<{ narrative: string; metrics: Row; model?: string; cache: string; latency_ms: number } | null>(null);
  const [leadershipLoading, setLeadershipLoading] = useState(false);
  const [dataTable, setDataTable] = useState<"deals" | "work_orders">("deals");
  const [data, setData] = useState<DataResponse | null>(null);
  const [dataLoading, setDataLoading] = useState(false);
  const [error, setError] = useState("");

  const loadHealth = useCallback(async (force = false) => {
    setLoadingHealth(true); setHealthError("");
    try { setHealth(await api<Health>(`/api/health${force ? "?force=true" : ""}`)); }
    catch (e) { setHealthError(e instanceof Error ? e.message : "Health check failed"); }
    finally { setLoadingHealth(false); }
  }, []);
  useEffect(() => { void loadHealth(); }, [loadHealth]);

  const sendQuestion = async (text: string) => {
    const clean = text.trim(); if (!clean || asking) return;
    const prior = messages.map(({ role, content }) => ({ role, content }));
    setMessages((old) => [...old, { role: "user", content: clean }]);
    setQuestion(""); setAsking(true); setError("");
    try {
      const result = await api<AskResult>("/api/ask", { method: "POST", body: JSON.stringify({ question: clean, history: prior }) });
      const content = result.answer || result.clarify || result.error || "No answer was produced.";
      setMessages((old) => [...old, { role: "assistant", content, result }]);
    } catch (e) { setError(e instanceof Error ? e.message : "Question failed"); }
    finally { setAsking(false); }
  };

  const generateLeadership = async () => {
    setLeadershipLoading(true); setError("");
    try { setLeadership(await api("/api/leadership", { method: "POST", body: JSON.stringify({ focus: focus.trim() || null }) })); }
    catch (e) { setError(e instanceof Error ? e.message : "Leadership update failed"); }
    finally { setLeadershipLoading(false); }
  };

  const loadData = useCallback(async (table: "deals" | "work_orders") => {
    setDataLoading(true); setError("");
    try { setData(await api<DataResponse>(`/api/data?table=${table}&limit=500`)); }
    catch (e) { setError(e instanceof Error ? e.message : "Data load failed"); }
    finally { setDataLoading(false); }
  }, []);
  useEffect(() => { if (tab === "data") void loadData(dataTable); }, [tab, dataTable, loadData]);

  const tabs = useMemo(() => ([
    ["ask", "Ask"], ["leadership", "Leadership update"], ["data", "Data"]
  ] as [Tab, string][]), []);

  return (
    <div className="min-h-screen">
      <Sidebar health={health} loading={loadingHealth} refresh={() => void loadHealth(true)} />
      <main className="px-5 py-8 sm:px-8 lg:ml-[19rem] lg:px-12 lg:py-10">
        <header className="mx-auto mb-8 max-w-6xl">
          <p className="mb-2 text-xs font-bold uppercase tracking-[.22em] text-moss">Live operations intelligence</p>
          <h1 className="max-w-3xl text-3xl font-black tracking-[-.04em] text-forest sm:text-5xl">Ask the business. Inspect the evidence.</h1>
          <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-600 sm:text-base">Founder-level answers over live monday.com boards, with every SQL query and data-quality limitation visible.</p>
        </header>
        <div className="mx-auto max-w-6xl">
          <nav className="mb-6 flex gap-1 overflow-x-auto rounded-2xl border border-forest/10 bg-white p-1.5 shadow-sm">
            {tabs.map(([key, label]) => <button key={key} onClick={() => setTab(key)} className={`whitespace-nowrap rounded-xl px-4 py-2.5 text-sm font-semibold transition ${tab === key ? "bg-forest text-white" : "text-slate-500 hover:bg-mint/40 hover:text-forest"}`}>{label}</button>)}
          </nav>
          {(error || healthError) && <div className="mb-5 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">{error || healthError}</div>}

          {tab === "ask" && <section>
            {!messages.length && <div className="mb-6 rounded-2xl border border-forest/10 bg-white p-5 shadow-panel sm:p-7">
              <p className="mb-4 text-sm font-bold text-forest">Start with a benchmark question</p>
              <div className="grid gap-2 md:grid-cols-2">{QUESTIONS.map((item) => <button key={item} onClick={() => void sendQuestion(item)} className="rounded-xl border border-forest/10 bg-paper px-4 py-3 text-left text-sm leading-snug transition hover:-translate-y-0.5 hover:border-moss hover:bg-mint/40">{item}</button>)}</div>
            </div>}
            <div className="space-y-4">{messages.map((message, index) => <article key={index} className={`rounded-2xl p-5 sm:p-6 ${message.role === "user" ? "ml-auto max-w-3xl bg-forest text-white" : "max-w-4xl border border-forest/10 bg-white shadow-panel"}`}>
              {message.role === "assistant" ? <div className="prose-skylark text-sm"><ReactMarkdown>{message.content}</ReactMarkdown></div> : <p>{message.content}</p>}
              {message.result && <div className="mt-4 border-t border-forest/10 pt-4">
                <div className="mb-3 flex flex-wrap items-center gap-2 text-[11px] text-slate-500"><StatusPill value={message.result.cache} /><span>{(message.result.latency_ms / 1000).toFixed(1)}s</span>{message.result.model && <span>{message.result.model}</span>}{message.result.action === "unsupported" && <span className="font-bold uppercase tracking-wider text-amber">unsupported</span>}</div>
                {!!message.result.assumptions?.length && <p className="mb-3 text-xs text-slate-500"><strong>Assumptions:</strong> {message.result.assumptions.join("; ")}</p>}
                {message.result.sql && <details className="mb-2 rounded-xl border border-forest/10 bg-paper p-3"><summary className="cursor-pointer text-xs font-bold text-forest">SQL executed</summary><pre className="mt-3 overflow-x-auto whitespace-pre-wrap text-xs leading-relaxed text-slate-700">{message.result.sql}</pre>{!!message.result.attempts?.length && <p className="mt-2 text-xs text-amber">Self-corrected {message.result.attempts.length} earlier attempt(s).</p>}</details>}
                {message.result.rows?.length > 0 && <details className="rounded-xl border border-forest/10 p-3"><summary className="cursor-pointer text-xs font-bold text-forest">Result data ({message.result.rowcount} rows)</summary><div className="mt-3"><DataTable rows={message.result.rows} /></div></details>}
              </div>}
            </article>)}</div>
            {asking && <div className="my-4 flex items-center gap-2 text-sm text-slate-500"><span className="h-2 w-2 animate-pulse rounded-full bg-moss" /> Planning, querying and checking the result…</div>}
            <form onSubmit={(event: FormEvent) => { event.preventDefault(); void sendQuestion(question); }} className="sticky bottom-4 mt-6 flex gap-2 rounded-2xl border border-forest/10 bg-white/95 p-2 shadow-panel backdrop-blur">
              <input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about pipeline, revenue, sectors, collections…" className="min-w-0 flex-1 rounded-xl px-4 py-3 text-sm outline-none placeholder:text-slate-400 focus:bg-paper" />
              <button disabled={asking || !question.trim()} className="rounded-xl bg-forest px-5 py-3 text-sm font-bold text-white hover:bg-ink disabled:opacity-40">Ask</button>
            </form>
          </section>}

          {tab === "leadership" && <section className="rounded-2xl border border-forest/10 bg-white p-5 shadow-panel sm:p-7">
            <h2 className="text-xl font-black text-forest">Leadership update</h2><p className="mt-1 text-sm text-slate-500">Numbers are computed deterministically; Gemini writes only the narrative.</p>
            <div className="mt-5 flex flex-col gap-2 sm:flex-row"><input value={focus} onChange={(e) => setFocus(e.target.value)} placeholder="Optional emphasis, e.g. cash exposure" className="flex-1 rounded-xl border border-forest/15 bg-paper px-4 py-3 text-sm outline-none focus:border-moss" /><button onClick={() => void generateLeadership()} disabled={leadershipLoading} className="rounded-xl bg-forest px-5 py-3 text-sm font-bold text-white disabled:opacity-50">{leadershipLoading ? "Generating…" : "Generate update"}</button></div>
            {leadership && <div className="mt-7 border-t border-forest/10 pt-6">
              <div className="prose-skylark"><ReactMarkdown>{leadership.narrative}</ReactMarkdown></div>
              <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-forest/10 pt-4 text-xs text-slate-500">
                <div className="flex items-center gap-2">
                  <StatusPill value={leadership.cache} />
                  <span>{(leadership.latency_ms / 1000).toFixed(1)}s</span>
                  {leadership.model && <span>{leadership.model}</span>}
                </div>
                <button
                  onClick={() => {
                    const blob = new Blob([leadership.narrative], { type: "text/markdown" });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = "skylark_leadership_update.md";
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  className="rounded-xl border border-forest/20 bg-paper px-3 py-1.5 font-bold text-forest hover:bg-mint/50"
                >
                  Download as Markdown
                </button>
              </div>
              <details className="mt-4 rounded-xl border border-forest/10 bg-paper p-3"><summary className="cursor-pointer text-xs font-bold text-forest">Raw computed metrics</summary><pre className="mt-3 max-h-[32rem] overflow-auto text-xs">{JSON.stringify(leadership.metrics, null, 2)}</pre></details>
            </div>}
          </section>}

          {tab === "data" && <section className="rounded-2xl border border-forest/10 bg-white p-5 shadow-panel sm:p-7">
            <div className="mb-5 flex flex-wrap items-end justify-between gap-3"><div><h2 className="text-xl font-black text-forest">Normalized data</h2><p className="mt-1 text-sm text-slate-500">The exact clean tables queried by the agent.</p></div><div className="flex rounded-xl bg-paper p-1">{(["deals", "work_orders"] as const).map((name) => <button key={name} onClick={() => setDataTable(name)} className={`rounded-lg px-3 py-2 text-xs font-bold capitalize ${dataTable === name ? "bg-white text-forest shadow-sm" : "text-slate-500"}`}>{name.replace("_", " ")}</button>)}</div></div>
            {dataLoading ? <p className="py-10 text-center text-sm text-slate-500">Loading normalized rows…</p> : data && <><div className="mb-3 flex items-center gap-2 text-xs text-slate-500"><strong>{data.rowcount} rows</strong><StatusPill value={data.cache} /></div><DataTable rows={data.rows} columns={data.columns} /></>}
          </section>}
        </div>
      </main>
    </div>
  );
}
