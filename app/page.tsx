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
type ProviderOption = "auto" | "vertex" | "ai_studio" | "deterministic";

type TableQuality = {
  board: string; board_id: string; rows: number; dropped_header_rows: number;
  unusable_columns: string[]; low_coverage: Record<string, number>;
  coercion_failures?: Record<string, number>; notes?: string[];
};

type ProviderStatusItem = {
  status: "healthy" | "throttled" | "standby";
  consecutive_failures: number;
  throttled_remaining_sec: number;
  last_error?: string | null;
  total_calls: number;
  total_failures: number;
};

type Health = {
  ok: boolean; boards: Record<string, string>; row_counts: Record<string, number>;
  quality: { tables: Record<string, TableQuality> }; cache: string;
  data_age_seconds: number; warning?: string | null;
  providers?: Record<string, ProviderStatusItem>;
};

type AskResult = {
  action: string; intent?: string | null; sql?: string | null; answer?: string | null;
  assumptions: string[]; rows: Row[]; rowcount: number; columns?: string[];
  caveats?: string[]; attempts: { sql: string; error: string }[]; clarify?: string | null;
  options?: string[]; error?: string | null; provider?: string; model?: string | null;
  provider_chain_attempted?: string[]; providers_status?: Record<string, ProviderStatusItem>;
  cache: string; latency_ms: number; warning?: string | null;
};

type Message = { role: "user" | "assistant"; content: string; result?: AskResult; narrating?: boolean };
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
  const isHit = value === "hit";
  const isQCache = value === "question_cache";
  return (
    <span
      className={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest ${
        isHit || isQCache ? "bg-mint text-forest" : "bg-amber/15 text-amber"
      }`}
    >
      {isQCache ? "cached" : value}
    </span>
  );
}

function LiveConnectionPill({ value }: { value: string }) {
  const isLiveFetch = value === "miss";
  const label = value === "hit" ? "CACHED" : isLiveFetch ? "LIVE FETCH" : value;

  return (
    <span
      title="LIVE FETCH = pulled fresh from monday.com. CACHED = served from the warehouse snapshot."
      className={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest ${
        isLiveFetch ? "bg-mint/20 text-mint" : "bg-white/10 text-white/60"
      }`}
    >
      {label}
    </span>
  );
}

function ProviderBadge({ result }: { result: AskResult }) {
  const provider = result.provider || "vertex";
  const latency = (result.latency_ms / 1000).toFixed(1);
  const chain = result.provider_chain_attempted || [];
  const hadFailover = chain.some(c => c.includes("429") || c.includes("error") || (c.includes("vertex") && provider === "ai_studio"));

  let label = "Vertex";
  let badgeColor = "bg-blue-50 text-blue-800 border-blue-200";

  if (provider === "ai_studio") {
    label = "AI Studio";
    badgeColor = hadFailover ? "bg-amber-50 text-amber-900 border-amber-300" : "bg-purple-50 text-purple-800 border-purple-200";
  } else if (provider === "deterministic") {
    label = "Deterministic";
    badgeColor = "bg-emerald-50 text-emerald-800 border-emerald-200";
  }

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-xs font-semibold ${badgeColor}`}>
        <span className="h-1.5 w-1.5 rounded-full bg-current opacity-75" />
        <span>{label} &bull; {latency}s</span>
        {hadFailover && <span className="font-normal text-[11px] opacity-90">(Vertex unavailable &rarr; auto failed over)</span>}
      </span>
      {result.model && result.provider !== "deterministic" && (
        <span className="font-mono text-[10px] text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200">
          {result.model}
        </span>
      )}
    </div>
  );
}

function DataTable({ rows, columns }: { rows: Row[]; columns?: string[] }) {
  const keys = columns || (rows[0] ? Object.keys(rows[0]) : []);
  if (!rows.length) return <p className="py-5 text-sm text-slate-500">No rows returned.</p>;
  return (
    <div className="max-h-[32rem] w-full overflow-auto rounded-xl border border-forest/10 bg-white shadow-inner">
      <table className="min-w-full w-full whitespace-nowrap text-left text-xs">
        <thead className="sticky top-0 z-10 bg-forest text-white">
          <tr>{keys.map((key) => <th key={key} className="px-4 py-3 font-semibold tracking-wider">{key === "monday_item_url" ? "source" : key}</th>)}</tr>
        </thead>
        <tbody className="divide-y divide-forest/5">{rows.map((row, index) => (
          <tr key={index} className="odd:bg-paper/60 hover:bg-mint/40 transition-colors">
            {keys.map((key) => <td key={key} className="max-w-md overflow-hidden text-ellipsis px-4 py-2.5">{
              row[key] == null
                ? <span className="font-mono text-slate-300">NULL</span>
                : key === "monday_item_url"
                  ? <a href={String(row[key])} target="_blank" rel="noreferrer" className="font-semibold text-forest underline decoration-moss underline-offset-2 hover:text-moss">Open item ↗</a>
                  : String(row[key])
            }</td>)}
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

function Sidebar({ health, loading, refresh }: { health: Health | null; loading: boolean; refresh: () => void }) {
  const dropped = health ? Object.values(health.quality.tables).reduce((n, table) => n + table.dropped_header_rows, 0) : 0;
  const providers = health?.providers;

  return (
    <aside className="border-b border-forest/10 bg-forest px-5 py-6 text-white lg:fixed lg:inset-y-0 lg:w-[19rem] lg:overflow-y-auto lg:border-b-0 lg:border-r">
      <div className="mb-7 flex items-center gap-3">
        <div className="grid h-10 w-10 place-items-center rounded-xl bg-mint font-black text-forest">S</div>
        <div><p className="font-bold tracking-tight">Skylark BI</p><p className="text-xs text-white/60">Founder intelligence console</p></div>
      </div>

      <section className="mb-7">
        <p className="mb-3 text-[10px] font-bold uppercase tracking-[.22em] text-moss">LLM Router & Providers</p>
        <div className="space-y-2 rounded-xl border border-white/10 bg-white/[.06] p-3 text-xs">
          <div className="flex items-center justify-between">
            <span className="font-semibold">Vertex AI (Primary)</span>
            <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase ${
              providers?.vertex?.status === "throttled"
                ? "bg-amber/20 text-amber"
                : providers?.vertex?.status === "standby"
                  ? "bg-white/10 text-white/60"
                  : "bg-mint/20 text-mint"
            }`}>
              {providers?.vertex?.status === "throttled"
                ? `throttled (${providers.vertex.throttled_remaining_sec}s)`
                : providers?.vertex?.status || "healthy"}
            </span>
          </div>
          <p className="text-[11px] text-white/50">asia-southeast1 &bull; gemini-2.5-flash</p>

          <div className="mt-2.5 flex items-center justify-between border-t border-white/10 pt-2">
            <span className="font-semibold">AI Studio (Secondary)</span>
            <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase ${
              providers?.ai_studio?.status === "throttled"
                ? "bg-amber/20 text-amber"
                : "bg-mint/20 text-mint"
            }`}>
              {providers?.ai_studio?.status === "throttled"
                ? `throttled (${providers.ai_studio.throttled_remaining_sec}s)`
                : providers?.ai_studio?.status || "healthy"}
            </span>
          </div>
          <p className="text-[11px] text-white/50">Independent free tier pool</p>

          <div className="mt-2.5 flex items-center justify-between border-t border-white/10 pt-2">
            <span className="font-semibold">Deterministic Fallback</span>
            <span className="rounded-full bg-mint/20 px-2 py-0.5 text-[10px] font-bold uppercase text-mint">Active</span>
          </div>
          <p className="text-[11px] text-white/50">Rule engine floor &bull; 0ms</p>
        </div>
      </section>

      <section className="mb-7">
        <p className="mb-3 text-[10px] font-bold uppercase tracking-[.22em] text-moss">Live connection</p>
        {health ? Object.entries(health.quality.tables).map(([name, table]) => (
          <div key={name} className="mb-3 rounded-xl border border-white/10 bg-white/[.06] p-3">
            <div className="flex items-center justify-between"><strong className="text-sm">{table.board}</strong><span className="text-xs text-mint">{table.rows} rows</span></div>
            <code className="mt-1 block text-[10px] text-white/45">{table.board_id}</code>
          </div>
        )) : <p className="text-sm text-white/50">{loading ? "Reading live boards…" : "Connection unavailable"}</p>}
        {health && <div className="mt-3 flex items-center justify-between text-xs text-white/60"><span>{Math.round(health.data_age_seconds)}s old</span><LiveConnectionPill value={health.cache} /></div>}
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
  const [selectedProvider, setSelectedProvider] = useState<ProviderOption>("auto");
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
    const clean = text.trim();
    if (!clean || asking) return;
    const prior = messages.map(({ role, content }) => ({ role, content }));
    setMessages((old) => [...old, { role: "user", content: clean }]);
    setQuestion("");
    setAsking(true);
    setError("");
    try {
      const providerParam = selectedProvider === "auto" ? null : selectedProvider;
      const result = await api<AskResult>("/api/ask", {
        method: "POST",
        body: JSON.stringify({ question: clean, history: prior, provider: providerParam }),
      });
      const initialContent = result.answer || result.clarify || result.error || "";
      const needsNarration = !initialContent && result.action === "sql";

      setMessages((old) => [
        ...old,
        {
          role: "assistant",
          content: initialContent,
          result,
          narrating: needsNarration,
        },
      ]);

      if (needsNarration) {
        try {
          const narration = await api<{ answer: string; provider?: string; model?: string; latency_ms: number; provider_chain_attempted?: string[] }>("/api/narrate", {
            method: "POST",
            body: JSON.stringify({
              question: clean,
              intent: result.intent,
              sql: result.sql,
              assumptions: result.assumptions,
              rows: result.rows,
              caveats: result.caveats || [],
              history: prior,
              provider: providerParam,
            }),
          });
          setMessages((old) =>
            old.map((m, i) =>
              i === old.length - 1
                ? {
                    ...m,
                    content: narration.answer || "No narrative produced.",
                    narrating: false,
                    result: m.result ? {
                      ...m.result,
                      answer: narration.answer,
                      provider: narration.provider || m.result.provider,
                      model: narration.model || m.result.model,
                      latency_ms: m.result.latency_ms + (narration.latency_ms || 0),
                      provider_chain_attempted: narration.provider_chain_attempted || m.result.provider_chain_attempted,
                    } : m.result,
                  }
                : m
            )
          );
        } catch {
          setMessages((old) =>
            old.map((m, i) =>
              i === old.length - 1
                ? {
                    ...m,
                    content: "SQL executed and verified.",
                    narrating: false,
                  }
                : m
            )
          );
        }
      }
      void loadHealth();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Question failed");
    } finally {
      setAsking(false);
    }
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
    <div className="min-h-screen flex flex-col lg:flex-row">
      <Sidebar health={health} loading={loadingHealth} refresh={() => void loadHealth(true)} />
      <main className="flex-1 w-full px-5 py-8 sm:px-8 lg:ml-[19rem] lg:px-10 lg:py-10">
        <div className="mx-auto w-full max-w-[1140px]">
          <header className="mb-8 w-full">
            <p className="mb-2 text-xs font-bold uppercase tracking-[.22em] text-moss">Live operations intelligence</p>
            <h1 className="text-3xl font-black tracking-[-.04em] text-forest sm:text-4xl lg:text-5xl">Ask the business. Inspect the evidence.</h1>
            <p className="mt-3 max-w-3xl text-sm leading-relaxed text-slate-600 sm:text-base">Founder-level answers over live monday.com boards, with every SQL query and data-quality limitation visible.</p>
          </header>

          <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between w-full">
            <nav className="flex gap-1 overflow-x-auto rounded-2xl border border-forest/10 bg-white p-1.5 shadow-sm">
              {tabs.map(([key, label]) => <button key={key} onClick={() => setTab(key)} className={`whitespace-nowrap rounded-xl px-5 py-2.5 text-sm font-semibold transition ${tab === key ? "bg-forest text-white shadow-sm" : "text-slate-500 hover:bg-mint/40 hover:text-forest"}`}>{label}</button>)}
            </nav>

            {tab === "ask" && (
              <div className="flex items-center gap-2 rounded-2xl border border-forest/10 bg-white px-3 py-2 shadow-sm">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Router:</span>
                <div className="flex gap-1">
                  {(["auto", "vertex", "ai_studio", "deterministic"] as ProviderOption[]).map((p) => (
                    <button
                      key={p}
                      onClick={() => setSelectedProvider(p)}
                      className={`rounded-lg px-2.5 py-1 text-xs font-semibold capitalize transition ${
                        selectedProvider === p
                          ? "bg-forest text-white shadow-xs"
                          : "text-slate-600 hover:bg-paper hover:text-forest"
                      }`}
                    >
                      {p === "ai_studio" ? "AI Studio" : p}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {(error || healthError) && <div className="mb-5 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">{error || healthError}</div>}

          {tab === "ask" && <section className="w-full">
            {!messages.length && <div className="mb-6 w-full rounded-2xl border border-forest/10 bg-white p-6 shadow-panel sm:p-8">
              <div className="mb-4 flex items-center justify-between">
                <p className="text-sm font-bold text-forest">Start with a benchmark question</p>
                <span className="text-xs text-slate-400">Click to run against live data</span>
              </div>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 w-full">
                {QUESTIONS.map((item) => <button key={item} onClick={() => void sendQuestion(item)} className="rounded-xl border border-forest/10 bg-paper p-4 text-left text-sm font-medium leading-snug text-forest transition hover:-translate-y-0.5 hover:border-moss hover:bg-mint/40 hover:shadow-sm">{item}</button>)}
              </div>
            </div>}

            <div className="space-y-5 w-full">{messages.map((message, index) => <article key={index} className={`w-full rounded-2xl p-5 sm:p-7 shadow-panel transition ${message.role === "user" ? "ml-auto max-w-3xl border border-forest/20 bg-forest text-white" : "border border-forest/10 bg-white"}`}>
              {message.role === "assistant" ? (
                <div>
                  {message.narrating ? (
                    <div className="flex items-center gap-2.5 py-2 text-sm text-slate-500">
                      <span className="h-2 w-2 animate-ping rounded-full bg-moss" />
                      <span className="font-semibold text-forest">Synthesizing executive summary…</span>
                    </div>
                  ) : message.content ? (
                    <div className="prose-skylark text-sm w-full max-w-none">
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    </div>
                  ) : null}
                </div>
              ) : (
                <div className="flex items-start gap-2.5">
                  <span className="rounded-md bg-white/20 px-2 py-0.5 text-xs font-bold text-mint">You</span>
                  <p className="text-sm sm:text-base font-medium">{message.content}</p>
                </div>
              )}

              {message.result && <div className="mt-5 border-t border-forest/10 pt-5 space-y-4 w-full">
                <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                  <StatusPill value={message.result.cache} />
                  <ProviderBadge result={message.result} />
                  {message.result.action === "unsupported" && <span className="rounded bg-amber/15 px-2 py-0.5 font-bold uppercase tracking-wider text-amber">unsupported</span>}
                </div>
                {!!message.result.assumptions?.length && <div className="rounded-xl border border-forest/10 bg-paper/80 p-3 text-xs text-slate-600"><strong className="text-forest">Assumptions:</strong> {message.result.assumptions.join("; ")}</div>}
                {message.result.sql && <details className="w-full rounded-xl border border-forest/10 bg-paper p-4"><summary className="cursor-pointer text-xs font-bold text-forest hover:text-moss">SQL executed</summary><pre className="mt-3 overflow-x-auto whitespace-pre-wrap font-mono text-xs leading-relaxed text-slate-800 bg-white/80 p-3 rounded-lg border border-forest/5">{message.result.sql}</pre>{!!message.result.attempts?.length && <p className="mt-2 text-xs font-semibold text-amber">Self-corrected {message.result.attempts.length} earlier attempt(s).</p>}</details>}
                {message.result.rows?.length > 0 && <details className="w-full rounded-xl border border-forest/10 p-4 bg-paper/20" open={true}><summary className="cursor-pointer text-xs font-bold text-forest hover:text-moss">Result data ({message.result.rowcount} rows)</summary><div className="mt-3 w-full"><DataTable rows={message.result.rows} /></div></details>}
              </div>}
            </article>)}</div>

            {asking && <div className="my-5 flex items-center gap-2 text-sm text-slate-500"><span className="h-2.5 w-2.5 animate-pulse rounded-full bg-moss" /> Planning, querying and checking the result…</div>}

            <form onSubmit={(event: FormEvent) => { event.preventDefault(); void sendQuestion(question); }} className="sticky bottom-4 mt-8 flex gap-2 rounded-2xl border border-forest/10 bg-white/95 p-2.5 shadow-panel backdrop-blur w-full">
              <input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about pipeline, revenue, sectors, collections…" className="min-w-0 flex-1 rounded-xl px-4 py-3 text-sm outline-none placeholder:text-slate-400 focus:bg-paper" />
              <button disabled={asking || !question.trim()} className="rounded-xl bg-forest px-6 py-3 text-sm font-bold text-white transition hover:bg-ink disabled:opacity-40 shadow-sm">Ask</button>
            </form>
          </section>}

          {tab === "leadership" && <section className="w-full rounded-2xl border border-forest/10 bg-white p-6 shadow-panel sm:p-8">
            <div className="mb-4"><h2 className="text-xl font-black text-forest">Leadership update</h2><p className="mt-1 text-sm text-slate-500">Numbers are computed deterministically; Gemini writes only the narrative.</p></div>
            <div className="mt-5 flex flex-col gap-3 sm:flex-row w-full"><input value={focus} onChange={(e) => setFocus(e.target.value)} placeholder="Optional emphasis (e.g. 'board is worried about collections')" className="flex-1 rounded-xl border border-forest/15 bg-paper px-4 py-3 text-sm outline-none focus:border-moss" /><button onClick={() => void generateLeadership()} disabled={leadershipLoading} className="rounded-xl bg-forest px-6 py-3 text-sm font-bold text-white transition hover:bg-ink disabled:opacity-50 shadow-sm">{leadershipLoading ? "Generating…" : "Generate update"}</button></div>
            {leadership && <div className="mt-8 border-t border-forest/10 pt-6 w-full space-y-4">
              <div className="prose-skylark w-full max-w-none"><ReactMarkdown>{leadership.narrative}</ReactMarkdown></div>
              <div className="flex flex-wrap items-center justify-between gap-3 border-t border-forest/10 pt-4 text-xs text-slate-500">
                <div className="flex items-center gap-2">
                  <StatusPill value={leadership.cache} />
                  <span>{(leadership.latency_ms / 1000).toFixed(1)}s</span>
                  {leadership.model && <span className="font-mono text-[11px]">{leadership.model}</span>}
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
                  className="rounded-xl border border-forest/20 bg-paper px-4 py-2 font-bold text-forest hover:bg-mint/50 transition shadow-sm"
                >
                  Download as Markdown
                </button>
              </div>
              <details className="rounded-xl border border-forest/10 bg-paper p-4 w-full"><summary className="cursor-pointer text-xs font-bold text-forest hover:text-moss">Raw computed metrics</summary><pre className="mt-3 max-h-[32rem] overflow-auto rounded-lg bg-white p-3 font-mono text-xs text-slate-700 border border-forest/5">{JSON.stringify(leadership.metrics, null, 2)}</pre></details>
            </div>}
          </section>}

          {tab === "data" && <section className="w-full rounded-2xl border border-forest/10 bg-white p-6 shadow-panel sm:p-8">
            <div className="mb-6 flex flex-wrap items-end justify-between gap-4"><div><h2 className="text-xl font-black text-forest">Normalized data</h2><p className="mt-1 text-sm text-slate-500">The exact clean tables queried by the agent.</p></div><div className="flex rounded-xl bg-paper p-1 border border-forest/10">{(["deals", "work_orders"] as const).map((name) => <button key={name} onClick={() => setDataTable(name)} className={`rounded-lg px-4 py-2 text-xs font-bold capitalize transition ${dataTable === name ? "bg-white text-forest shadow-sm" : "text-slate-500 hover:text-forest"}`}>{name.replace("_", " ")}</button>)}</div></div>
            {dataLoading ? <p className="py-12 text-center text-sm text-slate-500">Loading normalized rows…</p> : data && <div className="space-y-3 w-full"><div className="flex items-center gap-2 text-xs text-slate-500"><strong className="text-forest">{data.rowcount} rows</strong><span>&bull;</span><span>{data.columns.length} columns</span><StatusPill value={data.cache} /></div><DataTable rows={data.rows} columns={data.columns} /></div>}
          </section>}
        </div>
      </main>
    </div>
  );
}
