create table if not exists warehouse_cache (
  key text primary key,
  payload jsonb not null,
  fetched_at timestamptz not null default now()
);