#!/usr/bin/env python3
"""
One-off board provisioner: creates the Deals and Work Orders boards on monday.com
from the CSVs in monday_import/ and populates them.

This is SETUP TOOLING, not part of the agent. The agent's own client
(`agent/monday_client.py`) refuses to transmit mutations by design; this script
uses its own writer so that guarantee stays intact and unqualified.

It exists so a reviewer can stand the whole thing up with one command instead of
clicking through monday's import wizard twice:

    python scripts/setup_boards.py --token <MONDAY_API_TOKEN>

Column-type choice
------------------
Every column is created as `text`, deliberately. monday's importer coerces on
ingest and silently discards values it cannot parse -- which would quietly repair
the exact defects this assignment asks the agent to handle (the repeated header
rows, the unit-bearing quantity cells). Importing as text is lossless; typing and
validation happen in `agent/normalize.py`, where coercion failures are counted and
reported to the user instead of vanishing at the boundary.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import requests

API = "https://api.monday.com/v2"
ROOT = Path(__file__).resolve().parents[1]


def gql(token: str, query: str, variables: dict | None = None, retries: int = 5) -> dict:
    headers = {"Authorization": token, "Content-Type": "application/json",
               "API-Version": "2024-10"}
    for attempt in range(retries):
        r = requests.post(API, json={"query": query, "variables": variables or {}},
                          headers=headers, timeout=90)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2 ** attempt * 2, 30))
            continue
        if r.status_code >= 400:
            raise SystemExit(f"HTTP {r.status_code}: {r.text[:500]}")
        body = r.json()
        if body.get("errors"):
            msg = "; ".join(e.get("message", "") for e in body["errors"])
            if "complexity" in msg.lower() and attempt < retries - 1:
                wait = 10
                m = re.search(r"reset in (\d+)", msg)
                if m:
                    wait = int(m.group(1)) + 2
                print(f"    complexity budget hit, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise SystemExit(f"GraphQL error: {msg}")
        return body["data"]
    raise SystemExit("monday.com unreachable after retries.")


def create_board(token: str, name: str) -> str:
    data = gql(token,
               'mutation ($n: String!) { create_board (board_name: $n, board_kind: public) { id } }',
               {"n": name})
    return str(data["create_board"]["id"])


def create_columns(token: str, board_id: str, titles: list[str]) -> dict[str, str]:
    """Create one text column per CSV header (skipping the first, which monday's
    built-in item Name field carries). Returns {csv_title: monday_column_id}."""
    mapping: dict[str, str] = {}
    for i, t in enumerate(titles):
        data = gql(token,
                   'mutation ($b: ID!, $t: String!) { create_column '
                   '(board_id: $b, title: $t, column_type: text) { id title } }',
                   {"b": board_id, "t": t[:255]})
        mapping[t] = data["create_column"]["id"]
        print(f"    column {i+1}/{len(titles)}: {t[:48]}")
        time.sleep(0.15)
    return mapping


def create_items(token: str, board_id: str, header: list[str], rows: list[list[str]],
                 colmap: dict[str, str], batch: int = 5) -> int:
    """Batch create_item mutations with GraphQL aliases to cut round trips."""
    made = 0
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        parts, variables, defs = [], {}, []
        for j, row in enumerate(chunk):
            row = row + [""] * (len(header) - len(row))
            cv = {colmap[h]: row[i] for i, h in enumerate(header)
                  if i > 0 and h in colmap and row[i] not in ("", None)}
            defs += [f"$n{j}: String!", f"$c{j}: JSON!"]
            variables[f"n{j}"] = (row[0] or f"(unnamed {start+j})")[:255]
            variables[f"c{j}"] = json.dumps(cv)
            parts.append(
                f'i{j}: create_item (board_id: {board_id}, item_name: $n{j}, '
                f'column_values: $c{j}, create_labels_if_missing: false) {{ id }}'
            )
        gql(token, "mutation (" + ", ".join(defs) + ") { " + " ".join(parts) + " }", variables)
        made += len(chunk)
        print(f"    items {made}/{len(rows)}", end="\r", flush=True)
        time.sleep(0.35)
    print()
    return made


def load_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    return rows[0], rows[1:]


def main() -> int:
    ap = argparse.ArgumentParser(description="Provision Skylark BI monday.com boards.")
    ap.add_argument("--token", required=True, help="monday.com API v2 token")
    ap.add_argument("--deals-csv", default=str(ROOT / "monday_import" / "deals.csv"))
    ap.add_argument("--wo-csv", default=str(ROOT / "monday_import" / "work_orders.csv"))
    ap.add_argument("--deals-name", default="Deals")
    ap.add_argument("--wo-name", default="Work Orders")
    args = ap.parse_args()

    me = gql(args.token, "query { me { name account { name } } }")["me"]
    print(f"Authenticated as {me.get('name')} ({(me.get('account') or {}).get('name')})\n")

    ids = {}
    for label, path, name in ((("deals"), Path(args.deals_csv), args.deals_name),
                              (("work_orders"), Path(args.wo_csv), args.wo_name)):
        if not path.exists():
            raise SystemExit(f"Missing CSV: {path}")
        header, rows = load_csv(path)
        print(f"{name}: {len(rows)} rows x {len(header)} columns")
        bid = create_board(args.token, name)
        print(f"  board id {bid}")
        colmap = create_columns(args.token, bid, header[1:])
        n = create_items(args.token, bid, header, rows, colmap)
        print(f"  {n} items created\n")
        ids[label] = bid

    print("Done. Add these to your Streamlit secrets:\n")
    print(f'DEALS_BOARD_ID       = "{ids["deals"]}"')
    print(f'WORK_ORDERS_BOARD_ID = "{ids["work_orders"]}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
