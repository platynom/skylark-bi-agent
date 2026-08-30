"""
monday.com GraphQL API v2 client.

Design notes
------------
* READ-ONLY BY CONSTRUCTION. `_post` rejects any document containing a
  mutation before it is ever sent. The integration spec is read-only and we
  enforce it in code rather than trusting the token's scopes.
* Board data is fetched dynamically every time the cache expires. Nothing
  about the CSVs is baked into this module -- board *names* are the only
  configuration, and even those fall back to fuzzy matching.
* monday paginates items with an opaque cursor and enforces a per-minute
  complexity budget, so we page in chunks and back off on 429/5xx.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from . import config


class MondayError(RuntimeError):
    """Raised for any unrecoverable problem talking to monday.com."""


_MUTATION_RE = re.compile(r"\bmutation\b", re.IGNORECASE)

_ITEM_FIELDS = """
  cursor
  items {
    id
    name
    created_at
    updated_at
    column_values { id text type value }
  }
"""


@dataclass
class Board:
    id: str
    name: str
    columns: list[dict[str, str]]
    items: list[dict[str, Any]] = field(default_factory=list)

    @property
    def column_titles(self) -> list[str]:
        return [c["title"] for c in self.columns]


class MondayClient:
    def __init__(self, token: str | None = None, url: str | None = None,
                 timeout: int = 60, max_retries: int = 4):
        self.token = token or config.MONDAY_API_TOKEN
        self.url = url or config.MONDAY_API_URL
        self.timeout = timeout
        self.max_retries = max_retries
        if not self.token:
            raise MondayError(
                "MONDAY_API_TOKEN is not set. Add it to Streamlit secrets or the "
                "environment before starting the agent."
            )
        self._session = requests.Session()

    # ------------------------------------------------------------------ #
    # transport
    # ------------------------------------------------------------------ #
    def _post(self, query: str, variables: dict | None = None) -> dict:
        if _MUTATION_RE.search(query):
            raise MondayError("Refused: this client is read-only and will not send mutations.")

        headers = {
            "Authorization": self.token,
            "Content-Type": "application/json",
            "API-Version": config.MONDAY_API_VERSION,
        }
        payload = {"query": query, "variables": variables or {}}

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(
                    self.url, json=payload, headers=headers, timeout=self.timeout
                )
            except requests.RequestException as exc:
                last_err = exc
                time.sleep(2 ** attempt)
                continue

            # Rate limit / complexity budget exhausted / transient server error.
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                delay = int(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
                last_err = MondayError(f"monday.com returned HTTP {resp.status_code}")
                time.sleep(min(delay, 30))
                continue

            if resp.status_code == 401:
                raise MondayError(
                    "monday.com rejected the API token (401). Regenerate it under "
                    "Avatar -> Developers -> My Access Tokens."
                )
            if resp.status_code >= 400:
                raise MondayError(f"monday.com HTTP {resp.status_code}: {resp.text[:400]}")

            body = resp.json()
            if body.get("errors"):
                msg = "; ".join(e.get("message", str(e)) for e in body["errors"])
                # Complexity errors are retryable after a pause.
                if "complexity" in msg.lower() and attempt < self.max_retries - 1:
                    last_err = MondayError(msg)
                    time.sleep(10)
                    continue
                raise MondayError(f"monday.com GraphQL error: {msg}")
            if "data" not in body:
                raise MondayError(f"Unexpected monday.com response: {str(body)[:400]}")
            return body["data"]

        raise MondayError(f"monday.com unreachable after {self.max_retries} attempts: {last_err}")

    # ------------------------------------------------------------------ #
    # discovery
    # ------------------------------------------------------------------ #
    def list_boards(self) -> list[dict[str, str]]:
        """Every board the token can see. Used for board-name resolution and diagnostics."""
        out: list[dict[str, str]] = []
        page = 1
        while True:
            data = self._post(
                "query ($page: Int!) { boards (limit: 50, page: $page, "
                "state: active) { id name items_count } }",
                {"page": page},
            )
            batch = data.get("boards") or []
            out.extend(batch)
            if len(batch) < 50:
                break
            page += 1
            if page > 20:  # hard stop, nobody has 1000 boards in this exercise
                break
        return out

    def resolve_board_id(self, explicit_id: str | None, wanted_name: str) -> str:
        """Explicit ID wins; otherwise match on board name, case/space-insensitively."""
        if explicit_id:
            return str(explicit_id).strip()

        boards = self.list_boards()
        norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
        target = norm(wanted_name)

        exact = [b for b in boards if norm(b["name"]) == target]
        if exact:
            return str(exact[0]["id"])
        partial = [b for b in boards if target in norm(b["name"]) or norm(b["name"]) in target]
        if len(partial) == 1:
            return str(partial[0]["id"])
        if len(partial) > 1:
            names = ", ".join(f"{b['name']} ({b['id']})" for b in partial)
            raise MondayError(
                f"Board name '{wanted_name}' is ambiguous -- matched: {names}. "
                f"Set the board ID explicitly in secrets."
            )
        available = ", ".join(b["name"] for b in boards) or "(none visible to this token)"
        raise MondayError(
            f"No board matching '{wanted_name}'. Boards visible to this token: {available}"
        )

    # ------------------------------------------------------------------ #
    # data
    # ------------------------------------------------------------------ #
    def fetch_board(self, board_id: str, page_size: int = 100) -> Board:
        """Fetch a board's schema and every item, following the cursor to the end."""
        data = self._post(
            """
            query ($ids: [ID!], $limit: Int!) {
              boards (ids: $ids) {
                id
                name
                columns { id title type }
                items_page (limit: $limit) { %s }
              }
            }
            """ % _ITEM_FIELDS,
            {"ids": [str(board_id)], "limit": page_size},
        )
        boards = data.get("boards") or []
        if not boards:
            raise MondayError(
                f"Board {board_id} returned no data. Check the ID and that the token's "
                f"account can see it."
            )
        raw = boards[0]
        board = Board(
            id=str(raw["id"]),
            name=raw["name"],
            columns=[
                {"id": c["id"], "title": c["title"], "type": c["type"]}
                for c in raw.get("columns", [])
            ],
        )

        page = raw.get("items_page") or {}
        board.items.extend(page.get("items") or [])
        cursor = page.get("cursor")

        guard = 0
        while cursor:
            guard += 1
            if guard > 200:
                raise MondayError("Pagination guard tripped -- refusing to loop further.")
            nxt = self._post(
                "query ($cursor: String!, $limit: Int!) { next_items_page "
                "(cursor: $cursor, limit: $limit) { %s } }" % _ITEM_FIELDS,
                {"cursor": cursor, "limit": page_size},
            )
            page = nxt.get("next_items_page") or {}
            board.items.extend(page.get("items") or [])
            cursor = page.get("cursor")

        return board

    def health(self) -> dict[str, Any]:
        """Cheap connectivity + identity probe for the UI status panel."""
        data = self._post("query { me { id name email account { name } } }")
        me = data.get("me") or {}
        return {
            "user": me.get("name"),
            "email": me.get("email"),
            "account": (me.get("account") or {}).get("name"),
        }
