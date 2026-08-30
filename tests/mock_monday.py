"""
Offline fixture that mimics a monday.com board response.

This exists so the normalisation and query layers can be tested without burning
API quota or requiring a live token in CI. It is a TEST FIXTURE ONLY -- the
application never reads a CSV at runtime; app.py always goes through
MondayClient against the live API.

It reproduces two behaviours of a real monday CSV import that matter:
  * the first CSV column is absorbed into the built-in item `name` field and its
    own column_value comes back blank;
  * every column_value is returned as a string in `text`, regardless of the
    column type monday auto-assigned.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from agent.monday_client import Board


def _col_id(title: str, i: int) -> str:
    return re.sub(r"[^a-z0-9]", "_", title.lower())[:20] + f"_{i}"


def board_from_csv(path: str | Path, board_id: str, board_name: str,
                   absorb_first_column: bool = True) -> Board:
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    header, body = rows[0], rows[1:]
    columns = [{"id": _col_id(t, i), "title": t, "type": "text"} for i, t in enumerate(header)]
    board = Board(id=board_id, name=board_name, columns=columns)

    for n, r in enumerate(body):
        r = r + [""] * (len(header) - len(r))
        cvs = []
        for i, c in enumerate(columns):
            val = r[i]
            if i == 0 and absorb_first_column:
                val = ""  # monday blanks it; the value lives in item.name
            cvs.append({"id": c["id"], "text": val, "type": "text", "value": None})
        board.items.append({
            "id": f"{board_id}{n:04d}",
            "name": r[0] if absorb_first_column else f"item-{n}",
            "created_at": None, "updated_at": None,
            "column_values": cvs,
        })
    return board
