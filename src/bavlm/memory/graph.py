"""Temporal knowledge graph of player tendencies (SQLite).

Scoping note, repeated here because it is easy to get wrong when writing up:
this does NOT contribute to FBBench accuracy. FBBench asks about individual
clips. Memory answers cross-rally questions and is evaluated separately,
through the scouting demonstration.

No pose data is required. Tendencies are built from stroke type, court region
and rally outcome -- all of which come from the annotations or the agent.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS observation (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    player      TEXT    NOT NULL,
    match_id    TEXT    NOT NULL,
    rally_id    TEXT    NOT NULL,
    shot_index  INTEGER NOT NULL,
    hit_type    TEXT,
    subtype     TEXT,
    court_region TEXT,
    quality     INTEGER,            -- FineBadminton 1..7
    rally_won   INTEGER,            -- 1 / 0 / NULL
    match_date  TEXT,
    UNIQUE(match_id, rally_id, shot_index)
);
CREATE INDEX IF NOT EXISTS idx_player       ON observation(player);
CREATE INDEX IF NOT EXISTS idx_player_region ON observation(player, court_region);
"""


@dataclass
class Tendency:
    hit_type: str
    court_region: str
    n: int
    win_rate: float
    mean_quality: float | None


class MemoryGraph:
    def __init__(self, path: str = "results/memory.sqlite") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def add(self, **row) -> None:
        cols = ", ".join(row)
        marks = ", ".join("?" * len(row))
        self.conn.execute(
            f"INSERT OR IGNORE INTO observation ({cols}) VALUES ({marks})",
            tuple(row.values()),
        )

    def commit(self) -> None:
        self.conn.commit()

    def tendencies(self, player: str, min_n: int = 5) -> list[Tendency]:
        """What this player does, and how it works out for them."""
        q = """
        SELECT hit_type, court_region,
               COUNT(*)            AS n,
               AVG(COALESCE(rally_won, 0.0)) AS win_rate,
               AVG(quality)        AS mean_quality
        FROM observation
        WHERE player = ? AND hit_type IS NOT NULL
        GROUP BY hit_type, court_region
        HAVING n >= ?
        ORDER BY n DESC
        """
        return [
            Tendency(r["hit_type"], r["court_region"], r["n"],
                     round(r["win_rate"], 3),
                     round(r["mean_quality"], 2) if r["mean_quality"] is not None else None)
            for r in self.conn.execute(q, (player, min_n))
        ]

    def weaknesses(self, player: str, min_n: int = 5, threshold: float = 0.4) -> list[Tendency]:
        """Region/shot combinations this player loses from. The scouting report
        is built from these, and every row links back to its observations."""
        return [t for t in self.tendencies(player, min_n) if t.win_rate < threshold]
