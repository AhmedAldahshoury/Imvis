from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from app.scanner import ImageRecord


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                filepath TEXT NOT NULL,
                filename TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                image_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                dim INTEGER NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS edges (
                src_id TEXT NOT NULL,
                dst_id TEXT NOT NULL,
                similarity REAL NOT NULL,
                PRIMARY KEY (src_id, dst_id)
            );

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def upsert_images(self, records: Iterable[ImageRecord]) -> int:
        count = 0
        for r in records:
            self.conn.execute(
                """
                INSERT INTO images (id, filepath, filename, size, mtime, width, height, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    filepath=excluded.filepath,
                    filename=excluded.filename,
                    size=excluded.size,
                    mtime=excluded.mtime,
                    width=excluded.width,
                    height=excluded.height,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (r.id, r.filepath, r.filename, r.size, r.mtime, r.width, r.height),
            )
            count += 1
        self.conn.commit()
        return count

    def all_images(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM images ORDER BY filename").fetchall()

    def save_embedding(self, image_id: str, vector_blob: bytes, dim: int) -> None:
        self.conn.execute(
            """
            INSERT INTO embeddings (image_id, vector, dim, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(image_id) DO UPDATE SET
                vector=excluded.vector,
                dim=excluded.dim,
                updated_at=CURRENT_TIMESTAMP
            """,
            (image_id, vector_blob, dim),
        )

    def get_embedding_rows(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT image_id, vector, dim FROM embeddings").fetchall()

    def clear_edges(self) -> None:
        self.conn.execute("DELETE FROM edges")

    def insert_edges(self, edges: list[tuple[str, str, float]]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO edges (src_id, dst_id, similarity) VALUES (?, ?, ?)", edges
        )
        self.conn.commit()

    def set_metadata(self, key: str, value: Any) -> None:
        payload = json.dumps(value)
        self.conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, payload),
        )
        self.conn.commit()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        return json.loads(row["value"])

    def status_counts(self) -> dict[str, int]:
        images = self.conn.execute("SELECT COUNT(*) AS c FROM images").fetchone()["c"]
        embeddings = self.conn.execute("SELECT COUNT(*) AS c FROM embeddings").fetchone()["c"]
        edges = self.conn.execute("SELECT COUNT(*) AS c FROM edges").fetchone()["c"]
        return {"images": images, "embeddings": embeddings, "edges": edges}

    def graph(self, limit: int, min_sim: float) -> dict[str, Any]:
        nodes = self.conn.execute(
            "SELECT id, filename, width, height FROM images ORDER BY filename LIMIT ?", (limit,)
        ).fetchall()
        node_ids = [n["id"] for n in nodes]
        if not node_ids:
            return {"nodes": [], "edges": []}
        placeholders = ",".join("?" for _ in node_ids)
        edges = self.conn.execute(
            f"""
            SELECT src_id, dst_id, similarity
            FROM edges
            WHERE similarity >= ?
              AND src_id IN ({placeholders})
              AND dst_id IN ({placeholders})
            """,
            [min_sim, *node_ids, *node_ids],
        ).fetchall()
        return {
            "nodes": [dict(n) for n in nodes],
            "edges": [dict(e) for e in edges],
        }
