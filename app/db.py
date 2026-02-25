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

    def _table_columns(self, table: str) -> set[str]:
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {r[1] for r in rows}

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                filepath TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

        emb_cols = self._table_columns("embeddings")
        if emb_cols and "model_name" not in emb_cols:
            self.conn.execute("DROP TABLE embeddings")
        edge_cols = self._table_columns("edges")
        if edge_cols and "model_name" not in edge_cols:
            self.conn.execute("DROP TABLE edges")

        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                image_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                vector BLOB NOT NULL,
                dim INTEGER NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(image_id, model_name),
                FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS edges (
                src_id TEXT NOT NULL,
                dst_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                similarity REAL NOT NULL,
                PRIMARY KEY (src_id, dst_id, model_name)
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

    def delete_images_by_paths(self, filepaths: list[str]) -> list[str]:
        if not filepaths:
            return []
        q = ",".join("?" for _ in filepaths)
        rows = self.conn.execute(
            f"SELECT id FROM images WHERE filepath IN ({q})", filepaths
        ).fetchall()
        ids = [r["id"] for r in rows]
        self.conn.execute(f"DELETE FROM images WHERE filepath IN ({q})", filepaths)
        self.conn.commit()
        return ids

    def images_by_path(self) -> dict[str, sqlite3.Row]:
        rows = self.conn.execute("SELECT * FROM images").fetchall()
        return {r["filepath"]: r for r in rows}

    def all_images(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM images ORDER BY filename").fetchall()

    def save_embedding(
        self, image_id: str, model_name: str, vector_blob: bytes, dim: int
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO embeddings (image_id, model_name, vector, dim, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(image_id, model_name) DO UPDATE SET
                vector=excluded.vector,
                dim=excluded.dim,
                updated_at=CURRENT_TIMESTAMP
            """,
            (image_id, model_name, vector_blob, dim),
        )

    def get_embedding_rows(self, model_name: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT image_id, vector, dim FROM embeddings WHERE model_name=?",
            (model_name,),
        ).fetchall()

    def get_images_missing_embeddings(self, model_name: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT i.* FROM images i
            LEFT JOIN embeddings e ON e.image_id=i.id AND e.model_name=?
            WHERE e.image_id IS NULL
            ORDER BY i.filename
            """,
            (model_name,),
        ).fetchall()

    def clear_edges(self, model_name: str) -> None:
        self.conn.execute("DELETE FROM edges WHERE model_name=?", (model_name,))

    def insert_edges(
        self, model_name: str, edges: list[tuple[str, str, float]]
    ) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO edges (src_id, dst_id, model_name, similarity) VALUES (?, ?, ?, ?)",
            [(a, b, model_name, s) for (a, b, s) in edges],
        )
        self.conn.commit()

    def set_setting(self, key: str, value: Any) -> None:
        payload = json.dumps(value)
        self.conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, payload),
        )
        self.conn.commit()

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute(
            "SELECT value FROM settings WHERE key=?", (key,)
        ).fetchone()
        if not row:
            return default
        return json.loads(row["value"])

    def status_counts(self, model_name: str) -> dict[str, int]:
        images = self.conn.execute("SELECT COUNT(*) AS c FROM images").fetchone()["c"]
        embeddings = self.conn.execute(
            "SELECT COUNT(*) AS c FROM embeddings WHERE model_name=?", (model_name,)
        ).fetchone()["c"]
        edges = self.conn.execute(
            "SELECT COUNT(*) AS c FROM edges WHERE model_name=?", (model_name,)
        ).fetchone()["c"]
        return {"images": images, "embeddings": embeddings, "edges": edges}

    def graph(self, model_name: str, limit: int, min_sim: float) -> dict[str, Any]:
        nodes = self.conn.execute(
            "SELECT id, filename, width, height FROM images ORDER BY filename LIMIT ?",
            (limit,),
        ).fetchall()
        node_ids = [n["id"] for n in nodes]
        if not node_ids:
            return {"nodes": [], "edges": []}

        placeholders = ",".join("?" for _ in node_ids)
        edge_rows = self.conn.execute(
            f"""
            SELECT src_id, dst_id, similarity FROM edges
            WHERE model_name=? AND similarity >= ?
              AND src_id IN ({placeholders})
              AND dst_id IN ({placeholders})
            ORDER BY similarity DESC
            """,
            (model_name, min_sim, *node_ids, *node_ids),
        ).fetchall()

        layout = self.get_setting(f"layout::{model_name}", default={})
        out_nodes = []
        for n in nodes:
            coords = layout.get(n["id"], {})
            out_nodes.append(
                {
                    "id": n["id"],
                    "filename": n["filename"],
                    "width": n["width"],
                    "height": n["height"],
                    "x": coords.get("x"),
                    "y": coords.get("y"),
                    "cluster": coords.get("cluster", -1),
                }
            )

        out_edges = [dict(e) for e in edge_rows]
        return {"nodes": out_nodes, "edges": out_edges}
