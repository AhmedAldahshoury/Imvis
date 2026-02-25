from pathlib import Path

from app.db import Database


def test_schema_tables_created(tmp_path: Path):
    db = Database(tmp_path / "test.sqlite")
    names = {
        row[0]
        for row in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"images", "embeddings", "edges", "metadata"}.issubset(names)
