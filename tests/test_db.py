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
    assert {"images", "embeddings", "edges", "settings"}.issubset(names)


def test_graph_includes_layout_fields(tmp_path: Path):
    db = Database(tmp_path / "test.sqlite")
    db.conn.execute(
        """
        INSERT INTO images (id, filepath, filename, size, mtime, width, height)
        VALUES ('img1', '/tmp/img1.jpg', 'img1.jpg', 1, 1.0, 10, 10)
        """
    )
    db.conn.execute(
        """
        INSERT INTO embeddings (image_id, model_name, vector, dim)
        VALUES ('img1', 'clip_b32', ?, 3)
        """,
        (b"123",),
    )
    db.conn.commit()
    db.set_setting("layout::clip_b32", {"img1": {"x": 0.2, "y": -0.4, "cluster": 3}})

    graph = db.graph(model_name="clip_b32", limit=10, min_sim=0)

    assert len(graph["nodes"]) == 1
    node = graph["nodes"][0]
    assert node["x"] == 0.2
    assert node["y"] == -0.4
    assert node["cluster"] == 3
