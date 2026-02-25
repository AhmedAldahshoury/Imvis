import base64
from pathlib import Path

import pytest
from app.scanner import discover_images, stable_id

pytest.importorskip("PIL")

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WlQvVQAAAAASUVORK5CYII="
)


def test_stable_id_is_deterministic(tmp_path: Path):
    p = tmp_path / "a.jpg"
    p.write_bytes(b"abc")
    first = stable_id(p, 3, 1.0)
    second = stable_id(p, 3, 1.0)
    assert first == second


def test_discover_images_reads_metadata(tmp_path: Path):
    img = tmp_path / "img.png"
    img.write_bytes(PNG_1X1)
    found = list(discover_images(tmp_path))
    assert len(found) == 1
    rec = found[0]
    assert rec.filename == "img.png"
    assert rec.width == 1
    assert rec.height == 1
