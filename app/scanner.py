from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ImageRecord:
    id: str
    filepath: str
    filename: str
    size: int
    mtime: float
    width: int
    height: int


def stable_id(path: Path, size: int, mtime: float) -> str:
    key = f"{path.resolve()}::{size}::{mtime}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTS


def discover_images(root: Path) -> Iterable[ImageRecord]:
    from PIL import Image

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            if not _is_image(path):
                continue
            stat = path.stat()
            img_id = stable_id(path, stat.st_size, stat.st_mtime)
            try:
                with Image.open(path) as img:
                    width, height = img.size
            except Exception:
                continue
            yield ImageRecord(
                id=img_id,
                filepath=str(path),
                filename=path.name,
                size=stat.st_size,
                mtime=stat.st_mtime,
                width=width,
                height=height,
            )
