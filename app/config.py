from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    image_dir: Path = Path(os.getenv("IMAGE_DIR", "/data/images"))
    data_dir: Path = Path(os.getenv("DATA_DIR", "/data"))
    db_path: Path = Path(os.getenv("DB_PATH", "/data/syno_photo_graph.db"))
    model_dir: Path = Path(os.getenv("MODEL_DIR", "/data/models"))
    thumbs_dir: Path = Path(os.getenv("THUMBS_DIR", "/data/thumbs"))
    cache_dir: Path = Path(os.getenv("CACHE_DIR", "/data/cache"))
    top_k: int = int(os.getenv("TOP_K", "8"))
    default_limit: int = int(os.getenv("DEFAULT_GRAPH_LIMIT", "1000"))
    watch_enabled: bool = os.getenv("WATCH_ENABLED", "true").lower() == "true"
    watch_debounce_seconds: int = int(os.getenv("WATCH_DEBOUNCE_SECONDS", "20"))
    index_throttle_imgs_per_sec: float = float(
        os.getenv("INDEX_THROTTLE_IMGS_PER_SEC", "2")
    )
    thumbnail_workers: int = max(1, int(os.getenv("THUMBNAIL_WORKERS", "4")))
    embedding_workers: int = max(1, int(os.getenv("EMBEDDING_WORKERS", "4")))
    default_active_model: str = os.getenv("ACTIVE_MODEL", "clip_b32")


settings = Settings()


def ensure_dirs() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.thumbs_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.image_dir.mkdir(parents=True, exist_ok=True)
