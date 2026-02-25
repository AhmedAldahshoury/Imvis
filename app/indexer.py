from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import Settings
from app.db import Database
from app.embeddings import Embedder, topk_cosine_graph
from app.scanner import discover_images


def project_vectors_2d(vectors: np.ndarray) -> np.ndarray:
    """Project embedding vectors to a stable 2D coordinate system via PCA."""
    if vectors.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    if centered.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T
    if basis.shape[1] < 2:
        basis = np.pad(basis, ((0, 0), (0, 2 - basis.shape[1])))
    projected = centered @ basis
    scale = np.max(np.abs(projected), axis=0)
    scale[scale == 0] = 1.0
    return (projected / scale).astype(np.float32)


def kmeans_labels(points: np.ndarray, k: int, iterations: int = 12) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=np.int32)
    k = max(1, min(k, len(points)))
    if k == 1:
        return np.zeros(len(points), dtype=np.int32)

    centroids = points[np.linspace(0, len(points) - 1, num=k, dtype=int)].copy()
    labels = np.zeros(len(points), dtype=np.int32)
    for _ in range(iterations):
        distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(distances, axis=1).astype(np.int32)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for idx in range(k):
            members = points[labels == idx]
            if len(members):
                centroids[idx] = members.mean(axis=0)
    return labels


@dataclass
class IndexStatus:
    running: bool = False
    stage: str = "idle"
    progress_current: int = 0
    progress_total: int = 0
    last_run: str | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "stage": self.stage,
                "progress_current": self.progress_current,
                "progress_total": self.progress_total,
                "last_run": self.last_run,
                "error": self.error,
            }


class Indexer:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings
        self.embedder = Embedder(settings.model_dir)
        self.status = IndexStatus(last_run=self.db.get_metadata("last_run"))

    def _set(self, **kwargs: object) -> None:
        with self.status.lock:
            for k, v in kwargs.items():
                setattr(self.status, k, v)

    def kickoff(self) -> bool:
        if self.status.running:
            return False
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        return True

    def _thumbnail(self, source: Path, img_id: str) -> None:
        out = self.settings.thumbs_dir / f"{img_id}.jpg"
        if out.exists():
            return
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((256, 256))
            image.save(out, format="JPEG", quality=85)

    def _run(self) -> None:
        self._set(running=True, stage="scan", progress_current=0, progress_total=0, error=None)
        try:
            records = list(discover_images(self.settings.image_dir))
            self.db.upsert_images(records)

            self._set(stage="thumbnails", progress_total=len(records), progress_current=0)
            for i, rec in enumerate(records, start=1):
                self._thumbnail(Path(rec.filepath), rec.id)
                self._set(progress_current=i)

            self._set(stage="embeddings", progress_total=len(records), progress_current=0)
            vectors: list[np.ndarray] = []
            ids: list[str] = []
            for i, rec in enumerate(records, start=1):
                path = Path(rec.filepath)
                vec = self.embedder.embed(path)
                self.db.save_embedding(rec.id, vec.astype(np.float32).tobytes(), vec.shape[0])
                ids.append(rec.id)
                vectors.append(vec)
                self._set(progress_current=i)
            self.db.conn.commit()

            self._set(stage="graph", progress_total=len(ids), progress_current=0)
            if vectors:
                matrix = np.vstack(vectors).astype(np.float32)
                edges = topk_cosine_graph(ids, matrix, top_k=self.settings.top_k)
                projected = project_vectors_2d(matrix)
                target_clusters = int(np.clip(np.sqrt(len(ids) / 2), 2, 8)) if len(ids) >= 6 else 1
                labels = kmeans_labels(projected, target_clusters)
                layout = {
                    image_id: {
                        "x": float(projected[i, 0]),
                        "y": float(projected[i, 1]),
                        "cluster": int(labels[i]),
                    }
                    for i, image_id in enumerate(ids)
                }
            else:
                edges = []
                layout = {}
            self.db.clear_edges()
            self.db.insert_edges(edges)
            self.db.set_metadata("layout", layout)

            payload = self.db.graph(limit=self.settings.default_limit, min_sim=0.0)
            cache_file = self.settings.cache_dir / "graph_cache.json"
            cache_file.write_text(json.dumps(payload))

            ts = datetime.now(timezone.utc).isoformat()
            self.db.set_metadata("last_run", ts)
            self._set(last_run=ts)
            self._set(stage="done", progress_current=len(ids), progress_total=len(ids))
        except Exception as exc:
            self._set(error=str(exc), stage="error")
        finally:
            self._set(running=False)
