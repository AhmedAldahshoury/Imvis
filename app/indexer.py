from __future__ import annotations

import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import Settings
from app.db import Database
from app.embeddings import MultiModelEmbedder, topk_cosine_graph
from app.model_registry import MODEL_SPECS
from app.scanner import discover_images


@dataclass
class _RowImage:
    id: str
    filepath: str


def project_vectors_2d(vectors: np.ndarray) -> np.ndarray:
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
    last_index_time: str | None = None
    last_error: str | None = None
    active_model: str = "clip_b32"
    watcher_enabled: bool = False
    dirty_flag: bool = False
    last_change_time: str | None = None
    current_job_state: str = "idle"
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "stage": self.stage,
                "progress_current": self.progress_current,
                "progress_total": self.progress_total,
                "last_index_time": self.last_index_time,
                "last_error": self.last_error,
                "active_model": self.active_model,
                "watcher_enabled": self.watcher_enabled,
                "dirty_flag": self.dirty_flag,
                "last_change_time": self.last_change_time,
                "current_job_state": self.current_job_state,
            }


class Indexer:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings
        self.embedder = MultiModelEmbedder(settings.model_dir)
        active = self.db.get_setting("active_model", settings.default_active_model)
        if active not in MODEL_SPECS:
            active = "clip_b32"
        self.db.set_setting("active_model", active)
        self.status = IndexStatus(active_model=active)
        self._lock = threading.Lock()
        self._pending = False
        self._pending_full = False
        self._pending_model: str | None = None

    def _set(self, **kwargs: object) -> None:
        with self.status.lock:
            for k, v in kwargs.items():
                setattr(self.status, k, v)

    def set_active_model(self, model: str) -> None:
        self.db.set_setting("active_model", model)
        self._set(active_model=model)

    def mark_dirty(self, timestamp: str | None = None) -> None:
        self._set(
            dirty_flag=True,
            last_change_time=timestamp or datetime.now(timezone.utc).isoformat(),
        )
        self.kickoff(full=False)

    def kickoff(self, full: bool = False, model: str | None = None) -> bool:
        start = False
        with self._lock:
            self._pending = True
            self._pending_full = self._pending_full or full
            self._pending_model = model or self._pending_model
            if not self.status.running:
                self.status.running = True
                start = True
        if start:
            threading.Thread(target=self._run_loop, daemon=True).start()
        return start

    def _thumbnail(self, source: Path, img_id: str) -> None:
        out = self.settings.thumbs_dir / f"{img_id}.jpg"
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((256, 256))
            image.save(out, format="JPEG", quality=85)

    def _build_thumbnail(self, rec: _RowImage) -> str:
        self._thumbnail(Path(rec.filepath), rec.id)
        return rec.id

    def _embed_vector(self, rec: _RowImage, model_name: str) -> tuple[str, np.ndarray]:
        path = Path(rec.filepath)
        vec = self.embedder.embed(path, model_name)
        return rec.id, vec

    def _run_parallel_jobs(self, records: list[_RowImage], stage: str, workers: int, job):
        self._set(stage=stage, progress_total=len(records), progress_current=0)
        if not records:
            return []

        throttle = self.settings.index_throttle_imgs_per_sec
        submit_interval = (1.0 / throttle) if throttle > 0 else 0.0
        next_submit_at = 0.0

        results = []
        completed = 0
        pending = set()
        submit_index = 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            while submit_index < len(records) or pending:
                now = time.monotonic()
                while submit_index < len(records) and len(pending) < workers and now >= next_submit_at:
                    pending.add(pool.submit(job, records[submit_index]))
                    submit_index += 1
                    next_submit_at = now + submit_interval if submit_interval > 0 else now
                    now = time.monotonic()

                if not pending:
                    sleep_for = max(0.001, next_submit_at - time.monotonic())
                    time.sleep(sleep_for)
                    continue

                done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    results.append(future.result())
                    completed += 1
                    self._set(progress_current=completed)

        return results


    def _run_loop(self) -> None:
        while True:
            with self._lock:
                if not self._pending:
                    self.status.running = False
                    self.status.current_job_state = "idle"
                    break
                full = self._pending_full
                model = self._pending_model
                self._pending = False
                self._pending_full = False
                self._pending_model = None
            try:
                self._set(current_job_state="indexing")
                self._run_once(full=full, model=model)
            except Exception as exc:
                self._set(last_error=str(exc), stage="error")

    def _run_once(self, full: bool, model: str | None) -> None:
        active_model = self.db.get_setting(
            "active_model", self.settings.default_active_model
        )
        target_model = model or active_model
        self._set(
            active_model=active_model,
            stage="scan",
            progress_current=0,
            progress_total=0,
            last_error=None,
        )
        records = list(discover_images(self.settings.image_dir))
        existing = self.db.images_by_path()
        current_paths = {r.filepath for r in records}
        deleted_paths = [p for p in existing if p not in current_paths]
        deleted_ids = self.db.delete_images_by_paths(deleted_paths)
        for image_id in deleted_ids:
            thumb = self.settings.thumbs_dir / f"{image_id}.jpg"
            if thumb.exists():
                thumb.unlink()

        changed_records = []
        for rec in records:
            old = existing.get(rec.filepath)
            if not old or old["size"] != rec.size or old["mtime"] != rec.mtime:
                changed_records.append(rec)
                if old and old["id"] != rec.id:
                    self.db.delete_images_by_paths([rec.filepath])
        self.db.upsert_images(records)

        self._run_parallel_jobs(
            records=changed_records,
            stage="thumbs",
            workers=self.settings.thumbnail_workers,
            job=self._build_thumbnail,
        )

        if full:
            embedding_rows = records
        else:
            missing = self.db.get_images_missing_embeddings(target_model)
            changed_paths = {r.filepath for r in changed_records}
            embedding_rows = changed_records + [
                _RowImage(row["id"], row["filepath"])
                for row in missing
                if row["filepath"] not in changed_paths
            ]

        embedding_results = self._run_parallel_jobs(
            records=embedding_rows,
            stage="embeddings",
            workers=self.settings.embedding_workers,
            job=lambda rec: self._embed_vector(rec, target_model),
        )
        for image_id, vec in embedding_results:
            self.db.save_embedding(
                image_id, target_model, vec.astype(np.float32).tobytes(), vec.shape[0]
            )
        self.db.conn.commit()

        rows = self.db.get_embedding_rows(target_model)
        ids = [r["image_id"] for r in rows]
        vectors = [np.frombuffer(r["vector"], dtype=np.float32) for r in rows]

        self._set(stage="edges", progress_total=len(ids), progress_current=0)
        if vectors:
            matrix = np.vstack(vectors).astype(np.float32)
            edges = topk_cosine_graph(ids, matrix, top_k=self.settings.top_k)
            projected = project_vectors_2d(matrix)
            target_clusters = (
                int(np.clip(np.sqrt(len(ids) / 2), 2, 8)) if len(ids) >= 6 else 1
            )
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
        self.db.clear_edges(target_model)
        self.db.insert_edges(target_model, edges)
        self.db.set_setting(f"layout::{target_model}", layout)

        payload = self.db.graph(
            model_name=target_model, limit=self.settings.default_limit, min_sim=0.0
        )
        cache_file = self.settings.cache_dir / f"graph_cache_{target_model}.json"
        cache_file.write_text(json.dumps(payload))

        ts = datetime.now(timezone.utc).isoformat()
        self.db.set_setting("last_index_time", ts)
        self._set(
            stage="done",
            progress_current=len(ids),
            progress_total=len(ids),
            last_index_time=ts,
            dirty_flag=False,
        )
