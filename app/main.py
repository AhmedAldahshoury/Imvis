from __future__ import annotations

import io
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.config import ensure_dirs, settings
from app.db import Database
from app.indexer import Indexer
from app.model_registry import MODEL_SPECS, list_models
from app.watcher import FolderWatcher

ensure_dirs()
db = Database(settings.db_path)
indexer = Indexer(db, settings)
watcher: FolderWatcher | None = None

app = FastAPI(title="Syno Photo Graph")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def startup_index() -> None:
    global watcher
    indexer.kickoff(full=False)
    if settings.watch_enabled:
        watcher = FolderWatcher(
            root=settings.image_dir,
            debounce_seconds=settings.watch_debounce_seconds,
            on_dirty=lambda: indexer.mark_dirty(),
        )
        enabled = watcher.start()
        indexer._set(watcher_enabled=enabled)


@app.on_event("shutdown")
def shutdown() -> None:
    if watcher:
        watcher.stop()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return Path("static/index.html").read_text(encoding="utf-8")


@app.get("/api/models")
def api_models() -> dict:
    return {"models": list_models()}


@app.get("/api/config")
def api_config() -> dict:
    active_model = db.get_setting("active_model", settings.default_active_model)
    return {
        "active_model": active_model,
        "watch_enabled": settings.watch_enabled,
        "watch_debounce_seconds": settings.watch_debounce_seconds,
        "index_throttle_imgs_per_sec": settings.index_throttle_imgs_per_sec,
    }


@app.post("/api/config")
def api_set_config(payload: dict) -> dict:
    model = payload.get("active_model")
    if model not in MODEL_SPECS:
        raise HTTPException(status_code=400, detail="unknown model")
    indexer.set_active_model(model)
    started = indexer.kickoff(full=True, model=model)
    return {"active_model": model, "started": started}


@app.get("/api/status")
def api_status() -> dict:
    active = db.get_setting("active_model", settings.default_active_model)
    counts = db.status_counts(active)
    snap = indexer.status.snapshot()
    if watcher:
        snap["dirty_flag"] = watcher.dirty_flag
        snap["last_change_time"] = watcher.last_change_time
    return {**snap, **counts}


@app.post("/api/reindex")
def api_reindex(payload: dict | None = None) -> dict:
    payload = payload or {}
    full = bool(payload.get("full", False))
    model = payload.get("model")
    if model and model not in MODEL_SPECS:
        raise HTTPException(status_code=400, detail="unknown model")
    started = indexer.kickoff(full=full, model=model)
    return {"started": started, "full": full, "model": model}


@app.get("/api/graph")
def api_graph(
    model: str | None = None,
    limit: int = Query(default=settings.default_limit, ge=1, le=5000),
    min_sim: float = Query(default=0.0, ge=-1.0, le=1.0),
):
    active = model or db.get_setting("active_model", settings.default_active_model)
    if limit == settings.default_limit and min_sim <= 0.0:
        cache_file = settings.cache_dir / f"graph_cache_{active}.json"
        if cache_file.exists():
            return JSONResponse(content=json.loads(cache_file.read_text()))
    return db.graph(model_name=active, limit=limit, min_sim=min_sim)


@app.get("/thumbs/{image_id}.jpg")
def thumbs(image_id: str):
    path = settings.thumbs_dir / f"{image_id}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="thumbnail not found")
    return FileResponse(path)


@app.get("/image/{image_id}")
def image(image_id: str, max_dim: int = Query(default=1280, ge=256, le=4000)):
    row = db.conn.execute(
        "SELECT filepath FROM images WHERE id=?", (image_id,)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="image not found")
    src = Path(row["filepath"]).resolve()
    image_root = settings.image_dir.resolve()
    if image_root not in src.parents and src != image_root:
        raise HTTPException(status_code=403, detail="forbidden")

    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
    return Response(content=buf.getvalue(), media_type="image/jpeg")
