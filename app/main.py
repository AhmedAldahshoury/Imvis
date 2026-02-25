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

ensure_dirs()
db = Database(settings.db_path)
indexer = Indexer(db, settings)

app = FastAPI(title="Syno Photo Graph")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def startup_index() -> None:
    indexer.kickoff()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return Path("static/index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def api_status() -> dict:
    counts = db.status_counts()
    return {**indexer.status.snapshot(), **counts}


@app.post("/api/reindex")
def api_reindex() -> dict:
    started = indexer.kickoff()
    return {"started": started}


@app.get("/api/graph")
def api_graph(
    limit: int = Query(default=settings.default_limit, ge=1, le=5000),
    min_sim: float = Query(default=0.0, ge=-1.0, le=1.0),
):
    if limit == settings.default_limit and min_sim <= 0.0:
        cache_file = settings.cache_dir / "graph_cache.json"
        if cache_file.exists():
            return JSONResponse(content=json.loads(cache_file.read_text()))
    return db.graph(limit=limit, min_sim=min_sim)


@app.get("/thumbs/{image_id}.jpg")
def thumbs(image_id: str):
    path = settings.thumbs_dir / f"{image_id}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="thumbnail not found")
    return FileResponse(path)


@app.get("/image/{image_id}")
def image(image_id: str, max_dim: int = Query(default=1280, ge=256, le=4000)):
    row = db.conn.execute("SELECT filepath FROM images WHERE id=?", (image_id,)).fetchone()
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
