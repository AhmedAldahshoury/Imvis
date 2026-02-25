# syno-photo-graph

Local CPU-only photo similarity graph MVP built with FastAPI, SQLite, ONNX Runtime, and a D3.js static frontend.

## Features
- Recursive ingestion from `IMAGE_DIR` (default `/data/images`) for jpg/jpeg/png/webp.
- Stable image IDs via `sha1(path + size + mtime)`.
- SQLite persistence for image metadata, embeddings, graph edges, and indexing status metadata.
- Auto thumbnail generation (`/data/thumbs/<id>.jpg`, max 256px).
- ONNX Runtime (CPU) embedding generation with per-model storage and first-run model download into `/data/models`.
- Top-k similarity graph (`k=8`) from cosine similarity.
- Background indexing job with progress (`/api/status`), model switching (`/api/models`, `/api/config`), and reindex trigger (`/api/reindex`).
- Optional recursive folder watcher (watchdog) with debounce for automatic incremental indexing.
- Interactive D3 force graph with thumbnails, search, similarity threshold filtering, and detail panel.

## Quickstart
1. Put images in `./sample_images` (or generate demo images with `python scripts/generate_sample_images.py`).
2. Start the app:
   ```bash
   docker compose up --build
   ```
3. Open: http://localhost:8000

## API
- `GET /api/status`
- `POST /api/reindex`
- `GET /api/graph?limit=1000&min_sim=0.0`
- `GET /thumbs/<id>.jpg`
- `GET /image/<id>?max_dim=1280`

## Local development
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make lint
make test
uvicorn app.main:app --reload
```

## Notes
- First indexing run downloads the ONNX model to `/data/models` automatically.
- Graph payload is cached at `/data/cache/graph_cache.json` for instant reloads.


## Model switching and automatic watching
- Available models: `clip_b32` and `mobileclip` (both ONNX CPU encoders).
- UI exposes model dropdown + **Apply & Reindex**; changing model rebuilds embeddings/edges for that model.
- Embeddings and edges are stored per model in SQLite.
- Watcher monitors `IMAGE_DIR` recursively for jpg/jpeg/png/webp changes and runs incremental indexing after debounce.

### Env vars
- `ACTIVE_MODEL` (default `clip_b32`)
- `WATCH_ENABLED` (default `true`)
- `WATCH_DEBOUNCE_SECONDS` (default `20`)
- `INDEX_THROTTLE_IMGS_PER_SEC` (default `2`)

### New API endpoints
- `GET /api/models`
- `GET /api/config`
- `POST /api/config`
- `POST /api/reindex` accepts `{"full": false, "model": "clip_b32"}`
- `GET /api/graph?model=<optional>&limit=1000&min_sim=0.0`
