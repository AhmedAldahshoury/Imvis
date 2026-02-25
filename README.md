# syno-photo-graph

Local CPU-only photo similarity graph MVP built with FastAPI, SQLite, ONNX Runtime, and a D3.js static frontend.

## Features
- Recursive ingestion from `IMAGE_DIR` (default `/data/images`) for jpg/jpeg/png/webp.
- Stable image IDs via `sha1(path + size + mtime)`.
- SQLite persistence for image metadata, embeddings, graph edges, and indexing status metadata.
- Auto thumbnail generation (`/data/thumbs/<id>.jpg`, max 256px).
- ONNX Runtime (CPU) embedding generation with first-run model download into `/data/models`.
- Top-k similarity graph (`k=8`) from cosine similarity.
- Background indexing job with progress (`/api/status`) and reindex trigger (`/api/reindex`).
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
