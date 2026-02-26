from __future__ import annotations

import threading
import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.model_registry import get_model_spec, model_file_path


class MultiModelEmbedder:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._sessions: dict[str, ort.InferenceSession] = {}
        self._session_lock = threading.Lock()

    def ensure_model(self, model_name: str) -> Path:
        spec = get_model_spec(model_name)
        model_path = model_file_path(self.model_dir, model_name)
        if model_path.exists() and model_path.stat().st_size > 1024:
            return model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(spec.url, model_path)
        if model_path.stat().st_size <= 1024:
            raise RuntimeError(f"downloaded model is invalid for {model_name}")
        return model_path

    def session(self, model_name: str) -> ort.InferenceSession:
        session = self._sessions.get(model_name)
        if session is not None:
            return session
        with self._session_lock:
            session = self._sessions.get(model_name)
            if session is not None:
                return session
            model_path = self.ensure_model(model_name)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._sessions[model_name] = session
            return session

    def _preprocess(self, image_path: Path, model_name: str) -> np.ndarray:
        spec = get_model_spec(model_name)
        image = (
            Image.open(image_path)
            .convert("RGB")
            .resize((spec.input_size, spec.input_size))
        )
        arr = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.array(spec.mean, dtype=np.float32)
        std = np.array(spec.std, dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        return arr.astype(np.float32)

    def embed(self, image_path: Path, model_name: str) -> np.ndarray:
        sess = self.session(model_name)
        input_name = sess.get_inputs()[0].name
        output = sess.run(None, {input_name: self._preprocess(image_path, model_name)})[
            0
        ]
        vec = output.reshape(-1).astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-12
        return vec / norm


def topk_cosine_graph(
    ids: Sequence[str], vectors: np.ndarray, top_k: int = 8
) -> list[tuple[str, str, float]]:
    if len(ids) <= 1:
        return []
    sim = vectors @ vectors.T
    edges: dict[tuple[str, str], float] = {}
    k = max(1, min(top_k, len(ids) - 1))
    for i, src_id in enumerate(ids):
        idx = np.argpartition(-sim[i], k + 1)[: k + 1]
        idx = [j for j in idx if j != i]
        for j in idx:
            dst_id = ids[j]
            a, b = sorted((src_id, dst_id))
            score = float(sim[i, j])
            if (a, b) not in edges or score > edges[(a, b)]:
                edges[(a, b)] = score
    return [(a, b, s) for (a, b), s in edges.items()]
