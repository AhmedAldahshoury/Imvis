from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
MODEL_FILENAME = "squeezenet1.1-7.onnx"


class Embedder:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model_path = model_dir / MODEL_FILENAME
        self._session: ort.InferenceSession | None = None

    def ensure_model(self) -> Path:
        if self.model_path.exists():
            return self.model_path
        import urllib.request

        self.model_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, self.model_path)
        return self.model_path

    @property
    def session(self) -> ort.InferenceSession:
        if self._session is None:
            model_path = self.ensure_model()
            self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        return self._session

    def _preprocess(self, image_path: Path) -> np.ndarray:
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return arr[np.newaxis, ...].astype(np.float32)

    def embed(self, image_path: Path) -> np.ndarray:
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        vec = self.session.run([output_name], {input_name: self._preprocess(image_path)})[0][0]
        vec = vec.astype(np.float32).flatten()
        norm = np.linalg.norm(vec) + 1e-12
        return vec / norm


def topk_cosine_graph(
    ids: Sequence[str], vectors: np.ndarray, top_k: int = 8
) -> list[tuple[str, str, float]]:
    sim = vectors @ vectors.T
    edges: dict[tuple[str, str], float] = {}
    for i, src_id in enumerate(ids):
        idx = np.argpartition(-sim[i], top_k + 1)[: top_k + 1]
        idx = [j for j in idx if j != i]
        for j in idx:
            dst_id = ids[j]
            a, b = sorted((src_id, dst_id))
            score = float(sim[i, j])
            if (a, b) not in edges or score > edges[(a, b)]:
                edges[(a, b)] = score
    return [(a, b, s) for (a, b), s in edges.items()]
