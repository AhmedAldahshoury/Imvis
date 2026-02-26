from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    description: str
    input_size: int
    embedding_dim: int
    speed_hint: str
    url: str
    filename: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


MODEL_SPECS: dict[str, ModelSpec] = {
    "clip_b32": ModelSpec(
        name="clip_b32",
        label="CLIP ViT-B/32",
        description="CLIP image encoder (ONNX, CPU). Better quality, slower.",
        input_size=224,
        embedding_dim=512,
        speed_hint="medium",
        url="https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model.onnx",
        filename="vision_model.onnx",
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
    "mobileclip": ModelSpec(
        name="mobileclip",
        label="MobileCLIP",
        description="Lightweight CLIP-like encoder (ONNX, CPU). Faster, lower quality.",
        input_size=224,
        embedding_dim=512,
        speed_hint="fast",
        url="https://huggingface.co/Xenova/mobileclip-1b/resolve/main/onnx/vision_model.onnx",
        filename="vision_model.onnx",
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
}


def list_models() -> list[dict[str, object]]:
    return [
        {
            "name": spec.name,
            "label": spec.label,
            "description": spec.description,
            "input_size": spec.input_size,
            "embedding_dim": spec.embedding_dim,
            "speed_hint": spec.speed_hint,
        }
        for spec in MODEL_SPECS.values()
    ]


def get_model_spec(model_name: str) -> ModelSpec:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_SPECS[model_name]


def model_file_path(model_dir: Path, model_name: str) -> Path:
    spec = get_model_spec(model_name)
    return model_dir / model_name / spec.filename
