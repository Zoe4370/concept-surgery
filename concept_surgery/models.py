"""Named model presets for reproducible cross-model experiments.

The registry keeps short, stable CLI aliases separate from upstream model
IDs.  Arbitrary Hugging Face IDs remain accepted by the runner.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict


@dataclass(frozen=True)
class ModelPreset:
    alias: str
    model_id: str
    family: str
    scale: str
    role: str


MODEL_PRESETS: Dict[str, ModelPreset] = {
    "gpt2": ModelPreset("gpt2", "gpt2", "GPT-2", "124M", "legacy baseline"),
    "qwen-coder-0.5b": ModelPreset(
        "qwen-coder-0.5b", "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "Qwen2.5-Coder", "0.5B", "instruction-tuned coding model",
    ),
    "qwen-coder-1.5b": ModelPreset(
        "qwen-coder-1.5b", "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "Qwen2.5-Coder", "1.5B", "instruction-tuned coding model",
    ),
    "qwen-coder-3b": ModelPreset(
        "qwen-coder-3b", "Qwen/Qwen2.5-Coder-3B-Instruct",
        "Qwen2.5-Coder", "3B", "instruction-tuned coding model",
    ),
    "qwen-coder-7b": ModelPreset(
        "qwen-coder-7b", "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen2.5-Coder", "7B", "instruction-tuned coding model (high memory)",
    ),
    "starcoder2-3b": ModelPreset(
        "starcoder2-3b", "bigcode/starcoder2-3b",
        "StarCoder2", "3B", "code-pretrained base model (not instruction-tuned)",
    ),
    "phi-2": ModelPreset(
        "phi-2", "microsoft/phi-2", "Phi-2", "2.7B",
        "general model with code training data",
    ),
}


def resolve_model_id(name: str) -> str:
    """Resolve a preset alias to its upstream ID; otherwise pass through."""
    return MODEL_PRESETS[name].model_id if name in MODEL_PRESETS else name


def model_slug(name: str) -> str:
    """Return a filesystem-safe, deterministic slug for a model alias/ID."""
    if name in MODEL_PRESETS:
        return name
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._").lower()
    if not slug:
        raise ValueError("model name must contain at least one alphanumeric character")
    return slug
