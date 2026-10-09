"""Central configuration for an erasure experiment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ErasureConfig:
    # --- datasets ---
    concept_dataset: str = "mickey_mouse"   # built-in name or JSONL path
    neutral_dataset: str = "neutral"

    # --- model ---
    model_name: str = "gpt2"          # gpt2 | meta-llama/Meta-Llama-3.2-1B | microsoft/Phi-3-mini-4k-instruct
    device: str = "auto"              # "auto" | "cpu" | "cuda"
    dtype: str = "auto"                # device-compatible precision, or an explicit float32/float16/bfloat16 override

    # --- activation capture ---
    capture_positions: str = "last"   # "last" = final token only, "all" = every token
    token_source: str = "last"        # "last" (final prompt token) or "mean" (mean over prompt tokens)

    # --- direction fitting ---
    layers: Optional[List[int]] = None  # None = auto (mid-to-late layers)
    methods: List[str] = field(default_factory=lambda: ["mean_diff", "leace", "inlp"])
    n_inlp_directions: int = 4
    n_pca_directions: int = 4

    # --- ablation ---
    ablate_layers: Optional[List[int]] = None  # None = the single best layer found
    ablate_mode: str = "project"      # "project" | "pad" (replace component with a small constant)
    pad_epsilon: float = 0.0

    # --- evaluation ---
    refit_iters: int = 2              # refit-under-ablation rounds for the multi-layer arm
    eval_seeds: int = 5
    max_gen_tokens: int = 60
    locality_kl_tokens: int = 40      # neutral prompts used for the KL locality check

    seed: int = 0
    output_dir: str = "runs/default"
