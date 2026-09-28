#!/usr/bin/env python3
"""CLI for the concept-erasure pipeline.

Examples
--------
# quick local demo (GPT-2, CPU-friendly):
python scripts/run_erasure.py --model gpt2 --concept mickey_mouse

# a bigger target on Colab GPU:
python scripts/run_erasure.py --model meta-llama/Llama-3.2-1B --concept harry_potter

# custom prompt sets:
python scripts/run_erasure.py --concept data/my_concept.jsonl --neutral data/neutral.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from concept_surgery.config import ErasureConfig
from concept_surgery.pipeline import run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Surgical concept erasure for causal LMs")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--concept", default="mickey_mouse", help="built-in name or JSONL path")
    ap.add_argument("--neutral", default="neutral")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--methods", nargs="+", default=["mean_diff", "leace", "inlp"],
                    choices=["mean_diff", "pca", "leace", "inlp"])
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--mode", default="project", choices=["project", "pad"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/default")
    args = ap.parse_args()

    cfg = ErasureConfig(
        model_name=args.model,
        concept_dataset=args.concept,
        neutral_dataset=args.neutral,
        device=args.device,
        dtype=args.dtype,
        methods=args.methods,
        layers=args.layers,
        ablate_mode=args.mode,
        seed=args.seed,
        output_dir=args.out,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
