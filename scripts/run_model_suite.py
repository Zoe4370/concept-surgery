#!/usr/bin/env python3
"""Run one identical concept-erasure protocol over several causal LMs.

Example:
    python scripts/run_model_suite.py --models gpt2 qwen-coder-0.5b qwen-coder-1.5b qwen-coder-3b starcoder2-3b phi-2

Each model is loaded and evaluated sequentially. Individual failures are
recorded so a memory or access issue for one model does not discard completed
runs for the others.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from concept_surgery.config import ErasureConfig
from concept_surgery.models import MODEL_PRESETS, model_slug, resolve_model_id


def _summary_row(alias: str, model_id: str, results: Dict) -> Dict:
    chosen = results.get("chosen", {})
    baseline = results.get("baseline", {})
    edited = results.get("ablated_single", results.get("ablated", {}))
    return {
        "alias": alias,
        "model_id": model_id,
        "status": "completed",
        "chosen_layer": chosen.get("layer"),
        "chosen_method": chosen.get("method"),
        "baseline_probe_accuracy": baseline.get("probe_acc"),
        "edited_probe_accuracy": edited.get("probe_acc"),
        "baseline_neutral_nll": baseline.get("neutral_nll"),
        "edited_neutral_nll": edited.get("neutral_nll"),
        "locality_kl": results.get("locality_kl"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODEL_PRESETS))
    parser.add_argument("--concept", default="mickey_mouse")
    parser.add_argument("--neutral", default="neutral")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--methods", nargs="+", default=["mean_diff", "leace", "inlp"],
                        choices=["mean_diff", "pca", "leace", "inlp"])
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/model_suite")
    parser.add_argument("--resume", action="store_true", help="reuse models with an existing results.json")
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first failed model")
    parser.add_argument("--list-models", action="store_true", help="print available presets and exit")
    args = parser.parse_args()

    if args.list_models:
        for preset in MODEL_PRESETS.values():
            print(f"{preset.alias:20} {preset.model_id:48} {preset.scale:>5}  {preset.role}")
        return 0

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "suite_results.json"
    rows: List[Dict] = []

    for requested in args.models:
        model_id = resolve_model_id(requested)
        alias = requested if requested in MODEL_PRESETS else model_slug(requested)
        result_path = root / alias / "results.json"
        if args.resume and result_path.exists():
            try:
                cached = json.loads(result_path.read_text())
                rows.append(_summary_row(alias, model_id, cached))
                print(f"[suite] reusing {alias}")
                continue
            except (OSError, json.JSONDecodeError):
                print(f"[suite] could not reuse {result_path}; rerunning {alias}")

        print(f"\n[suite] {alias}: {model_id}")
        try:
            from concept_surgery.pipeline import run_pipeline

            cfg = ErasureConfig(
                model_name=model_id,
                concept_dataset=args.concept,
                neutral_dataset=args.neutral,
                device=args.device,
                dtype=args.dtype,
                methods=args.methods,
                layers=args.layers,
                seed=args.seed,
                output_dir=str(root / alias),
            )
            results = run_pipeline(cfg)
            rows.append(_summary_row(alias, model_id, results))
        except Exception as exc:  # keep other model runs usable
            rows.append({"alias": alias, "model_id": model_id, "status": "failed", "error": str(exc)})
            print(f"[suite] {alias} failed: {type(exc).__name__}: {exc}", file=sys.stderr)

        manifest_path.write_text(json.dumps({"seed": args.seed, "models": rows}, indent=2))
        if rows[-1]["status"] == "failed" and args.fail_fast:
            break

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    manifest_path.write_text(json.dumps({"seed": args.seed, "models": rows}, indent=2))
    completed = sum(row["status"] == "completed" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    print(f"\n[suite] {completed} completed, {failed} failed; summary: {manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
