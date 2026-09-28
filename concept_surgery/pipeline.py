"""End-to-end erasure pipeline.

    map  ->  fit  ->  select  ->  ablate  ->  evaluate  ->  report
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .ablate import ConceptAblator
from .capture import capture_activations, get_decoder_layers, load_model_and_tokenizer
from .config import ErasureConfig
from .core_math import fit_direction, fit_directions_inlp, fit_leace_eraser, pca_difference_directions
from .data import load_prompts
from .evaluate import (
    average_nll,
    generate,
    linear_probe_accuracy,
    locality_kl,
)


def select_layers(model, cfg: ErasureConfig) -> List[int]:
    if cfg.layers is not None:
        return list(cfg.layers)
    n = len(get_decoder_layers(model))
    lo, hi = int(0.4 * n), int(0.9 * n)
    idx = np.linspace(lo, max(hi, lo + 1), min(6, max(hi - lo, 1))).round().astype(int)
    return sorted(set(int(i) for i in idx if 0 <= i < n))


def fit_all_methods(X_pos, X_neg, cfg: ErasureConfig) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for m in cfg.methods:
        if m == "mean_diff":
            out[m] = fit_direction(X_pos, X_neg)[None, :]
        elif m == "pca":
            out[m] = pca_difference_directions(X_pos, X_neg, k=cfg.n_pca_directions)
        elif m == "leace":
            P = fit_leace_eraser(X_pos, X_neg)
            # recover the erased subspace from (I - P); its principal
            # right-singular vector is the concept direction
            D = np.eye(P.shape[0]) - P
            u, s, _ = np.linalg.svd(D)
            keep = s > 0.1 * (s[0] if len(s) else 1.0)
            out[m] = u[:, keep].T if keep.any() else u[:, :1].T
        elif m == "inlp":
            out[m] = fit_directions_inlp(X_pos, X_neg, k=cfg.n_inlp_directions, seed=cfg.seed)
        else:
            raise ValueError(f"unknown method {m}")
    return out


def run_pipeline(cfg: ErasureConfig) -> Dict:
    import torch

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model_and_tokenizer(cfg.model_name, cfg.device, cfg.dtype)
    concept_prompts = load_prompts(cfg.concept_dataset)
    neutral_prompts = load_prompts(cfg.neutral_dataset)
    layers = select_layers(model, cfg)
    print(f"[pipeline] model={cfg.model_name} device={device} candidate_layers={layers}")

    # --- 1. MAP: capture activations -------------------------------------
    cap_pos = capture_activations(model, tokenizer, concept_prompts, layers, device, cfg.capture_positions)
    cap_neg = capture_activations(model, tokenizer, neutral_prompts, layers, device, cfg.capture_positions)

    # --- 2. FIT + 3. SELECT ----------------------------------------------
    # For each (layer, method) apply a temporary single-layer ablation and
    # score it: post-ablation probe accuracy (want ~0.5) at minimal cost to
    # neutral perplexity (want ~unchanged).
    base_nll = average_nll(model, tokenizer, neutral_prompts[:20], device)
    selection = []
    for i in layers:
        for name, dirs in fit_all_methods(cap_pos[i], cap_neg[i], cfg).items():
            trial = ConceptAblator({i: dirs}, mode=cfg.ablate_mode, pad_epsilon=cfg.pad_epsilon)
            with trial.applied(model):
                acc = linear_probe_accuracy(
                    capture_activations(model, tokenizer, concept_prompts[:40], [i], device)[i],
                    capture_activations(model, tokenizer, neutral_prompts[:40], [i], device)[i],
                    seed=cfg.seed,
                )
                cost = average_nll(model, tokenizer, neutral_prompts[:20], device)
            score = abs(acc - 0.5) + max(0.0, cost - base_nll)
            selection.append({"layer": i, "method": name, "post_probe_acc": acc,
                              "neutral_nll": cost, "score": score})
            print(f"[select] layer={i:2d} method={name:9s} post_probe_acc={acc:.3f} "
                  f"neutral_nll={cost:.3f} score={score:.3f}")

    best = min(selection, key=lambda s: s["score"])
    print(f"[select] chosen: layer={best['layer']} method={best['method']}")

    best_dirs = fit_all_methods(cap_pos[best["layer"]], cap_neg[best["layer"]], cfg)[best["method"]]
    ablate_layers = cfg.ablate_layers or [best["layer"]]
    ablator = ConceptAblator(
        {i: best_dirs for i in ablate_layers}, mode=cfg.ablate_mode, pad_epsilon=cfg.pad_epsilon
    )

    # Multi-layer arm: later layers can *re-derive* an erased feature from
    # upstream context, so a local ablation may not survive downstream.
    # Fit the same method at every candidate layer and compare.
    multi_dirs = {i: fit_all_methods(cap_pos[i], cap_neg[i], cfg)[best["method"]] for i in layers}
    multi_ablator = ConceptAblator(multi_dirs, mode=cfg.ablate_mode, pad_epsilon=cfg.pad_epsilon)

    # Iterative refit: directions fitted on clean activations go stale once
    # earlier layers are ablated (the residual distribution shifts). Refit
    # under the current ablation stream while the top-layer probe keeps
    # collapsing; stop at the first non-improvement.
    for it in range(cfg.refit_iters):
        with multi_ablator.applied(model):
            cap_pos_i = capture_activations(model, tokenizer, concept_prompts, layers, device)
            cap_neg_i = capture_activations(model, tokenizer, neutral_prompts, layers, device)
        new_dirs = {i: fit_all_methods(cap_pos_i[i], cap_neg_i[i], cfg)[best["method"]] for i in layers}
        trial_ablator = ConceptAblator(new_dirs, mode=cfg.ablate_mode, pad_epsilon=cfg.pad_epsilon)
        trial_card = _scorecard(model, tokenizer, device, concept_prompts, neutral_prompts,
                                layers, trial_ablator, cfg)
        card_prev = _scorecard(model, tokenizer, device, concept_prompts, neutral_prompts,
                               layers, multi_ablator, cfg)
        if abs(trial_card["probe_acc"] - 0.5) <= abs(card_prev["probe_acc"] - 0.5) - 0.01:
            multi_ablator = trial_ablator
            print(f"[refit] iteration {it + 1}: top-layer probe "
                  f"{card_prev['probe_acc']:.3f} -> {trial_card['probe_acc']:.3f}")
        else:
            print(f"[refit] iteration {it + 1}: no improvement "
                  f"({card_prev['probe_acc']:.3f} -> {trial_card['probe_acc']:.3f}); stopping")
            break

    # --- 4. PROOF: before/after scorecard --------------------------------
    results = {"config": vars(cfg), "selection": selection, "chosen": best}
    results["baseline"] = _scorecard(model, tokenizer, device, concept_prompts, neutral_prompts,
                                     layers, None, cfg)
    results["ablated_single"] = _scorecard(model, tokenizer, device, concept_prompts,
                                           neutral_prompts, layers, ablator, cfg)
    results["ablated_multi"] = _scorecard(model, tokenizer, device, concept_prompts,
                                          neutral_prompts, layers, multi_ablator, cfg)

    def _arm_score(arm):
        return (abs(arm["probe_acc"] - 0.5)
                + max(0.0, arm["neutral_nll"] - results["baseline"]["neutral_nll"]))

    if _arm_score(results["ablated_multi"]) <= _arm_score(results["ablated_single"]):
        results["ablated"], results["mode"] = results["ablated_multi"], "multi-layer"
        ablator = multi_ablator
    else:
        results["ablated"], results["mode"] = results["ablated_single"], "single-layer"
    print(f"[select] final ablation mode: {results['mode']}")

    results["locality_kl"] = locality_kl(
        model, tokenizer, ablator, neutral_prompts[: cfg.locality_kl_tokens], device
    )
    results["generations"] = _side_by_side(model, tokenizer, device, concept_prompts, neutral_prompts,
                                           ablator, cfg)

    with (outdir / "results.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[pipeline] wrote {outdir / 'results.json'}")
    return results


def _scorecard(model, tokenizer, device, concept_prompts, neutral_prompts, layers, ablator, cfg):
    # capture INSIDE the ablation context so probes see the edited model
    ctx = ablator.applied(model) if ablator is not None else _null_ctx()
    with ctx:
        cap_pos = capture_activations(model, tokenizer, concept_prompts, layers, device)
        cap_neg = capture_activations(model, tokenizer, neutral_prompts, layers, device)
        card = {}
        for i in layers:
            card[f"probe_acc_layer{i}"] = linear_probe_accuracy(cap_pos[i], cap_neg[i], seed=cfg.seed)
        top = layers[-1]
        card["probe_acc"] = card[f"probe_acc_layer{top}"]
        card["concept_nll"] = average_nll(model, tokenizer, concept_prompts[:30], device)
        card["neutral_nll"] = average_nll(model, tokenizer, neutral_prompts[:30], device)
    card["loss_ratio"] = card["neutral_nll"] / max(card["concept_nll"], 1e-6)
    return card


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _side_by_side(model, tokenizer, device, concept_prompts, neutral_prompts, ablator, cfg):
    rows = []
    test_prompts = concept_prompts[:6] + neutral_prompts[:4]
    for p in test_prompts:
        base = generate(model, tokenizer, p, device, cfg.max_gen_tokens)
        with ablator.applied(model):
            edit = generate(model, tokenizer, p, device, cfg.max_gen_tokens)
        rows.append({"prompt": p, "baseline": base, "ablated": edit})
        print(f"\n[prompt] {p}\n  before: {base[:160]}\n  after : {edit[:160]}")
    return rows
