"""Evaluation harness: specificity, collateral damage, and locality.

An erasure is only credible if it survives adversarial measurement:

1. Specificity — can the concept still be *linearly decoded* from hidden
   states? We fit a logistic probe with cross-validation before and after
   ablation. A prompt-level "I don't know" is not erasure; probe accuracy
   at chance level is.
2. Collateral damage — general capability must survive: perplexity on a
   neutral corpus before vs. after.
3. Locality — the KL divergence between the baseline and edited next-token
   distributions on *neutral* prompts (small = the edit is surgical).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .capture import format_prompts


# --------------------------------------------------------------------------
# 1. Linear-probe specificity
# --------------------------------------------------------------------------

def linear_probe_accuracy(
    X_pos: np.ndarray, X_neg: np.ndarray, n_splits: int = 5, seed: int = 0
) -> float:
    """Cross-validated accuracy of a logistic probe decoding concept
    presence from hidden states. 0.5 = chance for balanced classes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X[tr], y[tr])
        accs.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs))


# --------------------------------------------------------------------------
# 2. Generation + perplexity utilities
# --------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 60) -> str:
    prompt = format_prompts(tokenizer, [prompt], add_generation_prompt=True)[0]
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def average_nll(model, tokenizer, texts: List[str], device: str, batch_size: int = 4) -> float:
    """Mean negative log-likelihood per token over a list of texts."""
    nlls, n_tokens = [], 0
    for start in range(0, len(texts), batch_size):
        batch = format_prompts(tokenizer, texts[start : start + batch_size])
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        out = model(**enc, labels=labels)
        # recompute per-token NLL to normalize by real token count
        shift_mask = enc["attention_mask"][:, 1:].bool()
        logits = out.logits[:, :-1]
        targets = enc["input_ids"][:, 1:]
        tok_nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view(targets.shape)
        picked = tok_nll[shift_mask]
        nlls.append(picked.sum().item())
        n_tokens += int(shift_mask.sum().item())
    return float(np.sum(nlls) / max(n_tokens, 1))


@torch.no_grad()
def next_token_kl(model, tokenizer, prompts: List[str], device: str) -> float:
    """Mean KL(baseline || ablated) over next-token distributions.
    Call twice: once with the ablator inactive, once active, then diff the
    returned probability tables via :func:`kl_between`."""
    probs = []
    for prompt in prompts:
        prompt = format_prompts(tokenizer, [prompt], add_generation_prompt=True)[0]
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        logits = model(**enc).logits[0, -1]
        probs.append(torch.softmax(logits, dim=-1).cpu())
    return torch.stack(probs)  # (n_prompts, vocab)


def kl_between(p_base: torch.Tensor, p_edit: torch.Tensor, eps: float = 1e-9) -> float:
    kl = (p_base * ((p_base + eps).log() - (p_edit + eps).log())).sum(dim=-1)
    return float(kl.mean().item())


# --------------------------------------------------------------------------
# 3. The full scorecard
# --------------------------------------------------------------------------

def evaluate_erasure(
    model,
    tokenizer,
    device: str,
    concept_prompts: List[str],
    neutral_prompts: List[str],
    layer_indices: List[int],
    ablator=None,
    n_probe_splits: int = 5,
    seed: int = 0,
    n_locality: int = 25,
) -> Dict[str, float]:
    """Run before/after metrics. Pass ``ablator=None`` for the baseline
    pass; pass an (inactive) :class:`ConceptAblator` to measure the edited
    model."""
    from .capture import capture_activations

    layers_str = ",".join(map(str, layer_indices))
    cap_concept = capture_activations(model, tokenizer, concept_prompts, layer_indices, device)
    cap_neutral = capture_activations(model, tokenizer, neutral_prompts, layer_indices, device)

    metrics: Dict[str, float] = {}
    for i in layer_indices:
        metrics[f"probe_acc_layer{i}"] = linear_probe_accuracy(
            cap_concept[i], cap_neutral[i], n_splits=n_probe_splits, seed=seed
        )

    # aggregated view over the deepest captured layer
    top = layer_indices[-1]
    metrics["probe_acc"] = metrics[f"probe_acc_layer{top}"]

    # concept vs neutral loss ratio (recognition without probing)
    metrics["concept_nll"] = average_nll(model, tokenizer, concept_prompts[:30], device)
    metrics["neutral_nll"] = average_nll(model, tokenizer, neutral_prompts[:30], device)
    metrics["loss_ratio"] = metrics["neutral_nll"] / max(metrics["concept_nll"], 1e-6)
    return metrics


@torch.no_grad()
def locality_kl(
    model,
    tokenizer,
    ablator,
    prompts: List[str],
    device: str,
) -> float:
    """Mean KL(baseline || ablated) next-token distributions on neutral
    prompts. Small values mean the edit left ordinary behavior intact."""
    base = next_token_kl(model, tokenizer, prompts, device)
    with ablator.applied(model):
        edit = next_token_kl(model, tokenizer, prompts, device)
    return kl_between(base, edit)
