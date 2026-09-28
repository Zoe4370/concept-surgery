"""Activation capture via PyTorch forward hooks.

Works with any HuggingFace causal LM whose decoder layers expose a
``.mlp``/``.moe``-style container — GPT-2, Llama, Mistral, Qwen, Phi-3.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def get_decoder_layers(model) -> List:
    """Return the list of decoder layer modules for common HF architectures."""
    m = model
    candidates = [
        getattr(m, "layers", None),                 # llama/mistral/qwen style
        getattr(getattr(m, "model", None), "layers", None),
        getattr(getattr(m, "transformer", None), "h", None),  # gpt-2
        getattr(getattr(m, "model", None), "decoder", None) and getattr(m.model, "decoder").layers,  # phi-3 wraps llama
    ]
    for c in candidates:
        if c is not None and len(c) > 0:
            return list(c)
    raise ValueError(
        f"Could not locate decoder layers for {type(m).__name__}; "
        "pass layers explicitly or add an entry to get_decoder_layers()."
    )


def load_model_and_tokenizer(model_name: str, device: str = "auto", dtype: str = "float32"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer, device


@torch.no_grad()
def capture_activations(
    model,
    tokenizer,
    prompts: List[str],
    layer_indices: List[int],
    device: str = "cpu",
    position: str = "last",
    batch_size: int = 8,
) -> Dict[int, np.ndarray]:
    """Capture hidden states at the chosen decoder layers.

    Returns ``{layer_index: (n_prompts, d) float32 array}``. When
    ``position == "last"``, activations are read at the final token of each
    prompt (the usual choice for concept directions). With ``"all"``, the
    mean over all prompt tokens is returned instead.
    """
    layers = get_decoder_layers(model)
    captured: Dict[int, List[torch.Tensor]] = {i: [] for i in layer_indices}
    hooks = []

    def make_hook(idx):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[idx].append(h.detach().to("cpu", torch.float32))
        return hook

    for i in layer_indices:
        hooks.append(layers[i].register_forward_hook(make_hook(i)))

    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, padding_side="left").to(device)
            model(**enc)
            for i in layer_indices:
                h = captured[i][-1]  # (batch, seq, d)
                if position == "last":
                    sel = h[:, -1, :]  # left padding guarantees final position is the true last token
                else:
                    mask = enc["attention_mask"].bool()
                    sel = torch.stack([hh[m].mean(0) for hh, m in zip(h, mask)])
                captured[i][-1] = sel
    finally:
        for h in hooks:
            h.remove()

    return {i: torch.cat(chunks, dim=0).numpy() for i, chunks in captured.items()}
