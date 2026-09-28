"""Inference-time concept ablation: the actual "surgery".

Installs forward hooks that project hidden states out of a concept's
subspace *during generation*. The model's weights are never modified.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Optional

import numpy as np
import torch


class ConceptAblator:
    """Hold per-layer erasers and apply them inside forward hooks.

    Parameters
    ----------
    directions : {layer_index: (k, d) ndarray}
        Unit-norm concept directions per layer.
    mode : "project" removes the component entirely;
           "pad" replaces it with ``pad_epsilon`` (ablative ablation).
    """

    def __init__(
        self,
        directions: Dict[int, np.ndarray],
        mode: str = "project",
        pad_epsilon: float = 0.0,
    ):
        self.directions = {i: np.asarray(v, dtype=np.float32) for i, v in directions.items()}
        self.mode = mode
        self.pad_epsilon = pad_epsilon
        self.active = False
        self._hooks = []

    def _apply(self, layer_idx: int, h: torch.Tensor) -> torch.Tensor:
        v = self.directions.get(layer_idx)
        if v is None or v.size == 0 or not self.active:
            return h
        orig_shape = h.shape
        H = h.reshape(-1, orig_shape[-1]).to(torch.float32)
        V = torch.from_numpy(v).to(H.device, torch.float32)  # (k, d)
        comp = H @ V.T                                       # (..., k)
        H = H - comp @ V
        if self.mode == "pad":
            H = H + self.pad_epsilon * V.sum(0)
        return H.to(orig_shape and h.dtype).reshape(orig_shape)

    def _make_hook(self, idx: int):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h_new = self._apply(idx, h)
            if isinstance(out, tuple):
                return (h_new,) + out[1:]
            return h_new
        return hook

    @contextmanager
    def applied(self, model):
        """Context manager: enable ablation on the model for the duration."""
        from .capture import get_decoder_layers

        layers = get_decoder_layers(model)
        try:
            for idx in self.directions:
                if idx >= len(layers):
                    raise IndexError(f"layer {idx} out of range ({len(layers)} layers)")
                self._hooks.append(layers[idx].register_forward_hook(self._make_hook(idx)))
            self.active = True
            yield self
        finally:
            self.active = False
            for h in self._hooks:
                h.remove()
            self._hooks = []

    def to(self, device: str):
        self.directions = {i: torch.from_numpy(v).to(device).numpy() for i, v in self.directions.items()}
        return self
