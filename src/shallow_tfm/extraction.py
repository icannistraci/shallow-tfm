"""Capture of the representation after every block during one forward pass.

Depths are counted as follows: depth 0 is the input of the first block (the
tokenizer output, captured with a hook that runs before the block) and depth
``l >= 1`` is the output of the ``l``-th block.  A window ``(s, e)`` therefore
maps the representation at depth ``s`` to the one at depth ``e``.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from shallow_tfm.config import TRANSLATOR_FIT_POSITIONS
from shallow_tfm.translators import STATE_KWARGS


def _to_cpu_tensor(value, tuple_index: int = 0):
    """Unwrap a block output into a detached CPU snapshot.

    ``.cpu()`` on a CPU tensor returns the same storage, and TabPFN's sublayers
    mutate their input in place, so an aliased capture would be overwritten
    during the rest of the forward; clone on that path only.
    """
    if isinstance(value, (tuple, list)):
        value = value[tuple_index]
    if not isinstance(value, torch.Tensor):
        return value
    value = value.detach()
    return value.clone() if value.device.type == "cpu" else value.cpu()


def item_axis(shape) -> int:
    """Axis enumerating sequence positions: TabICL captures ``(n_ens, seq, d)``
    and TabFM ``(1, seq, d)`` (axis 1); TabPFN ``(seq, n_features, d)`` and
    TabDPT ``(seq, d)`` (axis 0)."""
    if len(shape) == 3 and shape[0] <= 16 and shape[0] < shape[1]:
        return 1
    return 0


def position_subset(n: int, n_cap) -> torch.Tensor | None:
    """Indices of a fixed random subset of at most ``n_cap`` of ``n`` positions
    (local generator, seed 0), or None when no subsampling is needed."""
    if n_cap is None or n <= n_cap:
        return None
    return torch.randperm(n, generator=torch.Generator().manual_seed(0))[:n_cap]


@torch.no_grad()
def _subsample_positions(tensor, max_positions, cache: dict):
    """Keep the same fixed subset of positions at every depth of one forward
    pass; one subset per sequence length, since a model may run the transformer
    several times with different lengths for one prediction."""
    if max_positions is None or not isinstance(tensor, torch.Tensor):
        return tensor
    shape = tensor.shape[1:] if (tensor.dim() >= 3 and tensor.shape[0] == 1) else tensor.shape
    axis = item_axis(shape) + (1 if len(shape) != tensor.dim() else 0)
    n = int(tensor.shape[axis])
    if n not in cache:
        cache[n] = position_subset(n, max_positions)
    idx = cache[n]
    return tensor if idx is None else tensor.index_select(axis, idx.to(tensor.device))


def _squeeze_batch(t):
    if isinstance(t, torch.Tensor) and t.dim() >= 3 and t.shape[0] == 1:
        return t.squeeze(0)
    return t


def extract_representations(
    model: nn.Module,
    layers: nn.ModuleList,
    predict_fn: Callable,
    X: np.ndarray,
    output_element: int | None = None,
    max_positions: int | None = TRANSLATOR_FIT_POSITIONS,
    per_pass: bool = False,
) -> dict[int, torch.Tensor] | list[dict[int, torch.Tensor]]:
    """Return ``{depth: activations}`` for one forward pass of ``predict_fn(X)``.

    Args:
        output_element: index into tuple-valued block outputs (default: first).
        max_positions: keep only a fixed random subset of this many sequence
            positions at every depth, applied inside the hooks so peak memory
            is bounded by the cap rather than the context size.
        per_pass: when ``predict_fn`` runs the transformer several times for
            one prediction (TabICL and TabDPT above ten classes), return one
            dict per run so a translator can be fitted for each; otherwise only
            the first run is kept and a warning is raised if there were more.
    """
    pos_cache: dict = {}
    # A dynamo-compiled graph bypasses forward hooks, so a warm cache would
    # make the capture silently incomplete.
    torch._dynamo.reset()
    model.eval()
    captured: dict[int, list[torch.Tensor]] = defaultdict(list)
    index = 0 if output_element is None else output_element
    hooks = []

    def pre_hook(module, args, kwargs=None):
        if args:
            state = args[0]
        elif kwargs:
            key = next((k for k in STATE_KWARGS if k in kwargs), None)
            state = kwargs[key] if key else next(iter(kwargs.values()))
        else:
            return
        captured[0].append(_subsample_positions(_to_cpu_tensor(state, index), max_positions, pos_cache))

    def make_post_hook(depth):
        def hook(module, inputs, output):
            captured[depth].append(_subsample_positions(_to_cpu_tensor(output, index), max_positions, pos_cache))
        return hook

    hooks.append(layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True))
    for i in range(len(layers)):
        hooks.append(layers[i].register_forward_hook(make_post_hook(i + 1)))
    try:
        predict_fn(X)
    finally:
        for hook in hooks:
            hook.remove()

    n_passes = max((len(v) for v in captured.values()), default=0)
    if per_pass:
        return [{depth: _squeeze_batch(captured[depth][k]) for depth in sorted(captured)}
                for k in range(n_passes)]
    if n_passes > 1:
        warnings.warn(
            f"extraction captured {n_passes} forward passes through the blocks and kept "
            "only the first: the model runs several passes per prediction (ensemble batches "
            "or chunked inference)",
            stacklevel=2,
        )
    return {depth: _squeeze_batch(captured[depth][0]) for depth in sorted(captured)}
