"""Temporarily replace a contiguous block window with a fitted translator."""

from __future__ import annotations

from contextlib import contextmanager

import torch.nn as nn

from shallow_tfm.config import TRANSLATOR_FIT_POSITIONS
from shallow_tfm.extraction import item_axis, position_subset
from shallow_tfm.translators import (
    IdentityTranslator,
    LinearTranslator,
    PerFeatureTranslator,
    fit_linear,
    fit_linear_auto,
    fit_per_feature,
    fit_per_feature_auto,
)

#: Translator names that fit a map; anything else degrades to identity.
FITTED_TRANSLATORS = ("linear", "ridge")


def _fit(source, target, translator, per_feature, device=None):
    """Fit one map; returns ``(W, b, epsilon, ridge_lambda | None)``."""
    if translator == "ridge":
        if per_feature:
            W, b, info = fit_per_feature_auto(source, target, device=device)
            return W, b, info["epsilon_mean"], info["ridge_lambda"]
        return fit_linear_auto(source, target)
    if per_feature:
        W, b, info = fit_per_feature(source, target, device=device)
        return W, b, info["epsilon_mean"], None
    W, b, epsilon = fit_linear(source, target)
    return W, b, epsilon, None


def _cap_positions(source, target, n_cap):
    """Keep the same random subset of at most ``n_cap`` positions in source and
    target; returns ``(source, target, n_positions_used)``."""
    axis = item_axis(source.shape)
    n = source.shape[axis]
    idx = position_subset(n, n_cap)
    if idx is None:
        return source, target, int(n)
    idx = idx.to(source.device)
    return source.index_select(axis, idx), target.index_select(axis, idx), int(n_cap)


def build_translator(representations, skip_start, skip_end, translator, per_feature,
                     return_tuple_size, device=None, fit_positions=TRANSLATOR_FIT_POSITIONS):
    """Fit the replacement for blocks ``[skip_start, skip_end)``; returns
    ``(module, epsilon, ridge_lambda, n_fit_positions)``."""
    if translator not in FITTED_TRANSLATORS or representations is None:
        return IdentityTranslator(return_tuple_size), None, None, None
    source, target = representations[skip_start], representations[skip_end]
    source, target, n_fit = _cap_positions(source, target, fit_positions)
    W, b, epsilon, lam = _fit(source, target, translator, per_feature, device)
    cls = PerFeatureTranslator if per_feature else LinearTranslator
    return cls(W, b, return_tuple_size), epsilon, lam, n_fit


class PerPassTranslator(nn.Module):
    """One fitted translator per run of the transformer, for models that run it
    several times for one prediction: the k-th map serves the k-th run."""

    def __init__(self, maps):
        super().__init__()
        self.maps = nn.ModuleList(maps)
        self.k = 0

    def forward(self, *args, **kwargs):
        out = self.maps[self.k % len(self.maps)](*args, **kwargs)
        self.k += 1
        return out


@contextmanager
def skip_context(
    layers: nn.ModuleList,
    skip_start: int,
    skip_end: int,
    representations: dict | list | None = None,
    translator: str = "ridge",
    device=None,
    return_tuple_size: int = 0,
    per_feature: bool = False,
    fit_positions=TRANSLATOR_FIT_POSITIONS,
    pad_tuple_none: bool = False,
):
    """Replace ``layers[skip_start:skip_end]`` with a translator for the duration
    of the block, then restore the original modules.

    The first position of the window carries the fitted map from depth
    ``skip_start`` to depth ``skip_end``; the remaining positions become
    pass-throughs, so the window collapses into one affine step.  With
    ``translator="identity"`` every position is a pass-through (naive dropping).
    A list of representations (one dict per transformer run) yields one map per run.

    Yields a dict with the residual ``epsilon``, the window bounds, the chosen
    ridge lambda and the number of positions the map was fitted on.
    """
    original = [layers[i] for i in range(skip_start, skip_end)]
    if device is None:
        try:
            device = next(layers[skip_start].parameters()).device
        except StopIteration:
            device = "cpu"

    reps = representations if isinstance(representations, list) else [representations]
    built = [build_translator(r, skip_start, skip_end, translator, per_feature,
                              return_tuple_size, device, fit_positions) for r in reps]
    maps = [m for m, *_ in built]
    for m in maps:
        m.pad_tuple_none = pad_tuple_none
    head = PerPassTranslator(maps) if len(maps) > 1 else maps[0]
    head.pad_tuple_none = pad_tuple_none
    _, epsilon, ridge_lambda, n_fit_positions = built[0]

    layers[skip_start] = head.to(device)
    for i in range(skip_start + 1, skip_end):
        filler = IdentityTranslator(return_tuple_size).to(device)
        filler.pad_tuple_none = pad_tuple_none
        layers[i] = filler
    try:
        yield {"epsilon": epsilon, "skip_start": skip_start, "skip_end": skip_end,
               "ridge_lambda": ridge_lambda, "n_fit_positions": n_fit_positions}
    finally:
        for i, module in zip(range(skip_start, skip_end), original):
            layers[i] = module
