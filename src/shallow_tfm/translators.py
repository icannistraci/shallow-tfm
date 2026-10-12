"""Closed-form affine translators that stand in for a window of transformer blocks.

A translator is fitted by least squares on calibration activations and swapped
into the block list in place of a contiguous stack of blocks.  Two variants
cover the two tokenisations: a single shared map ``state @ W + b`` for row-token
models, and one map per feature index for dual-axis models (TabPFNv2, LimiX),
whose attention runs along both items and features, so a shared map would mix
the two axes and break the per-feature correspondence between source and target.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

#: Keyword names under which models pass the hidden state to a block.
STATE_KWARGS = ("q", "x", "src", "input", "state", "hidden_states", "support")


def first_tensor(*args, **kwargs) -> torch.Tensor:
    """Pull the hidden state out of a block's call signature."""
    if args:
        state = args[0]
        return state[0] if isinstance(state, (tuple, list)) else state  # tabpfn>=8 wraps the state in a list
    for key in STATE_KWARGS:
        if isinstance(kwargs.get(key), torch.Tensor):
            return kwargs[key]
    for value in kwargs.values():
        if isinstance(value, torch.Tensor):
            return value
    raise ValueError(f"No tensor in args={args}, kwargs={list(kwargs)}")


class _TranslatorBase(nn.Module):
    """Common plumbing for block replacements.

    ``return_tuple_size`` mirrors blocks that return a tuple: the translated
    state is returned alongside the original trailing elements so the next
    block still receives the shape it expects.  ``pad_tuple_none`` covers
    blocks that are called with one positional argument but whose caller
    unpacks a fixed-size tuple (LimiX).  TabPFN's memory estimator reads
    ``save_peak_mem_factor``, so every replacement carries the attribute.
    """

    def __init__(self, return_tuple_size: int = 0):
        super().__init__()
        self.return_tuple_size = return_tuple_size
        self.save_peak_mem_factor = None
        self.pad_tuple_none = False

    def _wrap(self, translated, args):
        if self.return_tuple_size >= 2 and self.pad_tuple_none:
            extra = tuple(args[1:self.return_tuple_size])
            extra += (None,) * (self.return_tuple_size - 1 - len(extra))
            return (translated,) + extra
        if self.return_tuple_size >= 2 and len(args) >= 2:
            return (translated,) + args[1:self.return_tuple_size]
        return translated


class LinearTranslator(_TranslatorBase):
    """A single shared affine map: ``state @ W + b``."""

    def __init__(self, W: torch.Tensor, b: torch.Tensor, return_tuple_size: int = 0):
        super().__init__(return_tuple_size)
        self.register_buffer("W", W)
        self.register_buffer("b", b)

    def forward(self, *args, **kwargs):
        state = first_tensor(*args, **kwargs)
        return self._wrap(state @ self.W.to(state.dtype) + self.b.to(state.dtype), args)


class PerFeatureTranslator(_TranslatorBase):
    """One affine map per feature index: ``y[..., f, :] = x[..., f, :] @ W[f] + b[f]``."""

    def __init__(self, W: torch.Tensor, b: torch.Tensor, return_tuple_size: int = 0):
        super().__init__(return_tuple_size)
        self.register_buffer("W", W)  # (n_features, d, d)
        self.register_buffer("b", b)  # (n_features, d)

    def forward(self, *args, **kwargs):
        state = first_tensor(*args, **kwargs)
        W, b = self.W.to(state.dtype), self.b.to(state.dtype)
        return self._wrap(torch.einsum("...fd,fde->...fe", state, W) + b, args)


class IdentityTranslator(_TranslatorBase):
    """Pass-through: the naive block-dropping baseline, and the filler for the
    blocks after the first one inside a skipped window."""

    def forward(self, *args, **kwargs):
        return self._wrap(first_tensor(*args, **kwargs), args)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

#: Ridge grid and holdout settings for the label-free lambda selection.
RIDGE_LAMBDA_GRID = (0.0, 1e-3, 1e-2, 1e-1)
RIDGE_HOLDOUT_FRAC = 0.25
RIDGE_HOLDOUT_CAP = 1000


def _flatten(H: torch.Tensor) -> torch.Tensor:
    return H.reshape(-1, H.shape[-1]).float()


def _residual(src: torch.Tensor, dst: torch.Tensor, W, b) -> float:
    """Relative Frobenius residual ``||dst - (src W + b)||_F / ||dst||_F``."""
    err = dst - (src @ W + b)
    return (torch.norm(err, p="fro") / (torch.norm(dst, p="fro") + 1e-8)).item()


def fit_linear(H_s: torch.Tensor, H_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Plain least squares ``H_e ~= H_s @ W + b``; returns ``(W, b, epsilon)``.

    The ``gelsd`` driver gives the minimum-norm solution, which keeps
    under-determined systems (fewer positions than ``d_model``) finite.
    """
    d = H_s.shape[-1]
    src, dst = _flatten(H_s), _flatten(H_e)
    augmented = torch.cat([src, torch.ones(src.shape[0], 1, device=src.device)], dim=1)
    solution, *_ = torch.linalg.lstsq(augmented, dst, driver="gelsd")
    W, b = solution[:d, :], solution[d, :]
    return W, b, _residual(src, dst, W, b)


def fit_ridge(H_s: torch.Tensor, H_e: torch.Tensor, lam_rel: float):
    """Least squares with a ridge penalty on ``W`` only; ``lam_rel=0`` is ``fit_linear``.

    The penalty is scaled by the mean diagonal of the Gram matrix, so
    ``lam_rel`` is relative to the activation scale; the intercept is unpenalised.
    """
    d = H_s.shape[-1]
    src, dst = _flatten(H_s), _flatten(H_e)
    A = torch.cat([src, torch.ones(src.shape[0], 1, device=src.device)], dim=1)
    if lam_rel == 0.0:
        solution, *_ = torch.linalg.lstsq(A, dst, driver="gelsd")
    else:
        gram = A.T @ A
        lam = lam_rel * torch.diagonal(gram)[:d].mean()
        reg = torch.eye(d + 1, device=A.device) * lam
        reg[d, d] = 0.0
        solution = torch.linalg.solve(gram + reg, A.T @ dst)
    return solution[:d, :], solution[d, :]


def _holdout_split(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed held-out/kept position split (a local random generator, so the
    models' global one is untouched)."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    n_hold = max(1, min(int(RIDGE_HOLDOUT_FRAC * n), RIDGE_HOLDOUT_CAP))
    return perm[:n_hold], perm[n_hold:]


def fit_linear_auto(
    H_s: torch.Tensor, H_e: torch.Tensor, lam_grid=RIDGE_LAMBDA_GRID,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Ridge fit with label-free lambda selection; returns ``(W, b, epsilon, lam)``.

    Lambda is chosen by the residual on positions held out of the fit, then the
    map is refitted on all positions at the chosen lambda.
    """
    src, dst = _flatten(H_s), _flatten(H_e)
    hold, keep = _holdout_split(src.shape[0])
    best_lam, best_eps = lam_grid[0], float("inf")
    for lam in lam_grid:
        W, b = fit_ridge(src[keep], dst[keep], lam)
        eps_out = _residual(src[hold], dst[hold], W, b)
        if eps_out < best_eps:
            best_lam, best_eps = lam, eps_out
    W, b = fit_ridge(src, dst, best_lam)
    return W, b, _residual(src, dst, W, b), best_lam


def _per_feature_stack(H: torch.Tensor) -> torch.Tensor:
    """``(..., n_features, d)`` -> ``(n_features, N, d)`` float32."""
    d = H.shape[-1]
    return H.reshape(-1, H.shape[-2], d).transpose(0, 1).float().contiguous()


def _solve_device(device=None) -> torch.device:
    """Where the batched ridge solves run: the caller's device when given, else
    CUDA when present.  ``gelsd`` has no CUDA backend, so lam=0 stays on CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _batched_fit(src: torch.Tensor, dst: torch.Tensor, lam_rel: float, device=None):
    """``fit_ridge`` over a ``(n_features, N, d)`` stack; returns ``(W, b)`` on CPU."""
    d = src.shape[-1]
    A = torch.cat([src, torch.ones(*src.shape[:-1], 1, device=src.device)], dim=-1)
    if lam_rel == 0.0:
        solution = torch.linalg.lstsq(A, dst, driver="gelsd").solution
    else:
        dev = _solve_device(device)
        A_d, dst_d = A.to(dev), dst.to(dev)
        gram = A_d.mT @ A_d
        lam = lam_rel * torch.diagonal(gram, dim1=-2, dim2=-1)[..., :d].mean(-1)
        eye = torch.eye(d + 1, device=dev)
        eye[d, d] = 0.0
        lhs, rhs = gram + lam[:, None, None] * eye, A_d.mT @ dst_d
        if dev.type == "cpu":
            # MKL's batched LU returns wrong pivots on CPU once the OpenMP pool
            # has been re-initialised by torch.set_num_threads(); per-matrix
            # solves take the plain LAPACK path and cost milliseconds here.
            solution = torch.stack([torch.linalg.solve(lhs[i], rhs[i]) for i in range(lhs.shape[0])])
        else:
            solution = torch.linalg.solve(lhs, rhs).cpu()
    return solution[:, :d, :], solution[:, d, :]


def _batched_residual(src, dst, W, b) -> torch.Tensor:
    err = dst - (src @ W + b[:, None, :])
    return torch.linalg.matrix_norm(err) / (torch.linalg.matrix_norm(dst) + 1e-8)


def fit_per_feature(H_s: torch.Tensor, H_e: torch.Tensor,
                    device=None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Plain least squares, one map per feature index, in a single batched solve.

    Returns ``(W, b, info)`` with ``W`` of shape ``(n_features, d, d)``, ``b`` of
    shape ``(n_features, d)`` and the mean and per-feature residuals in ``info``.
    """
    src, dst = _per_feature_stack(H_s), _per_feature_stack(H_e)
    W, b = _batched_fit(src, dst, 0.0, device)
    epsilons = _batched_residual(src, dst, W, b).tolist()
    return W, b, {"epsilon_mean": float(np.mean(epsilons)), "epsilon_per_feature": epsilons}


def fit_per_feature_auto(H_s: torch.Tensor, H_e: torch.Tensor,
                         device=None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Per-feature counterpart of ``fit_linear_auto``; lambda is chosen per feature."""
    src, dst = _per_feature_stack(H_s), _per_feature_stack(H_e)
    n_features, n, d = src.shape
    hold, keep = _holdout_split(n)

    best_eps = torch.full((n_features,), float("inf"))
    best_idx = torch.zeros(n_features, dtype=torch.long)
    for i, lam in enumerate(RIDGE_LAMBDA_GRID):
        W_i, b_i = _batched_fit(src[:, keep], dst[:, keep], lam, device)
        eps_out = _batched_residual(src[:, hold], dst[:, hold], W_i, b_i)
        better = eps_out < best_eps  # strict, so the smallest lambda wins ties
        best_idx[better] = i
        best_eps[better] = eps_out[better]

    W = torch.empty(n_features, d, d)
    b = torch.empty(n_features, d)
    for i, lam in enumerate(RIDGE_LAMBDA_GRID):
        group = best_idx == i
        if group.any():
            W[group], b[group] = _batched_fit(src[group], dst[group], lam, device)

    epsilons = _batched_residual(src, dst, W, b).tolist()
    lams = [RIDGE_LAMBDA_GRID[i] for i in best_idx.tolist()]
    info = {
        "epsilon_mean": float(np.mean(epsilons)),
        "epsilon_per_feature": epsilons,
        "ridge_lambda": float(np.median(lams)),
        "ridge_lambda_per_feature": lams,
    }
    return W, b, info
