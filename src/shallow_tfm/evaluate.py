"""Evaluation of a compressed model.

The two entry points, ``evaluate_skip`` and ``evaluate_skip_regression``, do
the same thing: score the unchanged model, replace one contiguous window of
blocks with a translator, and score the compressed model on the test split and
on the calibration split.  The calibration scores are what allows the window
to be chosen without looking at test labels.

Classification is scored with nine metrics (ranking, decision and calibration
quality), each reported raw and after post-hoc recalibration fitted on the
calibration split only.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)

from shallow_tfm.skip import skip_context

BASE_CLASSIFICATION_METRICS = [
    "accuracy", "balanced_accuracy", "auroc", "f1",
    "auprc", "brier", "ece", "sensitivity", "specificity",
]
CLASSIFICATION_METRICS = BASE_CLASSIFICATION_METRICS + [f"{m}_recal" for m in BASE_CLASSIFICATION_METRICS]
REGRESSION_METRICS = ["r2", "rmse", "mae"]


def _as_2d_proba(y_proba) -> np.ndarray:
    proba = np.asarray(y_proba, dtype=np.float64)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    return proba


def _macro_specificity(y_true, y_pred, n_classes: int) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    total = cm.sum()
    per_class = []
    for k in range(n_classes):
        tp = cm[k, k]
        fn = cm[k].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total - tp - fn - fp
        if tn + fp > 0:
            per_class.append(tn / (tn + fp))
    return float(np.mean(per_class)) if per_class else float("nan")


def _expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    """Top-label expected calibration error with equal-width confidence bins."""
    proba = _as_2d_proba(y_proba)
    confidence = proba.max(axis=1)
    correct = (np.argmax(proba, axis=1) == np.asarray(y_true)).astype(float)
    n = len(correct)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        count = mask.sum()
        if count:
            ece += (count / n) * abs(confidence[mask].mean() - correct[mask].mean())
    return float(ece)


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """The nine classification metrics from one set of predictions.

    Ranking metrics are macro one-vs-rest for multi-class tasks; sensitivity
    and specificity are the class-1 / class-0 recalls for binary tasks and
    macro averages otherwise.  A metric undefined on a degenerate split is NaN.
    """
    y_true = np.asarray(y_true)
    proba = _as_2d_proba(y_proba)
    n_classes = proba.shape[1]
    if y_true.size and int(np.max(y_true)) >= n_classes:
        raise RuntimeError(
            f"proba has {n_classes} columns but labels reach {int(np.max(y_true))}: "
            "a class in the evaluation split is missing from the model's support set")

    def guarded(fn) -> float:
        try:
            return float(fn())
        except ValueError:
            return float("nan")

    if n_classes == 2:
        pos = proba[:, 1]
        auroc = guarded(lambda: roc_auc_score(y_true, pos))
        auprc = guarded(lambda: average_precision_score(y_true, pos))
        brier = guarded(lambda: brier_score_loss(y_true, pos))
        sensitivity = guarded(lambda: recall_score(y_true, y_pred, pos_label=1, zero_division=np.nan))
        specificity = guarded(lambda: recall_score(y_true, y_pred, pos_label=0, zero_division=np.nan))
    else:
        onehot = np.eye(n_classes)[y_true.astype(int)]
        auroc = guarded(lambda: roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
        auprc = guarded(lambda: average_precision_score(onehot, proba, average="macro"))
        # Sum-over-classes Brier (range 0-2); at K=2 this is twice the binary
        # score above; pooling binary and multi-class rows requires halving it.
        brier = guarded(lambda: np.mean(np.sum((proba - onehot) ** 2, axis=1)))
        sensitivity = guarded(lambda: recall_score(y_true, y_pred, average="macro", zero_division=np.nan))
        specificity = guarded(lambda: _macro_specificity(y_true, y_pred, n_classes))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auroc": auroc,
        "f1": f1_score(y_true, y_pred, average="macro"),
        "auprc": auprc,
        "brier": brier,
        "ece": guarded(lambda: _expected_calibration_error(y_true, proba)),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


class Recalibrator:
    """Post-hoc recalibration fitted on the calibration split.

    Candidates in order of capacity: vector scaling (``softmax(w * log p + b)``,
    one ``(w, b)`` per class), then temperature scaling (a single scalar).  A
    candidate is accepted only if its 5-fold cross-validated NLL on the
    calibration split beats the raw probabilities; otherwise the identity is
    used.  Fits are deterministic full-batch LBFGS.
    """

    MIN_ROWS_PER_CLASS = 5
    CV_FOLDS = 5

    def __init__(self, w: np.ndarray | None = None, b: np.ndarray | None = None, kind: str = "identity"):
        self.w, self.b, self.kind = w, b, kind

    @property
    def is_identity(self) -> bool:
        return self.w is None

    @staticmethod
    def _fit_params(z: torch.Tensor, t: torch.Tensor, n_classes: int, kind: str):
        if kind == "vector":
            w = torch.ones(n_classes, dtype=torch.float64, requires_grad=True)
            b = torch.zeros(n_classes, dtype=torch.float64, requires_grad=True)
            params = [w, b]
        else:
            w = torch.ones(1, dtype=torch.float64, requires_grad=True)
            b = torch.zeros(n_classes, dtype=torch.float64)
            params = [w]
        opt = torch.optim.LBFGS(params, max_iter=200, line_search_fn="strong_wolfe")
        nll = torch.nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad()
            loss = nll(z * w + b, t)
            loss.backward()
            return loss

        opt.step(closure)
        if not (torch.isfinite(w).all() and torch.isfinite(b).all()):
            return None
        w_out = w.detach().expand(n_classes).clone() if kind == "temperature" else w.detach()
        return w_out.numpy(), b.detach().numpy()

    @staticmethod
    def _nll(z: np.ndarray, y: np.ndarray, w=None, b=None) -> float:
        logits = z if w is None else z * w + b
        logits = logits - logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(y)), y].mean())

    @classmethod
    def _cv_nll(cls, z: np.ndarray, y: np.ndarray, n_classes: int, kind: str) -> float | None:
        from sklearn.model_selection import StratifiedKFold
        counts = np.bincount(y, minlength=n_classes)
        n_splits = min(cls.CV_FOLDS, int(counts[counts > 0].min()))
        if n_splits < 2:
            return None
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        total, n = 0.0, 0
        for tr, va in skf.split(z, y):
            fit = cls._fit_params(torch.tensor(z[tr]), torch.tensor(y[tr]), n_classes, kind)
            if fit is None:
                return None
            total += cls._nll(z[va], y[va], *fit) * len(va)
            n += len(va)
        return total / n

    @classmethod
    def fit(cls, proba_cal, y_cal) -> "Recalibrator":
        proba = _as_2d_proba(proba_cal)
        y = np.asarray(y_cal).astype(int)
        n_classes = proba.shape[1]
        counts = np.bincount(y, minlength=n_classes)
        if (counts > 0).sum() < 2 or counts[counts > 0].min() < cls.MIN_ROWS_PER_CLASS:
            return cls()
        z = np.log(np.clip(proba, 1e-12, None))
        raw_nll = cls._nll(z, y)
        for kind in ("vector", "temperature"):
            cv = cls._cv_nll(z, y, n_classes, kind)
            if cv is None or cv >= raw_nll:
                continue
            fit = cls._fit_params(torch.tensor(z), torch.tensor(y), n_classes, kind)
            if fit is not None:
                return cls(*fit, kind=kind)
        return cls()

    def transform(self, y_proba) -> np.ndarray:
        proba = _as_2d_proba(y_proba)
        if self.is_identity:
            return proba
        logits = np.log(np.clip(proba, 1e-12, None)) * self.w + self.b
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


def _classification_metrics(y_true, y_proba, recal: Recalibrator | None) -> dict[str, float]:
    """Raw metrics, plus the ``_recal`` set when a recalibrator is supplied."""
    proba = _as_2d_proba(y_proba)
    metrics = compute_metrics(y_true, np.argmax(proba, axis=1), proba)
    if recal is not None:
        proba_recal = recal.transform(proba)
        recal_metrics = compute_metrics(y_true, np.argmax(proba_recal, axis=1), proba_recal)
        metrics.update({f"{k}_recal": v for k, v in recal_metrics.items()})
    return metrics


def compute_regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def baseline_metrics(predict_fn: Callable, X_test, y_test, regression: bool = False,
                     return_probas: bool = False) -> dict:
    """Score the unchanged model, keyed with the ``_full`` suffix.

    Call this right after fitting and before any prediction on the calibration
    rows: every prediction call changes the model's internal state (the random
    generator, or TabICL's ensemble generator), so scoring the full model later
    would make its numbers depend on the calibration set.
    With ``return_probas`` the result carries a ``_probas`` key with the raw
    test probabilities; pop it before serialising.
    """
    if regression:
        return {f"{k}_full": v for k, v in compute_regression_metrics(y_test, predict_fn(X_test)).items()}
    proba = predict_fn(X_test)
    result = {f"{k}_full": v for k, v in _classification_metrics(y_test, proba, None).items()}
    if return_probas:
        result["_probas"] = {"proba_test_full": proba}
    return result


def full_recalibration(predict_fn: Callable, y_test, proba_test, X_cal, y_cal) -> tuple[dict, Any]:
    """Recalibrated full-model metrics, keyed ``*_recal_full``, plus the
    calibration probabilities.

    Call this after every window has been evaluated: predicting on the
    calibration rows changes the model's internal state, so doing it earlier
    would change every window's numbers.
    """
    proba_cal = predict_fn(X_cal)
    recal = Recalibrator.fit(proba_cal, y_cal)
    proba_recal = recal.transform(proba_test)
    metrics = compute_metrics(y_test, np.argmax(proba_recal, axis=1), proba_recal)
    return {f"{k}_recal_full": v for k, v in metrics.items()}, proba_cal


def apply_full_recalibration(results: list[dict], recal_full: dict) -> None:
    """Merge ``*_recal_full`` metrics into evaluated rows and fill the
    ``delta_*_recal`` columns."""
    for row in results:
        row.update(recal_full)
        for metric in BASE_CLASSIFICATION_METRICS:
            skip_key, full_key = f"{metric}_recal_skip", f"{metric}_recal_full"
            if skip_key in row and full_key in row:
                row[f"delta_{metric}_recal"] = row[skip_key] - row[full_key]


def _assemble(info, baseline, skip, cal_skip, metric_names) -> dict[str, Any]:
    result = {
        "skip_start": info["skip_start"],
        "skip_end": info["skip_end"],
        "n_skipped": info["skip_end"] - info["skip_start"],
        "epsilon": info["epsilon"],
        "ridge_lambda": info.get("ridge_lambda"),
        "n_fit_positions": info.get("n_fit_positions"),
        **{k: v for k, v in baseline.items() if k != "_probas"},
        **{f"{k}_skip": v for k, v in skip.items()},
        **{f"{k}_cal_skip": v for k, v in (cal_skip or {}).items()},
    }
    for metric in metric_names:
        if metric in skip and f"{metric}_full" in baseline:
            result[f"delta_{metric}"] = skip[metric] - baseline[f"{metric}_full"]
    return result


def evaluate_skip(
    predict_fn: Callable,
    layers: nn.ModuleList,
    X_test: np.ndarray,
    y_test: np.ndarray,
    skip_start: int,
    skip_end: int,
    representations: dict | list | None = None,
    translator: str = "ridge",
    baseline: dict[str, float] | None = None,
    device=None,
    return_tuple_size: int = 0,
    per_feature: bool = False,
    X_cal: np.ndarray | None = None,
    y_cal: np.ndarray | None = None,
    return_probas: bool = False,
    fit_positions="default",
    pad_tuple_none: bool = False,
) -> dict[str, Any]:
    """Classification: evaluate one window and return full / skip / delta metrics.

    With a calibration split, a recalibrator is fitted on it per window and the
    ``_recal`` metrics are reported alongside the raw ones; the ``_cal_skip``
    numbers are in-sample for that recalibrator and serve window selection only.
    """
    if baseline is None:
        baseline = baseline_metrics(predict_fn, X_test, y_test)
    ctx_kwargs = {} if fit_positions == "default" else {"fit_positions": fit_positions}
    proba_cal = None
    with skip_context(layers, skip_start, skip_end, representations, translator, device,
                      return_tuple_size=return_tuple_size, per_feature=per_feature,
                      pad_tuple_none=pad_tuple_none, **ctx_kwargs) as info:
        proba_skip = predict_fn(X_test)
        if X_cal is not None and y_cal is not None:
            proba_cal = predict_fn(X_cal)

    recal = cal_skip = None
    if proba_cal is not None:
        recal = Recalibrator.fit(proba_cal, y_cal)
        cal_skip = _classification_metrics(y_cal, proba_cal, recal)
    skip = _classification_metrics(y_test, proba_skip, recal)
    result = _assemble(info, baseline, skip, cal_skip, CLASSIFICATION_METRICS)
    result["translator"] = translator
    if return_probas:
        result["_probas"] = {"proba_test_skip": proba_skip, "proba_cal_skip": proba_cal}
    return result


def evaluate_skip_regression(
    predict_fn: Callable,
    layers: nn.ModuleList,
    X_test: np.ndarray,
    y_test: np.ndarray,
    skip_start: int,
    skip_end: int,
    representations: dict | list | None = None,
    translator: str = "ridge",
    baseline: dict[str, float] | None = None,
    device=None,
    return_tuple_size: int = 0,
    per_feature: bool = False,
    X_cal: np.ndarray | None = None,
    y_cal: np.ndarray | None = None,
    fit_positions="default",
    pad_tuple_none: bool = False,
) -> dict[str, Any]:
    """Regression counterpart of ``evaluate_skip`` (R^2 / RMSE / MAE)."""
    if baseline is None:
        baseline = baseline_metrics(predict_fn, X_test, y_test, regression=True)
    ctx_kwargs = {} if fit_positions == "default" else {"fit_positions": fit_positions}
    cal_skip = None
    with skip_context(layers, skip_start, skip_end, representations, translator, device,
                      return_tuple_size=return_tuple_size, per_feature=per_feature,
                      pad_tuple_none=pad_tuple_none, **ctx_kwargs) as info:
        preds_skip = predict_fn(X_test)
        if X_cal is not None and y_cal is not None:
            cal_skip = compute_regression_metrics(y_cal, predict_fn(X_cal))
    skip = compute_regression_metrics(y_test, preds_skip)
    result = _assemble(info, baseline, skip, cal_skip, REGRESSION_METRICS)
    result["translator"] = translator
    return result


def kl_forward(p_full: np.ndarray, p_skip: np.ndarray, eps: float = 1e-12) -> float:
    """Mean over rows of ``KL(p_full || p_skip)``: the label-free selection score."""
    p, q = np.clip(p_full, eps, None), np.clip(p_skip, eps, None)
    return float((p * np.log(p / q)).sum(1).mean())


def select_by_kl(candidates: list[dict], proba_cal_full: np.ndarray) -> dict | None:
    """Pick the window with the smallest calibration KL, inside a run rather than from stored files.

    ``candidates`` are ``evaluate_skip(..., X_cal=..., return_probas=True)``
    results for the windows of one (model, dataset, seed, width);
    ``proba_cal_full`` is the full model's calibration prediction, taken after
    the windows as ``full_recalibration`` does.  Each candidate gets
    ``kl_cal_fwd``; windows with non-finite output are never selected.
    """
    best = None
    for r in candidates:
        q = r["_probas"]["proba_cal_skip"]
        kl = kl_forward(proba_cal_full, q) if np.isfinite(q).all() else np.nan
        r["kl_cal_fwd"] = kl
        if np.isfinite(kl) and (best is None or kl < best["kl_cal_fwd"]):
            best = r
    return best
