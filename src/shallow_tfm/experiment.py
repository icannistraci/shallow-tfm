"""Helpers shared by the experiment scripts: CLI argument parsing, split
preparation, results persistence, provenance and window selection."""

from __future__ import annotations

import csv
import fcntl
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from shallow_tfm import RESULTS_DIR
from shallow_tfm.config import ALL_MODELS, MAX_FEATURES, MODEL_CONFIGS, PAPER_DATASETS, SEEDS, TRANSLATOR
from shallow_tfm.data.datasets import load_dataset
from shallow_tfm.models import is_available

SKIP_EVAL_CSV = RESULTS_DIR / "consistent_skip_eval" / "consistent_skip_results.csv"
FAILURES_CSV = RESULTS_DIR / "failures.csv"

# ---------------------------------------------------------------------------
# CLI argument parsing (Fire passes `--models a` as a string, `--models a,b` as
# a string and `--models '["a","b"]'` as a list; these normalise all three)
# ---------------------------------------------------------------------------


def as_list(value, default: list | None = None) -> list:
    if value is None or value == "all":
        return list(default or [])
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, (int, float)):
        return [value]
    return list(value)


def parse_models(models=None, *, available_only: bool = True) -> list[str]:
    """Normalise a ``--models`` argument, dropping models whose package is missing."""
    out = []
    for name in as_list(models, ALL_MODELS):
        if name not in MODEL_CONFIGS:
            print(f"  WARNING: unknown model {name!r}, skipping")
        elif available_only and not is_available(name):
            print(f"  WARNING: package for {name!r} not installed, skipping")
        else:
            out.append(name)
    return out


def parse_datasets(datasets=None) -> list[str]:
    """Normalise a ``--datasets`` argument (default: the eight clinical datasets)."""
    return as_list(datasets, PAPER_DATASETS)


def parse_seeds(seeds=None) -> list[int]:
    """Normalise a ``--seeds`` argument (default: ``SEEDS``)."""
    return [int(s) for s in as_list(seeds, SEEDS)]


#: Symbolic widths, resolved per model from its block count L.
WIDTH_TOKENS = {"half": lambda L: L // 2, "max-1": lambda L: L - 2, "max": lambda L: L - 1}


def parse_widths(spec, default=(3, 4)) -> list:
    """A ``--skip_widths`` spec -> list of ints and tokens (``half``, ``max-1``, ``max``)."""
    if spec is None:
        return list(default)
    items = spec if isinstance(spec, (list, tuple)) else str(spec).split(",")
    out = []
    for w in items:
        w = str(w).strip()
        out.append(w if w in WIDTH_TOKENS else int(w))
    return out


def resolve_widths(widths, n_layers: int) -> list[int]:
    """Tokens -> concrete widths for a model with ``n_layers`` blocks; sorted,
    de-duplicated, restricted to ``1 <= w < n_layers``."""
    out: list[int] = []
    for w in widths:
        v = WIDTH_TOKENS[w](n_layers) if isinstance(w, str) else int(w)
        if 1 <= v < n_layers and v not in out:
            out.append(v)
    return sorted(out)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def capped_subsample(y: np.ndarray, cap: int, rng: np.random.RandomState) -> np.ndarray:
    """Indices of a ``cap``-sized subsample: class-stratified for label vectors,
    plain random for regression targets.  Every class keeps at least one row
    when the cap allows, so the model's probability columns stay aligned with
    the dataset's classes."""
    from sklearn.model_selection import train_test_split

    if cap >= len(y):
        return np.arange(len(y))
    if y.dtype.kind != "f" and len(np.unique(y)) > 1:
        try:
            idx, _ = train_test_split(np.arange(len(y)), train_size=cap, random_state=rng, stratify=y)
        except ValueError:
            idx = rng.choice(len(y), cap, replace=False)
        classes = np.unique(y)
        if cap >= len(classes):
            for c in np.setdiff1d(classes, np.unique(y[idx])):
                add = rng.choice(np.flatnonzero(y == c))
                counts = {k: v for k, v in zip(*np.unique(y[idx], return_counts=True))}
                biggest = max(counts, key=counts.get)
                drop_pos = rng.choice(np.flatnonzero(y[idx] == biggest))
                idx = np.concatenate([np.delete(idx, drop_pos), [add]])
        return idx
    return rng.choice(len(y), cap, replace=False)


def _select_features(X_train: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
    """Indices (sorted, so column order is preserved) of the ``k`` columns with
    the highest ANOVA F-score against the training labels; deterministic."""
    from sklearn.feature_selection import f_classif, f_regression

    regression = y_train.dtype.kind == "f"
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = (f_regression if regression else f_classif)(np.nan_to_num(X_train), y_train)[0]
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.lexsort((np.arange(len(scores)), -scores))
    return np.sort(order[:k])


def prepare_split(dataset: str, seed: int, cfg: dict) -> dict:
    """Load ``dataset`` and shape it the way the foundation models expect.

    Categorical columns are appended to the numerical block, wide datasets are
    reduced to the top ``MAX_FEATURES`` columns by F-score on the training
    split, then the support and calibration sets are capped at the model's
    limits with a class-stratified draw.  Returns X_train/y_train, X_cal/y_cal
    and X_test/y_test.
    """
    data = load_dataset(dataset, seed=seed)
    X_train, X_cal, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_cal, y_test = data["y_train"], data["y_val"], data["y_test"]
    if data["cat_train"] is not None:
        X_train = np.hstack([X_train, data["cat_train"].astype(np.float32)])
        X_cal = np.hstack([X_cal, data["cat_val"].astype(np.float32)])
        X_test = np.hstack([X_test, data["cat_test"].astype(np.float32)])
    if X_train.shape[1] > MAX_FEATURES:
        keep = _select_features(X_train, y_train, MAX_FEATURES)
        X_train, X_cal, X_test = X_train[:, keep], X_cal[:, keep], X_test[:, keep]

    rng = np.random.RandomState(seed)
    max_train = cfg.get("max_train_samples", 3000)
    max_cal = cfg.get("max_cal_samples", 500)
    if len(X_train) > max_train:
        idx = capped_subsample(y_train, max_train, rng)
        X_train, y_train = X_train[idx], y_train[idx]
    if len(X_cal) > max_cal:
        idx = capped_subsample(y_cal, max_cal, rng)
        X_cal, y_cal = X_cal[idx], y_cal[idx]
    return {
        "X_train": X_train, "y_train": y_train,
        "X_cal": X_cal, "y_cal": y_cal,
        "X_test": X_test, "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------


@contextmanager
def csv_lock(csv_path: Path | str):
    """Serialise read-merge-write cycles on a shared results CSV, so two
    processes writing the same file cannot drop each other's rows."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path.with_suffix(csv_path.suffix + ".lock"), "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def write_csv_atomic(df: pd.DataFrame, csv_path: Path | str) -> None:
    """Write via a temp file and ``os.replace``, so a crash mid-write cannot
    corrupt the previous version."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)


def merge_rows(df_new: pd.DataFrame, csv_path: Path | str, key_cols: list[str]) -> pd.DataFrame:
    """Merge new rows into the stored CSV: stored rows whose ``key_cols`` match
    a new row are replaced, every other row is kept."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return df_new
    df_old = pd.read_csv(csv_path)
    for col in key_cols:
        if col not in df_old.columns:
            df_old[col] = np.nan
    rerun = set(df_new[key_cols].apply(tuple, axis=1))
    keep = ~df_old[key_cols].apply(tuple, axis=1).isin(rerun)
    return pd.concat([df_old[keep], df_new], ignore_index=True)


def restrict_report_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the rows whose seed is in ``SEEDS``."""
    if "seed" not in df.columns:
        return df
    return df[pd.to_numeric(df["seed"], errors="coerce").isin(SEEDS)].copy()


def mark_best_by_cal(df: pd.DataFrame, key_cols: list[str], score_col: str,
                     flag_col: str = "is_best_for_width", lower: bool = False) -> pd.DataFrame:
    """Flag the best window per key group by a calibration score.  Rows without
    a calibration score are never flagged."""
    df[flag_col] = False
    for _, group in df.groupby(key_cols):
        valid = group[group[score_col].notna()]
        if not valid.empty:
            df.loc[valid[score_col].idxmin() if lower else valid[score_col].idxmax(), flag_col] = True
    return df


def record_failure(dataset: str, model: str, seed: int, stage: str, exc: Exception) -> None:
    """Append one row (dataset, model, seed, stage, error, timestamp) to
    ``results/failures.csv``, the list of (model, dataset, seed) runs that did
    not complete; without it a failed run would look like one that was never
    started."""
    FAILURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    header = not FAILURES_CSV.exists()
    with open(FAILURES_CSV, "a", newline="") as fh:
        writer = csv.writer(fh)
        if header:
            writer.writerow(["dataset", "model", "seed", "stage", "error", "timestamp"])
        error = " ".join(f"{type(exc).__name__}: {exc}".split())[:500]
        writer.writerow([dataset, model, seed, stage, error,
                         datetime.now(timezone.utc).isoformat(timespec="seconds")])


# ---------------------------------------------------------------------------
# Determinism and provenance
# ---------------------------------------------------------------------------


def set_determinism() -> None:
    """Fix the CUDA nondeterminism settings; call before any model is fitted.
    ``GPU_MEMORY_FRACTION`` caps this process's share of device memory so
    parallel sweeps on one GPU cannot starve each other."""
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    frac = os.environ.get("GPU_MEMORY_FRACTION")
    if frac and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(frac))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


_PROVENANCE: dict | None = None


def provenance(device: str | None = None) -> dict:
    """Columns identifying what produced a result row: git SHA, package
    versions, python, device and a timestamp."""
    global _PROVENANCE
    if _PROVENANCE is None:
        import sklearn
        import torch

        try:
            sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                          cwd=Path(__file__).resolve().parent, text=True,
                                          stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            sha = "unknown"
        _PROVENANCE = {
            "git_sha": sha,
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
            "run_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    row = dict(_PROVENANCE)
    if device is not None:
        import torch

        row["device"] = (torch.cuda.get_device_name(0)
                         if str(device).startswith("cuda") and torch.cuda.is_available() else str(device))
    return row


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------
#
# Per (model, dataset, seed, width), the selected window is the one with the
# smallest forward KL(p_full || p_compressed) on the unlabelled calibration
# rows.  The score is computed from the saved calibration probabilities by
# scripts/sweeps/compute_window_kl.py into window_kl.csv next to the sweep CSV.

SELECTION_COL = "kl_cal_fwd"
WINDOW_KL_CSV = SKIP_EVAL_CSV.parent / "window_kl.csv"
_KL_KEY = ["model", "dataset", "seed", "translator", "skip_start", "skip_end"]


def attach_window_kl(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Add ``kl_cal_fwd`` to sweep rows from ``window_kl.csv``.  A fitted-
    translator window missing from that file means the file is out of date;
    with ``strict``, refuse rather than silently select among fewer windows."""
    if SELECTION_COL in df.columns:
        return df
    kl = pd.read_csv(WINDOW_KL_CSV)[_KL_KEY + [SELECTION_COL]]
    out = df.copy()
    for c in ["seed", "skip_start", "skip_end"]:
        out[c] = out[c].astype(int)
    out = out.merge(kl, on=_KL_KEY, how="left", indicator=True)
    stale = (out["translator"] != "identity") & (out["_merge"] == "left_only")
    out = out.drop(columns="_merge")
    if strict and stale.any():
        gaps = out[stale].groupby(["model", "dataset", "translator"]).size()
        raise RuntimeError("window_kl.csv is out of date; rerun scripts/sweeps/compute_window_kl.py\n" + gaps.to_string())
    out.attrs["stale_models"] = set(out.loc[stale, "model"])
    return out


def select_windows(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One row per ``keys`` group (which must include ``seed``): the window
    with the smallest calibration KL.  Windows without a KL value (diverged
    output) are never selected."""
    df = attach_window_kl(df)
    scored = df[df[SELECTION_COL].notna()]
    return scored.loc[scored.groupby(keys)[SELECTION_COL].idxmin()]


_SWEEP_KL_CACHE: dict = {}


def _sweep_with_kl() -> pd.DataFrame | None:
    if "df" not in _SWEEP_KL_CACHE:
        if not SKIP_EVAL_CSV.exists():
            return None
        df = pd.read_csv(SKIP_EVAL_CSV, low_memory=False,
                         usecols=["model", "dataset", "seed", "width", "translator", "skip_start", "skip_end"])
        df = df[df["translator"] != "identity"]
        _SWEEP_KL_CACHE["df"] = attach_window_kl(restrict_report_seeds(df), strict=False)
    return _SWEEP_KL_CACHE["df"]


def best_skip_start(model: str, dataset: str, width: int, translator: str = TRANSLATOR,
                    seed: int | None = None) -> int | None:
    """Selected window start for (model, dataset, width) from the main sweep,
    for experiments that reuse that window rather than selecting a new one.
    With ``seed``, the start selected for that seed; without it, the start
    with the smallest KL averaged over seeds.  None when the sweep has no scored
    rows."""
    df = _sweep_with_kl()
    if df is None:
        return None
    if model in df.attrs.get("stale_models", ()):
        raise RuntimeError(f"window_kl.csv is out of date for {model}; rerun scripts/sweeps/compute_window_kl.py")
    subset = df[(df["model"] == model) & (df["dataset"] == dataset)
                & (df["width"] == width) & (df["translator"] == translator)]
    if seed is not None:
        subset = subset[subset["seed"] == seed]
    scored = subset[subset[SELECTION_COL].notna()]
    if scored.empty:
        if not subset.empty:
            print(f"best_skip_start: {model}/{dataset}/w{width}/{translator} has {len(subset)} rows "
                  f"but none with {SELECTION_COL}")
        return None
    return int(scored.groupby("skip_start")[SELECTION_COL].mean().idxmin())
