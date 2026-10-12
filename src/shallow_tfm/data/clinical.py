"""Shared pieces of the two PhysioNet cohort loaders: patient-grouped splits,
standardisation and the result-dict layout.

MIMIC rows are admissions and eICU rows are hospital stays, so the same
patient can contribute several rows; a row-level split would leak patients
across train/calibration/test.  Both cohorts are therefore split at the
patient level.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from shallow_tfm import DATA_DIR

PHYSIONET_DIR = DATA_DIR / "physionet"
CACHE_DIR = DATA_DIR / "cache"

AGG_FUNCS = ["mean", "std", "min", "max"]


def grouped_three_way_split(pids: np.ndarray, test_size: float, calibration_size: float,
                            seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row indices ``(idx_train, idx_val, idx_test)`` with patient-disjoint splits:
    first peel off ``test_size``, then ``calibration_size`` of the remainder."""
    idx = np.arange(len(pids))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_pos, test_pos = next(gss.split(idx, groups=pids))
    idx_trainval, idx_test = idx[trainval_pos], idx[test_pos]
    gss = GroupShuffleSplit(n_splits=1, test_size=calibration_size, random_state=seed)
    train_pos, val_pos = next(gss.split(idx_trainval, groups=pids[idx_trainval]))
    return idx_trainval[train_pos], idx_trainval[val_pos], idx_test


def standardise_splits(X_raw: np.ndarray, idx_train, idx_val, idx_test):
    """Standardise on the observed training cells and impute missing cells at
    the training mean; returns float32 ``(X_train, X_val, X_test)``."""
    scaler = StandardScaler().fit(X_raw[idx_train])
    return tuple(np.nan_to_num(scaler.transform(X_raw[idx]), nan=0.0).astype(np.float32)
                 for idx in (idx_train, idx_val, idx_test))


def assemble(name: str, X_raw: np.ndarray, y: np.ndarray, pids: np.ndarray, feature_names: list[str],
             n_classes: int, test_size: float, calibration_size: float, seed: int) -> dict:
    """Split, standardise and pack a cohort into the ``load_dataset`` layout."""
    idx_train, idx_val, idx_test = grouped_three_way_split(pids, test_size, calibration_size, seed)
    X_train, X_val, X_test = standardise_splits(X_raw, idx_train, idx_val, idx_test)
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y[idx_train], "y_val": y[idx_val], "y_test": y[idx_test],
        "cat_train": None, "cat_val": None, "cat_test": None,
        "cat_cardinalities": [],
        "n_num_features": X_raw.shape[1],
        "n_classes": n_classes,
        "feature_names": feature_names,
        "dataset_name": name,
    }


def aggregate_long(df, id_col: str, prefix: str):
    """Per-stay mean/std/min/max of a long (id, feature, value) frame, as
    columns ``{prefix}_{feature}_{stat}``."""
    agg = df.groupby([id_col, "feature"])["value"].agg(AGG_FUNCS).unstack(level="feature")
    agg.columns = [f"{prefix}_{feat}_{stat}" for stat, feat in agg.columns]
    return agg.reset_index()
