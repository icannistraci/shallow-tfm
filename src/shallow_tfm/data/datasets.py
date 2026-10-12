"""Dataset loading and preprocessing.

Splits are 68/12/20 train/calibration/test.  Numerical features are
standardised on the training rows, categoricals are integer-encoded with a
reserved out-of-vocabulary slot, and missing numericals are imputed at the
training mean (zero after scaling).  The calibration split doubles as the set
the translator is fitted on; it never contributes to a reported test metric.
"""

from __future__ import annotations

import importlib

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#: Row cap applied to any OpenML dataset without its own ``default_max_samples``.
GLOBAL_MAX_SAMPLES = 30_000

#: Classes with fewer rows than this are dropped: below it a class cannot be
#: guaranteed a row in every split, which breaks stratification and the
#: probability-column alignment.
MIN_CLASS_COUNT = 5

#: name -> loader spec.  ``openml_id`` entries go through the generic path;
#: ``loader`` entries name a builder function in this package.
DATASETS: dict[str, dict] = {
    # Clinical, binary
    "diabetes": {"openml_id": 37, "description": "Pima Indians Diabetes (768 x 8)"},
    "heart": {"openml_id": 53, "description": "Heart Disease, Statlog (270 x 13)"},
    "breast_cancer": {"openml_id": 15, "description": "Breast Cancer Wisconsin (699 x 9)"},
    "support2": {"openml_id": 46628, "description": "SUPPORT2 hospital mortality (9105 x 46)"},
    "mimic": {"loader": "shallow_tfm.data.mimic:build_mortality_dataset",
              "description": "MIMIC-III ICU mortality (26538 x 74)"},
    "eicu": {"loader": "shallow_tfm.data.eicu:build_mortality_dataset",
             "description": "eICU-CRD multi-center mortality (30000 x 74)"},
    # Clinical, multi-class
    "cardiotocography": {"openml_id": 1560, "description": "CTG fetal health, 3-class (2126 x 35)"},
    "thyroid": {"openml_id": 40497, "description": "ann-thyroid, 3-class (3772 x 21)"},
    # Regression
    "mimic_los": {"loader": "shallow_tfm.data.mimic:build_los_dataset",
                  "description": "MIMIC-III length-of-stay, log-transformed"},
}


def load_dataset(
    name: str,
    test_size: float = 0.2,
    calibration_size: float = 0.15,
    seed: int = 42,
    max_samples: int | None = None,
) -> dict:
    """Load and preprocess ``name``.

    Returns a dict with X_train/X_val/X_test (scaled numerical block),
    y_train/y_val/y_test, cat_train/cat_val/cat_test (or None),
    cat_cardinalities, n_num_features, n_classes, feature_names, dataset_name.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(DATASETS)}")
    cfg = DATASETS[name]
    if "loader" in cfg:
        module, attr = cfg["loader"].split(":")
        builder = getattr(importlib.import_module(module), attr)
        return builder(test_size=test_size, calibration_size=calibration_size, seed=seed)
    return _load_openml(name, cfg, test_size, calibration_size, seed, max_samples)


def _split_feature_types(X, cat_mask) -> tuple[list[int], list[int]]:
    """Partition columns into numerical and categorical.  OpenML's categorical
    mask is unreliable, so object/category dtypes count as categorical too."""
    if hasattr(X, "dtypes"):
        dtype_is_cat = [dt.kind == "O" or isinstance(dt, pd.CategoricalDtype) for dt in X.dtypes]
    else:
        dtype_is_cat = [False] * len(cat_mask)
    num_cols = [i for i in range(len(cat_mask)) if not cat_mask[i] and not dtype_is_cat[i]]
    cat_cols = [i for i in range(len(cat_mask)) if cat_mask[i] or dtype_is_cat[i]]
    return num_cols, cat_cols


def _encode_categoricals(X_raw, cat_cols, idx_train, idx_val, idx_test):
    """Integer-encode categoricals, fitting only on training rows.  Values
    unseen in training map to an out-of-vocabulary code at ``len(classes)``."""
    if not cat_cols:
        return None, None, None, []
    train_arrs, val_arrs, test_arrs, cardinalities = [], [], [], []
    for col_idx in cat_cols:
        column = np.array(["__missing__" if pd.isna(v) else str(v) for v in X_raw[:, col_idx]])
        encoder = LabelEncoder().fit(column[idx_train])
        mapping = {value: code for code, value in enumerate(encoder.classes_)}
        oov = len(encoder.classes_)

        def encode(rows, _mapping=mapping, _oov=oov):
            return np.array([_mapping.get(v, _oov) for v in rows], dtype=np.int64)

        train_arrs.append(encode(column[idx_train]))
        val_arrs.append(encode(column[idx_val]))
        test_arrs.append(encode(column[idx_test]))
        cardinalities.append(oov + 1)
    return np.stack(train_arrs, axis=1), np.stack(val_arrs, axis=1), np.stack(test_arrs, axis=1), cardinalities


def _stratified_split(indices, test_size, seed, y):
    """``train_test_split``, falling back to unstratified with a warning when a
    class is too rare to stratify on."""
    try:
        return train_test_split(indices, test_size=test_size, random_state=seed, stratify=y)
    except ValueError:
        print(f"  WARNING: stratified split failed (rarest class has "
              f"{np.bincount(y).min()} rows); splitting unstratified")
        return train_test_split(indices, test_size=test_size, random_state=seed)


def _load_openml(name, cfg, test_size, calibration_size, seed, max_samples):
    dataset = openml.datasets.get_dataset(cfg["openml_id"])
    X, y, cat_mask, feature_names = dataset.get_data(target=dataset.default_target_attribute)

    regression = cfg.get("task") == "regression"
    y = np.asarray(y.values if hasattr(y, "values") else y)
    if regression:
        y = y.astype(np.float64)
        n_classes = 1
    else:
        y = LabelEncoder().fit_transform(y)

    cap = max_samples if max_samples is not None else cfg.get("default_max_samples", GLOBAL_MAX_SAMPLES)
    if cap is not None and len(y) > cap:
        subset = np.random.RandomState(seed).choice(len(y), cap, replace=False)
        X = X.iloc[subset] if hasattr(X, "iloc") else X[subset]
        y = y[subset]

    if not regression:
        counts = np.bincount(y)
        if (counts < MIN_CLASS_COUNT).any():
            dropped = np.flatnonzero(counts < MIN_CLASS_COUNT)
            keep = ~np.isin(y, dropped)
            print(f"  WARNING: {name}: dropping {len(dropped)} class(es) with fewer "
                  f"than {MIN_CLASS_COUNT} rows ({keep.sum()}/{len(y)} rows kept)")
            X = X.iloc[keep] if hasattr(X, "iloc") else X[keep]
            y = LabelEncoder().fit_transform(y[keep])
        n_classes = int(y.max()) + 1

    num_cols, cat_cols = _split_feature_types(X, cat_mask)
    X_raw = X.values if hasattr(X, "values") else np.array(X)
    X_num = np.array(X_raw[:, num_cols], dtype=np.float64) if num_cols else np.empty((len(X_raw), 0), dtype=np.float64)

    # Split before fitting any encoder or scaler, so none of them see test rows.
    indices = np.arange(len(y))
    strat = None if regression else y
    idx_trainval, idx_test = _stratified_split(indices, test_size, seed, strat)
    idx_train, idx_val = _stratified_split(idx_trainval, calibration_size, seed,
                                           None if regression else y[idx_trainval])
    cat_train, cat_val, cat_test, cardinalities = _encode_categoricals(X_raw, cat_cols, idx_train, idx_val, idx_test)

    splits = {"X_train": idx_train, "X_val": idx_val, "X_test": idx_test}
    if X_num.shape[1] == 0:
        scaled = {k: X_num[idx].astype(np.float32) for k, idx in splits.items()}
    else:
        # StandardScaler ignores NaNs when fitting, so imputing afterwards places
        # missing cells exactly at the training mean.
        scaler = StandardScaler().fit(X_num[idx_train])
        scaled = {k: np.nan_to_num(scaler.transform(X_num[idx]), nan=0.0).astype(np.float32)
                  for k, idx in splits.items()}

    return {
        **scaled,
        "y_train": y[idx_train],
        "y_val": y[idx_val],
        "y_test": y[idx_test],
        "cat_train": cat_train,
        "cat_val": cat_val,
        "cat_test": cat_test,
        "cat_cardinalities": cardinalities,
        "n_num_features": X_num.shape[1],
        "n_classes": n_classes,
        "feature_names": [feature_names[i] for i in num_cols],
        "dataset_name": name,
    }
