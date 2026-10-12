#!/usr/bin/env python3
"""Download and check every suite dataset with the loader the sweeps use.

Goes through every dataset in data/dataset_manifest.csv, loads it with the
same checks and splits the sweeps use, and records the resulting shapes and
any failure, so an OpenML dataset that cannot be downloaded or a class with
fewer than ``MIN_CLASS_COUNT`` rows shows up before any model is run.  Needs
only the CPU and the network; every result is saved as soon as it is checked,
so a run interrupted by the network resumes without downloading again.

Writes: results/suite_prefetch/suite_prefetch.csv  (one row per dataset)
        results/suite_prefetch/failures.csv        (the rows that failed or
                                                    warned, with the exception text)

Usage:
    python scripts/sweeps/prefetch_suites.py
    python scripts/sweeps/prefetch_suites.py --task classification
    python scripts/sweeps/prefetch_suites.py --source cc18 --limit 5
    python scripts/sweeps/prefetch_suites.py --force
"""

from __future__ import annotations

import traceback
import warnings

import fire
import numpy as np
import pandas as pd

from shallow_tfm import RESULTS_DIR
from shallow_tfm.config import MAX_FEATURES
from shallow_tfm.data.datasets import MIN_CLASS_COUNT, load_dataset
from shallow_tfm.data.suites import register_suite_datasets, suite_manifest
from shallow_tfm.experiment import merge_rows, provenance, write_csv_atomic

warnings.filterwarnings("ignore", category=UserWarning)

OUT_DIR = RESULTS_DIR / "suite_prefetch"
CSV_PATH = OUT_DIR / "suite_prefetch.csv"
FAILURES_PATH = OUT_DIR / "failures.csv"
MERGE_COLS = ["name", "seed"]


def _check_one(name: str, seed: int, task: str = "classification") -> dict:
    """Load one dataset and report its shape against the limits.

    ``task`` comes from the dataset list: the loader reports ``n_classes = 1`` for
    regression, so the task cannot be inferred from the loaded shape.
    """
    data = load_dataset(name, seed=seed)
    n_num = data["X_train"].shape[1]
    n_cat = 0 if data["cat_train"] is None else data["cat_train"].shape[1]
    y_all = np.concatenate([data["y_train"], data["y_val"], data["y_test"]])
    classes, counts = np.unique(y_all, return_counts=True)
    n_classes = int(data["n_classes"])

    row = {
        "name": name, "seed": seed, "status": "ok", "error": "",
        "n_train": len(data["X_train"]), "n_cal": len(data["X_val"]),
        "n_test": len(data["X_test"]),
        "n_num_features": n_num, "n_cat_features": n_cat,
        "n_features_total": n_num + n_cat,
        "n_classes": n_classes,
        "min_class_count": int(counts.min()) if len(counts) else 0,
        "is_regression": task == "regression",
    }
    # The loader raises when a limit is exceeded; a dataset close to a limit
    # is reported too, so the sweep can plan around it.
    warn = []
    if row["n_features_total"] > MAX_FEATURES:
        warn.append(f"features {row['n_features_total']} > {MAX_FEATURES} (reduced by F-score in prepare_split)")
    if not row["is_regression"] and row["min_class_count"] < MIN_CLASS_COUNT:
        warn.append(f"min class {row['min_class_count']} < {MIN_CLASS_COUNT}")
    if not row["is_regression"] and n_classes > 10:
        warn.append(f"{n_classes} classes > 10 (class cap of TabFM, Orion and LimiX)")
    if row["n_cal"] < 100:
        warn.append(f"cal split {row['n_cal']} < 100 translator positions")
    if warn:
        row["status"] = "warn"
        row["error"] = "; ".join(warn)
    return row


def run(
    source=None,
    task: str | None = None,
    seed: int = 0,
    limit: int | None = None,
    force: bool = False,
):
    """Download and check the suite datasets.

    Args:
        source: Restrict to one or more suites (cc18, grinsztajn, tabzilla,
            tabarena, ctr23).
        task: "classification" or "regression" (default: both).
        seed: Split seed to validate with (shape checks are seed-independent;
            the split sizes are not).
        limit: Only the first N datasets, for a quick check.
        force: Re-check datasets already recorded ok/warn in the CSV.
    """
    register_suite_datasets(verbose=False, source=source, task=task)
    df = suite_manifest(source=source, task=task)
    names = df["name"].tolist()
    if limit:
        names = names[:limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if CSV_PATH.exists() and not force:
        prev = pd.read_csv(CSV_PATH)
        done = set(prev[prev["status"].isin(["ok", "warn"])]["name"])
        if done:
            print(f"{len(done)} datasets already validated; --force to re-check")

    todo = [n for n in names if n not in done]
    task_of = dict(zip(df["name"], df["task"]))
    print(f"Validating {len(todo)} of {len(names)} datasets (seed {seed})\n" + "=" * 70)

    rows = []
    for i, name in enumerate(todo, 1):
        try:
            row = _check_one(name, seed, task_of.get(name, "classification"))
        except Exception as exc:  # one failing dataset must not stop the check
            row = {"name": name, "seed": seed, "status": "fail",
                   "error": f"{type(exc).__name__}: {exc}".replace("\n", " ")[:500]}
            traceback.print_exc(limit=1)
        row.update(**provenance())
        rows.append(row)
        flag = {"ok": "  ", "warn": "! ", "fail": "XX"}[row["status"]]
        print(f"{flag} [{i}/{len(todo)}] {name}: {row['status']}"
              + (f" -- {row['error']}" if row["error"] else
                 f" ({row.get('n_features_total')} feat, "
                 f"{row.get('n_classes')} cls, {row.get('n_test')} test)"))
        df_out = merge_rows(pd.DataFrame(rows), CSV_PATH, MERGE_COLS)
        write_csv_atomic(df_out, CSV_PATH)

    if not rows:
        print("Nothing to do.")
        return

    final = pd.read_csv(CSV_PATH)
    counts = final["status"].value_counts().to_dict()
    print("\n" + "=" * 70)
    print(f"Validated {len(final)} datasets: {counts}")

    failures = final[final["status"] != "ok"]
    if not failures.empty:
        write_csv_atomic(failures, FAILURES_PATH)
        print(f"\n{len(failures)} need attention (written to {FAILURES_PATH}):")
        for r in failures.itertuples():
            print(f"  {r.status:4s} {r.name}: {r.error}")
    print(f"\nSaved {CSV_PATH}")
    return final


if __name__ == "__main__":
    fire.Fire(run, serialize=lambda x: None)
