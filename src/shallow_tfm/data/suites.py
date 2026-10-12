"""OpenML suite datasets: ``data/dataset_manifest.csv`` lists the datasets of
the four classification suites (OpenML-CC18, Grinsztajn, TabZilla, TabArena)
and of the regression suites (Grinsztajn, TabArena, CTR-23).

``register_suite_datasets`` adds them to ``datasets.DATASETS`` at run time, so
a script can use a suite dataset by name exactly like one of the clinical
datasets listed in ``datasets.py``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from shallow_tfm import DATA_DIR
from shallow_tfm.config import DISPLAY_NAMES
from shallow_tfm.data import datasets as D

MANIFEST_PATH = DATA_DIR / "dataset_manifest.csv"

SUITE_SOURCES = ("cc18", "grinsztajn", "tabzilla", "tabarena", "ctr23")

#: Names added by ``register_suite_datasets``, so suite entries can be told
#: apart from the clinical ones listed in ``datasets.py``.
_SUITE_REGISTERED: set[str] = set()


@lru_cache(maxsize=1)
def load_manifest(path: str | Path = MANIFEST_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"name", "source", "openml_id", "task"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def suite_manifest(source: str | list[str] | None = None, task: str | None = None) -> pd.DataFrame:
    """Rows of ``data/dataset_manifest.csv`` filtered by suite ``source`` and
    ``task`` (``classification`` or ``regression``)."""
    df = load_manifest()
    if source is not None:
        sources = [source] if isinstance(source, str) else list(source)
        unknown = set(sources) - set(SUITE_SOURCES)
        if unknown:
            raise ValueError(f"Unknown source(s) {sorted(unknown)}; choose from {SUITE_SOURCES}")
        df = df[df["source"].isin(sources)]
    if task is not None:
        df = df[df["task"] == task]
    return df.reset_index(drop=True)


def suite_datasets(**kwargs) -> list[str]:
    """Dataset names matching ``suite_manifest(**kwargs)``."""
    names = suite_manifest(**kwargs)["name"].tolist()
    return [n for n in names if n not in D.DATASETS or n in _SUITE_REGISTERED]


def register_suite_datasets(verbose: bool = True, **kwargs) -> list[str]:
    """Add the listed datasets to ``datasets.DATASETS``; returns the names added.
    Existing entries are left untouched, so calling it twice is harmless."""
    added = []
    for row in suite_manifest(**kwargs).itertuples():
        if row.name in D.DATASETS:
            continue
        cfg = {
            "openml_id": int(row.openml_id),
            "default_max_samples": D.GLOBAL_MAX_SAMPLES,
            "description": f"{row.source} {row.task} ({int(row.n_feat)} feat, {int(row.n_classes)} cls)",
        }
        if row.task == "regression":
            cfg["task"] = "regression"
        D.DATASETS[row.name] = cfg
        _SUITE_REGISTERED.add(row.name)
        added.append(row.name)
    if verbose:
        print(f"Registered {len(added)} suite datasets ({len(D.DATASETS)} available in total).")
    return added


def display_name(name: str) -> str:
    """Human-readable label: the entry in ``DISPLAY_NAMES`` if there is one,
    else the OpenML name."""
    if name in DISPLAY_NAMES:
        return DISPLAY_NAMES[name]
    hit = load_manifest().query("name == @name")
    return name if hit.empty else str(hit.iloc[0]["openml_name"])

