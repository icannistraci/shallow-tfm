#!/usr/bin/env python3
"""Forward KL on the calibration split for every window of the main sweep: the
label-free window-selection score.

The sweep stores, per window, the compressed model's calibration probabilities
(per_run/*_skip<s>-<e>_<translator>.npz, key ``proba_cal_skip``) and, per
(model, dataset, seed), the full model's (*_full.npz, key ``proba_cal_full``),
so KL(p_full || p_compressed) averaged over calibration rows is computed for
the whole sweep offline, without re-running any model.  Every table and figure
selects the window with the smallest value of this score
(``shallow_tfm.experiment.select_windows``).

Reads:  results/consistent_skip_eval/consistent_skip_results.csv
        results/consistent_skip_eval/per_run/*.npz
Writes: results/consistent_skip_eval/window_kl.csv  (keyed by model, dataset,
        seed, translator, skip_start, skip_end; columns kl_cal_fwd and n_cal;
        kl_cal_fwd is NaN where a window's probabilities are missing, so
        coverage gaps stay visible)

Usage:
    python scripts/sweeps/compute_window_kl.py
    python scripts/sweeps/compute_window_kl.py --translators ridge,identity
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import fire
import numpy as np
import pandas as pd

from shallow_tfm.evaluate import kl_forward
from shallow_tfm.experiment import SELECTION_COL, SKIP_EVAL_CSV, WINDOW_KL_CSV, restrict_report_seeds

PER_RUN = SKIP_EVAL_CSV.parent / "per_run"
KEY = ["model", "dataset", "seed", "translator", "skip_start", "skip_end"]


def _one_run(args) -> list[tuple]:
    """All windows of one (model, dataset, seed): load the full model's calibration probabilities once."""
    (model, dataset, seed), windows = args
    stem = f"{model}_{dataset}_s{seed}"
    out = []
    try:
        p_full = np.load(PER_RUN / f"{stem}_full.npz")["proba_cal_full"]
    except (FileNotFoundError, KeyError):
        p_full = None
    for tr, s, e in windows:
        kl, n = np.nan, 0
        if p_full is not None:
            try:
                q = np.load(PER_RUN / f"{stem}_skip{s}-{e}_{tr}.npz")["proba_cal_skip"]
                if q.shape == p_full.shape:
                    kl, n = kl_forward(p_full, q), len(q)
            except (FileNotFoundError, KeyError, ValueError):
                pass
        out.append((model, dataset, seed, tr, s, e, kl, n))
    return out


def run(translators: str = "ridge", workers: int = 16) -> None:
    """Score every stored window of the given translators.

    Args:
        translators: Comma-separated translators whose windows are scored.
        workers: Parallel processes reading the per-run files.
    """
    trs = translators.split(",") if isinstance(translators, str) else list(translators)
    df = restrict_report_seeds(pd.read_csv(SKIP_EVAL_CSV, low_memory=False, usecols=KEY))
    df = df[df["translator"].isin(trs)].drop_duplicates(KEY)
    df[["seed", "skip_start", "skip_end"]] = df[["seed", "skip_start", "skip_end"]].astype(int)
    jobs = [((m, d, s), list(zip(g.translator, g.skip_start, g.skip_end)))
            for (m, d, s), g in df.groupby(["model", "dataset", "seed"])]
    print(f"{len(df)} windows in {len(jobs)} (model, dataset, seed) runs")
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(_one_run, jobs, chunksize=8)):
            rows += r
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)} runs")
    res = pd.DataFrame(rows, columns=KEY + [SELECTION_COL, "n_cal"])
    res.to_csv(WINDOW_KL_CSV, index=False)
    miss = res[SELECTION_COL].isna()
    print(f"wrote {WINDOW_KL_CSV}: {len(res)} windows, {miss.sum()} without KL")
    if miss.any():
        print(res[miss].groupby("model").size().to_string())


if __name__ == "__main__":
    fire.Fire(run)
