#!/usr/bin/env python3
"""Block-replacement sweep on regression tasks: the regression results table.

Same procedure as scripts/sweeps/run_skip_eval.py with the regressor variant of
each model and R^2 / RMSE / MAE in place of the classification metrics: fit on
the support split, replace each contiguous block window with a translator
fitted on the calibration split, score on the test split; the best start per
width is flagged by calibration R^2.  The default task is MIMIC-III length of
stay; the regression datasets listed in data/dataset_manifest.csv can be given
by name.

Writes: results/regression_skip_eval/regression_skip_results.csv
        results/regression_skip_eval/per_run/<model>_<dataset>_s<seed>_skip<s>-<e>_<translator>.json
        results/failures.csv

Usage:
    python scripts/sweeps/run_regression_skip_eval.py --device cuda
    python scripts/sweeps/run_regression_skip_eval.py --models tabicl --translator identity
    python scripts/sweeps/run_regression_skip_eval.py --datasets <regression names from data/dataset_manifest.csv> --device cuda
"""

from __future__ import annotations

import json
import traceback
import warnings

import fire
import pandas as pd

from shallow_tfm import RESULTS_DIR
from shallow_tfm.config import MODEL_CONFIGS, TRANSLATOR
from shallow_tfm.data.suites import register_suite_datasets
from shallow_tfm.evaluate import baseline_metrics, evaluate_skip_regression
from shallow_tfm.experiment import (
    as_list,
    csv_lock,
    mark_best_by_cal,
    merge_rows,
    parse_models,
    parse_seeds,
    parse_widths,
    prepare_split,
    provenance,
    record_failure,
    resolve_widths,
    set_determinism,
    write_csv_atomic,
)
from shallow_tfm.extraction import extract_representations
from shallow_tfm.models import fit_model

warnings.filterwarnings("ignore", category=UserWarning)

DATASET = "mimic_los"
OUT_DIR = RESULTS_DIR / "regression_skip_eval"
CSV_PATH = OUT_DIR / "regression_skip_results.csv"
MERGE_COLS = ["model", "dataset", "seed", "n_skipped", "translator"]


def run(
    models=None,
    seeds=None,
    skip_widths=None,
    translator: str = TRANSLATOR,
    device: str = "cpu",
    dry_run: bool = False,
    datasets=None,
    force: bool = False,
):
    """Sweep block windows on regression tasks and merge the results into the CSV.

    Args:
        models: Subset of the models (default: every model installed here).
        seeds: Random seeds (default: 0,1,2).
        skip_widths: Window widths, integers and/or the tokens ``half``,
            ``max-1``, ``max`` (default: 3,4).
        translator: "ridge", "linear" or "identity".
        device: "cuda" or "cpu".
        dry_run: Print the plan and exit.
        datasets: Comma-separated regression datasets (default: mimic_los).
        force: Re-run (model, dataset, seed) combinations whose rows already exist.
    """
    models = parse_models(models)
    datasets = as_list(datasets, [DATASET])
    if any(d != DATASET for d in datasets):
        register_suite_datasets(verbose=False)
    seeds = parse_seeds(seeds)
    widths = parse_widths(skip_widths)
    if not models:
        raise SystemExit("No models available in this environment.")

    print(f"models={models} datasets={datasets} seeds={seeds} widths={widths} "
          f"translator={translator} device={device}")
    if dry_run:
        return

    set_determinism()
    per_run_dir = OUT_DIR / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    df = None
    existing = pd.read_csv(CSV_PATH) if CSV_PATH.exists() else None
    for model_name in models:
        widths_m = resolve_widths(widths, MODEL_CONFIGS[model_name]["n_layers"])
        for dataset in datasets:
            for seed in seeds:
                if existing is not None and not force:
                    # Resume: skip when every requested width already has rows.
                    rows = existing[(existing["model"] == model_name) & (existing["dataset"] == dataset)
                                    & (existing["seed"] == seed) & (existing["translator"] == translator)]
                    if all((rows["n_skipped"] == w).any() for w in widths_m):
                        print(f"--- {model_name} / {dataset} / seed {seed}: already in CSV, "
                              "skipping (--force to re-run)")
                        continue
                print(f"\n--- {model_name} / {dataset} / seed {seed} (widths {widths_m}) ---")
                try:
                    results = _evaluate_one(model_name, dataset, seed, widths_m, translator, device)
                except Exception as exc:  # one failure must not stop the sweep
                    traceback.print_exc()
                    print(f"  ERROR: {exc}")
                    record_failure(dataset, model_name, seed, f"regression:{translator}", exc)
                    continue

                if not results:
                    continue
                for result in results:
                    result.update(provenance(device))
                    name = (f"{model_name}_{dataset}_s{seed}_skip{result['skip_start']}-"
                            f"{result['skip_end']}_{translator}.json")
                    (per_run_dir / name).write_text(json.dumps(result, indent=2, default=str))
                df = _merge_into_csv(pd.DataFrame(results))

    if df is None:
        print("\nNo results produced.")
        return

    print(f"\nSaved {len(df)} rows to {CSV_PATH}")
    _print_summary(df)
    return df


def _evaluate_one(model_name, dataset, seed, widths, translator, device):
    cfg = MODEL_CONFIGS[model_name]
    split = prepare_split(dataset, seed, cfg)
    print(f"  train={split['X_train'].shape} cal={split['X_cal'].shape} test={split['X_test'].shape}")

    handle = fit_model(model_name, split["X_train"], split["y_train"], device=device, seed=seed, regression=True)

    # Score the full model before touching the calibration rows: every
    # prediction call changes the model's internal state, so scoring it later
    # would make the full-model numbers depend on the calibration set.
    baseline = baseline_metrics(handle.predict_fn, split["X_test"], split["y_test"], regression=True)
    print(f"  Full: R2={baseline['r2_full']:.4f} RMSE={baseline['rmse_full']:.4f}")

    reps = extract_representations(handle.model, handle.layers, handle.predict_fn, split["X_cal"])
    print(f"  extracted {len(reps)} depths")

    results = []
    for width in widths:
        if width >= handle.n_layers:
            continue
        for start in range(handle.n_layers - width + 1):
            end = start + width
            if start not in reps or end not in reps:
                continue
            result = evaluate_skip_regression(
                handle.predict_fn, handle.layers, split["X_test"], split["y_test"],
                skip_start=start, skip_end=end,
                representations=reps, translator=translator,
                baseline=baseline, device=handle.device,
                return_tuple_size=cfg["return_tuple_size"],
                per_feature=cfg["per_feature"],
                pad_tuple_none=cfg.get("pad_tuple_none", False),
                X_cal=split["X_cal"], y_cal=split["y_cal"],
            )
            result.update(model=model_name, dataset=dataset, seed=seed, n_layers=handle.n_layers, width=width)
            results.append(result)
            print(f"  skip {start}->{end}: dR2={result['delta_r2']:+.4f} dRMSE={result['delta_rmse']:+.4f}")
    return results


def _merge_into_csv(df_new: pd.DataFrame) -> pd.DataFrame:
    """Replace the stored rows that the new rows supersede, then save."""
    with csv_lock(CSV_PATH):  # a concurrent writer would lose rows otherwise
        df = merge_rows(df_new, CSV_PATH, MERGE_COLS)
        df = mark_best_by_cal(df, MERGE_COLS, "r2_cal_skip")
        write_csv_atomic(df, CSV_PATH)
    return df


def _print_summary(df: pd.DataFrame):
    best = df[df["is_best_for_width"]]
    print("\nMean over seeds (best window per width)")
    for (model, width), group in best.groupby(["model", "n_skipped"]):
        print(f"  {model:<12s} w={int(width)}: "
              f"dR2={group['delta_r2'].mean():+.4f}+-{group['delta_r2'].std():.4f} "
              f"dRMSE={group['delta_rmse'].mean():+.4f}")


if __name__ == "__main__":
    fire.Fire(run, serialize=lambda x: None)
