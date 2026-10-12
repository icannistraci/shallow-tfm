#!/usr/bin/env python3
"""Block-replacement sweep over every contiguous window of every model: the
main results tables, the cost figures and most ablations.

For every (model, dataset, seed, width) the model is fitted on the support
split, each contiguous window of ``width`` blocks is replaced by a translator
fitted on the calibration split, and full / compressed / delta metrics are
recorded on the test split together with the calibration-split metrics and
probabilities that the label-free window selection needs
(scripts/sweeps/compute_window_kl.py).  Representations are extracted once per
(model, dataset, seed) and shared by all widths and translators of one call.
Results already in the CSV are not recomputed, each translator's rows are
merged as soon as it finishes, and a failure is logged and skipped.

Writes: results/consistent_skip_eval/consistent_skip_results.csv
        results/consistent_skip_eval/per_run/<model>_<dataset>_s<seed>_skip<s>-<e>_<translator>.{json,npz}
        results/consistent_skip_eval/per_run/<model>_<dataset>_s<seed>_full.npz
        results/consistent_skip_eval/logs/sweep_<timestamp>.log
        results/failures.csv

Usage:
    python scripts/sweeps/run_skip_eval.py --device cuda
    python scripts/sweeps/run_skip_eval.py --models tabicl --datasets diabetes,heart
    python scripts/sweeps/run_skip_eval.py --translator ridge,identity --skip_widths 3,4,half,max-1,max
    python scripts/sweeps/run_skip_eval.py --suites all --translator ridge,identity --device cuda
    python scripts/sweeps/run_skip_eval.py --suites tabarena --limit 2 --models tabicl
"""

from __future__ import annotations

import json
import sys
import traceback
import warnings
from datetime import datetime, timezone

import fire
import numpy as np
import pandas as pd
import torch
import torch._dynamo

from shallow_tfm.config import MODEL_CONFIGS, TRANSLATOR
from shallow_tfm.data.suites import register_suite_datasets, suite_datasets
from shallow_tfm.evaluate import apply_full_recalibration, baseline_metrics, evaluate_skip, full_recalibration
from shallow_tfm.experiment import (
    SKIP_EVAL_CSV,
    as_list,
    csv_lock,
    mark_best_by_cal,
    merge_rows,
    parse_datasets,
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
from shallow_tfm.skip import FITTED_TRANSLATORS

torch._dynamo.config.suppress_errors = True
warnings.filterwarnings("ignore", category=UserWarning)

CSV_PATH = SKIP_EVAL_CSV
OUT_DIR = CSV_PATH.parent
LOG_DIR = OUT_DIR / "logs"

#: Columns that identify a row; new rows replace stored rows with the same key.
MERGE_COLS = ["model", "dataset", "seed", "n_skipped", "translator"]

#: Suite order for ``--suites all``.
SUITE_ORDER = ("tabarena", "tabzilla", "cc18", "grinsztajn")


class StageFailure(RuntimeError):
    """Carries the stage a run failed in, for results/failures.csv."""

    def __init__(self, stage: str, exc: Exception):
        super().__init__(str(exc))
        self.stage = stage


class _Tee:
    """Mirror a stream into the per-sweep log file."""

    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, text):
        self._stream.write(text)
        self._fh.write(text)
        self._fh.flush()

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def isatty(self):
        return self._stream.isatty()


def run(
    models=None,
    datasets=None,
    seeds=None,
    skip_widths=None,
    translator: str = TRANSLATOR,
    device: str = "cpu",
    dry_run: bool = False,
    force: bool = False,
    suites=None,
    limit: int | None = None,
):
    """Sweep block windows and merge the results into the CSV.

    A (model, dataset, seed) whose rows are already in the CSV is skipped, so
    an interrupted sweep resumes where it stopped; each translator's rows are
    merged as soon as it finishes, so a crash loses at most the translator
    being evaluated.  A failure is appended to results/failures.csv (dataset,
    model, seed, stage, error, timestamp) and the sweep moves on.

    Args:
        models: Subset of the models (default: every model installed in this
            environment).
        datasets: Subset of the eight clinical datasets (default: all);
            replaced by the suite datasets when ``suites`` is given.
        seeds: Random seeds (default: 0,1,2).
        skip_widths: Window widths to sweep, integers and/or the tokens
            ``half``, ``max-1``, ``max`` resolved per model (default: 3,4).
        translator: One or more of "ridge" (the fitted map with label-free
            ridge selection), "linear" (plain least squares) and "identity"
            (naive dropping), comma-separated.  All translators in one call
            share a single extraction per (model, dataset, seed).
        device: "cuda" or "cpu".
        dry_run: Print the plan and exit.
        force: Re-run (model, dataset, seed) combinations whose rows already exist.
        suites: Comma-separated suites from data/dataset_manifest.csv (tabarena,
            tabzilla, cc18, grinsztajn) or "all" for the four in that order;
            their classification datasets are swept in suite order.
        limit: With ``suites``, only the first N datasets of each suite.
    """
    models = parse_models(models)
    datasets = _suite_datasets(suites, limit) if suites is not None else parse_datasets(datasets)
    seeds = parse_seeds(seeds)
    translators = as_list(translator, [TRANSLATOR])
    widths = parse_widths(skip_widths)

    if not models:
        raise SystemExit("No models available in this environment.")

    print(f"models={models} datasets={datasets} seeds={seeds} "
          f"widths={widths} translators={translators} device={device}")
    if dry_run:
        return

    set_determinism()
    per_run_dir = OUT_DIR / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"sweep_{datetime.now(timezone.utc):%Y%m%d_%H%M%SZ}.log"
    print(f"log: {log_path}")

    stdout, stderr = sys.stdout, sys.stderr
    with open(log_path, "w") as log_fh:
        sys.stdout, sys.stderr = _Tee(stdout, log_fh), _Tee(stderr, log_fh)
        try:
            df = _sweep(models, datasets, seeds, widths, translators, device, force, per_run_dir)
        finally:
            sys.stdout, sys.stderr = stdout, stderr

    if df is None:
        print("\nNo new results produced.")
        return

    print(f"\nSaved {len(df)} rows to {CSV_PATH}")
    _print_summary(df)
    return df


def _suite_datasets(suites, limit) -> list[str]:
    """Classification datasets of the requested suites, in suite order."""
    sources = list(SUITE_ORDER) if suites == "all" else as_list(suites)
    register_suite_datasets(verbose=False)
    names = []
    for source in sources:
        found = suite_datasets(source=source, task="classification")
        found = found[:limit] if limit else found
        print(f"[{source}] {len(found)} datasets")
        names += found
    return names


def _sweep(models, datasets, seeds, widths, translators, device, force, per_run_dir):
    """Loop over (model, dataset, seed), evaluating every requested translator for each."""
    existing = pd.read_csv(CSV_PATH) if CSV_PATH.exists() else None

    df = None
    for model_name in models:
        widths_m = resolve_widths(widths, MODEL_CONFIGS[model_name]["n_layers"])
        print(f"[{model_name}] widths {widths} -> {widths_m}")
        for dataset in datasets:
            for seed in seeds:
                todo = [t for t in translators
                        if force or not _already_done(existing, model_name, dataset, seed, widths_m, t)]
                done = [t for t in translators if t not in todo]
                if done:
                    print(f"--- {model_name} / {dataset} / seed {seed}: "
                          f"{','.join(done)} already in CSV, skipping (--force to re-run)")
                if not todo:
                    continue
                print(f"\n--- {model_name} / {dataset} / seed {seed} ({','.join(todo)}) ---")
                try:
                    for results in _evaluate_one(model_name, dataset, seed, widths_m, todo, device, per_run_dir):
                        if not results:
                            continue
                        for result in results:
                            result.update(model=model_name, dataset=dataset, seed=seed, **provenance(device))
                            name = (f"{model_name}_{dataset}_s{seed}_skip{result['skip_start']}-"
                                    f"{result['skip_end']}_{result['translator']}.json")
                            (per_run_dir / name).write_text(json.dumps(result, indent=2, default=str))
                        df = _merge_into_csv(pd.DataFrame(results))
                except Exception as exc:  # one failure must not stop the sweep
                    traceback.print_exc()
                    stage = getattr(exc, "stage", "unknown")
                    cause = exc.__cause__ if isinstance(exc, StageFailure) and exc.__cause__ else exc
                    print(f"  ERROR [{stage}]: {cause}")
                    record_failure(dataset, model_name, seed, stage, cause)
                    continue
                finally:
                    torch.cuda.empty_cache()  # free what a failed run left on the GPU, or the next run may run out of memory
    return df


def _already_done(existing, model, dataset, seed, widths, translator) -> bool:
    """True when the stored CSV already has rows for every requested width of
    this (model, dataset, seed, translator).  Only presence is checked: a run
    interrupted in the middle of a width is repeated in full."""
    if existing is None or existing.empty:
        return False
    rows = existing[(existing["model"] == model) & (existing["dataset"] == dataset)
                    & (existing["seed"] == seed) & (existing["translator"] == translator)]
    return all((rows["n_skipped"] == width).any() for width in widths)


def _save_probas(path, **arrays):
    np.savez_compressed(path, **{k: v for k, v in arrays.items() if v is not None})


def _evaluate_one(model_name, dataset, seed, widths, translators, device, per_run_dir):
    """Evaluate every requested translator for one (model, dataset, seed),
    yielding each translator's rows as it completes.

    Representations are extracted once, by the first fitted translator, and
    reused by the rest; identity-only runs never extract.  Every translator
    still re-fits the model and repeats the same sequence of prediction calls
    as a separate run, because each call changes the model's internal state
    and every window's numbers depend on it.
    """
    try:
        split = prepare_split(dataset, seed, MODEL_CONFIGS[model_name])
    except Exception as exc:
        raise StageFailure("prepare_split", exc) from exc
    print(f"  train={split['X_train'].shape} cal={split['X_cal'].shape} test={split['X_test'].shape}")

    reps = None
    for translator in translators:
        results, reps = _evaluate_translator(model_name, dataset, seed, widths, translator, device,
                                             per_run_dir, split, reps)
        yield results


def _evaluate_translator(model_name, dataset, seed, widths, translator, device, per_run_dir, split, reps):
    """Sweep every start position at every width for one translator; returns
    (rows, representations) so the caller can reuse the extraction."""
    cfg = MODEL_CONFIGS[model_name]
    stage = "fit"
    try:
        # The model is re-fitted for every translator, so each one starts from
        # the same freshly seeded state as a separate run would.
        handle = fit_model(model_name, split["X_train"], split["y_train"], device=device, seed=seed)
        print(f"  [{translator}] {handle.n_layers} transformer blocks")

        # Score the full model before touching the calibration rows: every
        # prediction call changes the model's internal state (the random
        # generator, or TabICL's ensemble generator), so scoring it later would
        # make the full-model numbers depend on the calibration set.
        stage = "baseline"
        baseline = baseline_metrics(handle.predict_fn, split["X_test"], split["y_test"], return_probas=True)
        proba_test_full = baseline.pop("_probas")["proba_test_full"]
        print(f"  Full: acc={baseline['accuracy_full']:.4f} auroc={baseline['auroc_full']:.4f} "
              f"auprc={baseline['auprc_full']:.4f} f1={baseline['f1_full']:.4f}")

        stage = "extract"
        if translator in FITTED_TRANSLATORS and reps is None:
            # TabICL and TabDPT handle at most ten classes in one run of the
            # transformer.  With more classes they run it several times for a
            # single prediction, and each run gets its own translator.
            per_pass = bool(cfg.get("per_pass_translator")) and len(np.unique(split["y_train"])) > 10
            reps = extract_representations(
                handle.model, handle.layers, handle.predict_fn, split["X_cal"], per_pass=per_pass)
            if per_pass:
                if len(reps) == 1:
                    reps = reps[0]
                else:
                    print(f"  {len(reps)} transformer runs per prediction: one translator each")
            ref = reps[0] if isinstance(reps, list) else reps
            print(f"  extracted {len(ref)} depths, shape {tuple(ref[0].shape)}")
        else:
            # Predict on the calibration rows anyway: the call changes the
            # model's internal state exactly as the extraction pass would, and
            # every window's numbers depend on that state.
            handle.predict_fn(split["X_cal"])

        stage = "windows"
        results = []
        for width in widths:
            if width >= handle.n_layers:
                print(f"  width {width} >= {handle.n_layers} blocks, skipping")
                continue
            for start in range(handle.n_layers - width + 1):
                end = start + width
                ref = reps[0] if isinstance(reps, list) else reps
                if reps is not None and (start not in ref or end not in ref):
                    continue

                result = evaluate_skip(
                    handle.predict_fn, handle.layers, split["X_test"], split["y_test"],
                    skip_start=start, skip_end=end,
                    representations=reps, translator=translator,
                    baseline=baseline, device=handle.device,
                    return_tuple_size=cfg["return_tuple_size"],
                    per_feature=cfg["per_feature"],
                    pad_tuple_none=cfg.get("pad_tuple_none", False),
                    X_cal=split["X_cal"], y_cal=split["y_cal"],
                    return_probas=True,
                )
                _save_probas(
                    per_run_dir / f"{model_name}_{dataset}_s{seed}_skip{start}-{end}_{translator}.npz",
                    **result.pop("_probas"),
                )
                result["n_layers"] = handle.n_layers
                result["width"] = width
                results.append(result)
                print(f"  skip {start}->{end}: DAUC={result['delta_auroc']:+.4f} "
                      f"DAUPRC={result['delta_auprc']:+.4f} DF1={result['delta_f1']:+.4f}")

        # The full model predicts on the calibration rows last: this call also
        # changes the model's internal state, so doing it earlier would change
        # every window's numbers.
        stage = "recalibrate"
        recal_full, proba_cal_full = full_recalibration(
            handle.predict_fn, split["y_test"], proba_test_full, split["X_cal"], split["y_cal"],
        )
        apply_full_recalibration(results, recal_full)
        _save_probas(
            per_run_dir / f"{model_name}_{dataset}_s{seed}_full.npz",
            y_test=split["y_test"], y_cal=split["y_cal"],
            proba_test_full=proba_test_full, proba_cal_full=proba_cal_full,
        )
    except Exception as exc:
        raise StageFailure(f"{stage}:{translator}", exc) from exc

    return results, reps


def _merge_into_csv(df_new: pd.DataFrame) -> pd.DataFrame:
    """Replace the stored rows that the new rows supersede, then save."""
    with csv_lock(CSV_PATH):  # a concurrent writer would lose rows otherwise
        df = merge_rows(df_new, CSV_PATH, MERGE_COLS)
        # With `translator` in the key, the best ridge window and the best
        # identity window are both flagged.
        df = mark_best_by_cal(df, MERGE_COLS, "auroc_cal_skip")
        write_csv_atomic(df, CSV_PATH)
    return df


def _print_summary(df: pd.DataFrame):
    best = df[df["is_best_for_width"] & (df["translator"] == TRANSLATOR)]
    if best.empty:
        return
    print(f"\nMean over seeds (best window per width by calibration AUROC, {TRANSLATOR} translator)")
    print(f"  {'model':<12s} {'dataset':<16s} {'w':>2s} {'dAcc':>14s} {'dAUROC':>14s}")
    for (model, dataset, width), group in best.groupby(["model", "dataset", "n_skipped"]):
        print(f"  {model:<12s} {dataset:<16s} {int(width):>2d} "
              f"{group['delta_accuracy'].mean():>+8.3f}+-{group['delta_accuracy'].std():<5.3f} "
              f"{group['delta_auroc'].mean():>+8.3f}+-{group['delta_auroc'].std():<5.3f}")


if __name__ == "__main__":
    fire.Fire(run, serialize=lambda x: None)
