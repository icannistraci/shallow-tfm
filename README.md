<div align="center">

# Tabular Foundation Models Are Effectively Shallow

**Irene Cannistraci**\*, Julia E. Vogt

ETH Zurich

\* *irene.cannistraci@inf.ethz.ch*

[![arXiv](https://img.shields.io/badge/arXiv-2610.13253-b31b1b.svg)](https://arxiv.org/abs/2610.13253)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

> In-context-learning Tabular Foundation Models (TFMs) are uniquely suited to scenarios with limited resources or privacy constraints, as they solve new tasks without additional training. However, their architecture consists of multiple transformer blocks, resulting in high computational and memory footprints. Meanwhile, traditional alternatives like Gradient-Boosted Decision Trees (GBDTs) remain highly competitive and run efficiently on CPUs, suggesting that tabular tasks may not inherently require deep architectures. This contrast raises a fundamental question: *can TFMs be simplified by removing architectural redundancies without sacrificing performance?* To answer this, we introduce a closed-form, label-free affine translator that replaces a contiguous window of up to 96% of the blocks, leaving the rest unchanged. Evaluating eight frontier TFMs on 171 classification datasets, we reveal that tabular transformers are vastly overparameterized: replacing half of a model's blocks costs at most 0.002 median AUROC across all architectures, with the half-depth TabFM still outperforming every other full-depth TFM and tuned GBDTs. Furthermore, different model families are overparameterized in different ways: TabPFN-3 relies on its encoders, maintaining performance regardless of which single block is retained. Other single-axis models spread computation thinly, needing only their first block, while dual-axis models concentrate it in one critical block. Ultimately, removing depth saves up to 1.5B parameters, speeds up GPU inference by up to 6× at a median cost of 0.03 AUROC, while a single-block TabPFN-3 serves cached queries 2.4× faster using 6× less memory at negligible cost.

## Installation

```bash
pip install -e .
pip install -r environments/base.txt   # pinned versions used for the paper
pip install -e ".[dev]"                # pytest and ruff, needed only for make test
```

The base environment runs TabPFNv2, TabICL, TabDPT and TabFM. Three models need their own environment because of conflicting packages; each is a venv created on top of the base one with `--system-site-packages` (see the header of each file):

| Environment | Models | Notes |
|-------------|--------|-------|
| [`environments/tabpfn3.txt`](environments/tabpfn3.txt) | TabPFN-3 | needs `tabpfn>=8`; weights are license-gated (`TABPFN_TOKEN`) |
| [`environments/orion.txt`](environments/orion.txt) | Orion-MSP, Orion-BiX | installed from the upstream git repositories |
| [`environments/limix.txt`](environments/limix.txt) | LimiX-16M | upstream checkout under `third_party/LimiX`; checkpoint from the Hugging Face Hub |

Every script takes `--models` and runs the models that are importable in the current environment, so a sweep over all eight models is the same call once per environment.

## Quick start

Replace blocks 1–11 of TabICL (12 blocks) by one ridge-fitted affine map and compare with the full model:

```python
from shallow_tfm.config import MODEL_CONFIGS
from shallow_tfm.evaluate import baseline_metrics, evaluate_skip
from shallow_tfm.experiment import prepare_split
from shallow_tfm.extraction import extract_representations
from shallow_tfm.models import fit_model

cfg = MODEL_CONFIGS["tabicl"]
split = prepare_split("diabetes", seed=0, cfg=cfg)        # support / calibration / test
h = fit_model("tabicl", split["X_train"], split["y_train"], device="cuda", seed=0)

full = baseline_metrics(h.predict_fn, split["X_test"], split["y_test"])
reps = extract_representations(h.model, h.layers, h.predict_fn, split["X_cal"])  # {depth: activations}

row = evaluate_skip(h.predict_fn, h.layers, split["X_test"], split["y_test"],
                    skip_start=1, skip_end=12, representations=reps, translator="ridge",
                    baseline=full, return_tuple_size=cfg["return_tuple_size"],
                    per_feature=cfg["per_feature"])
print(f"full AUROC {row['auroc_full']:.3f}, one block kept {row['auroc_skip']:.3f}")
# full AUROC 0.882, one block kept 0.877
```

Run the snippet with `PYTHONHASHSEED=0 python quickstart.py`. TabICL orders its ensemble members from a Python set, so with another hash seed the translator is fitted on a different subset of members and the one-block AUROC comes out at 0.881 instead of 0.877 (see Usage); CPU and GPU give the same numbers. The warning that extraction captured two forward passes and kept the first is expected: TabICL runs its eight ensemble members in two batches of four, and the translator is fitted on the first batch, as in all released TabICL results.

`skip_context` performs the replacement as a context manager (the original blocks are restored on exit), `evaluate_skip` wraps it with the metrics, and `select_by_kl` picks the window without labels by the KL divergence to the full model on the calibration rows. The window is any pair `skip_start`, `skip_end` with `0 <= skip_start < skip_end <= n_blocks`: blocks `skip_start` to `skip_end - 1` are replaced and the rest run unchanged. The KL rule is how the paper chooses the window, not a requirement: any window can be evaluated this way.

## Usage

The pipeline has two steps. Step 1 produces the numbers: for every model, dataset and seed it replaces each contiguous window of blocks, at every width, with a translator, scores the compressed model on the test split, and stores one row per window together with that window's predictions on the calibration rows. This is the expensive part, since it runs the models on the GPU. Step 2 decides which window to report: a 12-block model has ten candidate windows of width 3, and the paper keeps one per (model, dataset, seed, width), the one with the lowest KL divergence to the full model on the calibration rows, which needs no labels. Step 2 only reads the files saved by step 1, so it runs in seconds on a CPU, and the selection can be recomputed without sweeping again. Step 2 is only needed to reproduce the paper's choice: after step 1 every window already has its own row in the CSV, and a single window of your choice can be evaluated directly with the library (see Quick start).

The `Makefile` chains the two steps (`make smoke` runs one small example end to end to check the install), and every script is a [Fire](https://github.com/google/python-fire) CLI, so `--help` lists its arguments. Sweeps are resumable: (model, dataset, seed) combinations already in the results CSV are skipped, and a failure is logged to `results/failures.csv` while the sweep moves on. Run the scripts with `PYTHONHASHSEED=0` (the Makefile sets it): TabICL orders its ensemble members from a Python set, so without a fixed hash seed the translator may be fitted on a different batch of members from run to run, and the paper's rows correspond to seed 0.

### Step 1 &mdash; Sweep block windows

```bash
python scripts/sweeps/run_skip_eval.py --skip_widths 3,4,half,max-1,max --translator ridge,identity --device cuda
python scripts/sweeps/run_skip_eval.py --skip_widths 3,4,half,max-1,max --translator ridge,identity --device cuda --suites all
python scripts/sweeps/run_regression_skip_eval.py --skip_widths 3,4,half,max-1,max --device cuda
```

Runs [`scripts/sweeps/run_skip_eval.py`](scripts/sweeps/run_skip_eval.py) (classification) and [`scripts/sweeps/run_regression_skip_eval.py`](scripts/sweeps/run_regression_skip_eval.py) (regression). Key parameters (a list is written comma-separated, without spaces):

| Parameter | Description | Example values |
|-----------|-------------|----------------|
| `models` | Subset of the models (default: every model installed in the current environment) | `tabicl` or `tabpfn,tabfm` |
| `datasets` | Subset of the clinical datasets (default: all) | `diabetes` or `mimic,eicu` |
| `suites` | Sweep the OpenML suite datasets instead (classification sweep only) | `all` or `tabarena,cc18` |
| `limit` | With `suites`, only the first N datasets of each suite | `5` |
| `seeds` | Random seeds (default: `0,1,2`) | `0` |
| `skip_widths` | Window widths, integers and/or tokens resolved per model | `3,4` or `half,max-1,max` |
| `translator` | Map that replaces the window; several at once as a list | `ridge` (fitted, label-free λ selection), `linear` or `identity` (naive dropping) |
| `device` | Where the model runs | `cuda` or `cpu` |
| `dry_run` | Print the plan and exit | `True` |
| `force` | Re-run combinations that already have rows | `True` |

Results go to `results/consistent_skip_eval/` and `results/regression_skip_eval/`.

### Step 2 &mdash; Select the window without labels

```bash
python scripts/sweeps/compute_window_kl.py
```

Runs [`scripts/sweeps/compute_window_kl.py`](scripts/sweeps/compute_window_kl.py): for every window stored by step 1 it computes, from the saved calibration predictions, the forward KL between the full and the compressed model and writes `results/consistent_skip_eval/window_kl.csv`. Every reported number uses the window with the lowest calibration KL.

| Parameter | Description | Example values |
|-----------|-------------|----------------|
| `translators` | Translators whose windows are scored; several at once as a list | `ridge` or `ridge,identity` |
| `workers` | Parallel processes reading the per-run files | `16` |

## Supported models

| Model | Short name | Package | Blocks | Environment |
|-------|-----------|---------|--------|-------------|
| TabPFNv2 | `tabpfn` | `tabpfn` 2.x | 12 | base |
| TabPFN-3 | `tabpfn3` | `tabpfn` 8.x | 24 | `tabpfn3` |
| TabICL | `tabicl` | `tabicl` | 12 | base |
| TabDPT | `tabdpt` | `tabdpt` | 16 | base |
| TabFM | `tabfm` | `tabfm` | 24 | base |
| Orion-MSP | `orion_msp` | `orion_msp` | 12 | `orion` |
| Orion-BiX | `orion_bix` | `orion_bix` | 12 | `orion` |
| LimiX-16M | `limix_16m_pf` | `limix` | 12 | `limix` |

See [`src/shallow_tfm/config.py`](src/shallow_tfm/config.py) for the full configuration (block paths, calibration set sizes, per-feature maps) and [`src/shallow_tfm/models.py`](src/shallow_tfm/models.py) for the loaders.

## Supported datasets

Eight clinical datasets and one clinical regression task. The six public ones are the classic UCI datasets; the code downloads OpenML's copies of them, so the rows and columns are exactly those of the OpenML ids below. (Pima Indians Diabetes is no longer hosted by UCI, and UCI's dataset 34 "Diabetes" is a different dataset of patient logs.)

| Name | Dataset | Downloaded from | Task |
|------|---------|-----------------|------|
| `diabetes` | Pima Indians Diabetes (Smith et al., 1988) | OpenML 37 | binary |
| `heart` | Statlog Heart (UCI 145) | OpenML 53 | binary |
| `breast_cancer` | Breast Cancer Wisconsin, original (UCI 15) | OpenML 15 | binary |
| `support2` | SUPPORT2 hospital mortality (Knaus et al., 1995) | OpenML 46628 | binary |
| `mimic` | MIMIC-III v1.4, 48 h ICU mortality | PhysioNet | binary |
| `eicu` | eICU-CRD v2.0, 48 h ICU mortality | PhysioNet | binary |
| `cardiotocography` | Cardiotocography fetal health (UCI 193) | OpenML 1560 | 3-class |
| `thyroid` | Thyroid Disease, ann-thyroid (UCI 102) | OpenML 40497 | 3-class |
| `mimic_los` | MIMIC-III v1.4, ICU length of stay | PhysioNet | regression |

Plus the datasets of five benchmark suites, each a published list of OpenML datasets, listed in [`data/dataset_manifest.csv`](data/dataset_manifest.csv) (163 classification and 85 regression datasets):

| Suite | Classification | Regression |
|-------|---------------:|-----------:|
| OpenML-CC18 | 56 | – |
| Grinsztajn | 25 | 36 |
| TabZilla | 44 | 1 |
| TabArena | 38 | 13 |
| CTR-23 | – | 35 |

OpenML datasets are downloaded automatically the first time they are used ([`scripts/sweeps/prefetch_suites.py`](scripts/sweeps/prefetch_suites.py) downloads and checks all of them in one go). MIMIC-III and eICU-CRD are not included in this repository: they are hosted on [PhysioNet](https://physionet.org), which requires a signed data-use agreement. Once you have access, download them and put the CSV files under `data/physionet/mimiciii/1.4` and `data/physionet/eicu-crd/2.0` (or point `MIMIC_DIR` / `EICU_DIR` to them). The patient cohorts are built once and cached under `data/cache/`.

## Layout

```
src/shallow_tfm/
  config.py          model architecture facts, dataset groups, seeds, settings
  models.py          fit any of the eight models behind one ModelHandle
  translators.py     the affine translators and their least-squares / ridge fits
  extraction.py      captures every block's output during a forward pass
  skip.py            context manager that swaps a block window for a translator
  evaluate.py        metrics, recalibration, compressed-vs-full scoring, KL selection
  experiment.py      CLI parsing, split preparation, saving results to CSV, window selection
  data/              OpenML loader, suite dataset list, MIMIC-III and eICU-CRD cohorts
scripts/sweeps/
  run_skip_eval.py              classification sweep (step 1)
  run_regression_skip_eval.py   regression sweep (step 1)
  compute_window_kl.py          window selection (step 2)
  prefetch_suites.py            downloads and checks every OpenML suite dataset
configs/             LimiX inference settings: the first entry of LimiX's default no-retrieval configs (Apache-2.0), i.e. one ensemble member
data/                dataset_manifest.csv, the list of suite datasets; the PhysioNet files and cohort caches go here (ignored by git)
environments/        pinned package versions of the four environments
tests/               unit tests (make test)
```

This first release contains the method and the main experiment of the paper: the library that replaces a block window with a ridge-fitted affine translator and selects the window without labels, the classification and regression sweeps over every contiguous window of the eight models, and the window-selection script.

## Citation

```bibtex
@article{cannistraci2026shallow,
  title   = {Tabular Foundation Models Are Effectively Shallow},
  author  = {Irene Cannistraci and Julia E. Vogt},
  journal = {arXiv preprint arXiv:2610.13253},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.13253}
}
```
