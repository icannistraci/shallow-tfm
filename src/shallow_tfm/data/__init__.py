"""Dataset loaders: the eight clinical datasets listed in ``datasets.py``, the
OpenML suite datasets listed in ``data/dataset_manifest.csv`` and the two
PhysioNet cohorts."""

from shallow_tfm.data.datasets import DATASETS, load_dataset
from shallow_tfm.data.suites import register_suite_datasets, suite_datasets

__all__ = ["DATASETS", "load_dataset", "register_suite_datasets", "suite_datasets"]
