"""Tabular foundation models are effectively shallow: training-free replacement
of a window of transformer blocks by a closed-form affine translator."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"
#: Override with SHALLOW_TFM_RESULTS_DIR to keep a run apart from the main results.
RESULTS_DIR = Path(os.environ.get("SHALLOW_TFM_RESULTS_DIR", PROJECT_ROOT / "results"))
