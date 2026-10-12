"""Model architecture facts, dataset groups and experiment settings."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
#
#   display_name       Name used in tables and figures.
#   package            Import name; the model counts as installed when it imports.
#   n_layers           Number of transformer blocks.
#   d_model            Block hidden width.
#   layers_path        Dotted path from the torch module to the block ModuleList.
#   max_train_samples  Cap on the in-context support set.
#   max_cal_samples    Cap on the calibration set the translator is fitted on.
#   max_classes        Class cap the model's classifier enforces (optional).
#   per_feature        Dual-axis attention: one affine map per feature index
#                      instead of a single shared (W, b).
#   return_tuple_size  Length of the tuple a block returns (0 = plain tensor).
#   output_element     Which element of a tuple-valued block output is the state.
#   pad_tuple_none     Blocks are called with one argument but the caller unpacks
#                      a fixed-size tuple; the translator pads with None.
#   per_pass_translator  The model handles at most ten classes in one run of the
#                      transformer; with more it runs it several times for a
#                      single prediction, and each run gets its own translator.

MODEL_CONFIGS: dict[str, dict] = {
    "tabpfn": {
        "display_name": "TabPFNv2",
        "package": "tabpfn",
        "max_package_major": 7,
        "n_layers": 12,
        "d_model": 192,
        "layers_path": "transformer_encoder.layers",
        "max_train_samples": 3000,
        "max_cal_samples": 500,
        "per_feature": True,
        "return_tuple_size": 0,
        "output_element": None,
    },
    "tabpfn3": {
        "display_name": "TabPFN-3",
        "package": "tabpfn",
        "min_package_major": 8,
        "n_layers": 24,
        "d_model": 512,
        "layers_path": "icl_blocks",
        "max_train_samples": 3000,
        "max_cal_samples": 500,
        "per_feature": False,
        "return_tuple_size": 2,
        "output_element": 0,
    },
    "tabicl": {
        "display_name": "TabICL",
        "package": "tabicl",
        "n_layers": 12,
        "d_model": 512,
        "layers_path": "icl_predictor.tf_icl.blocks",
        "max_train_samples": 3000,
        "max_cal_samples": 500,
        "per_feature": False,
        "return_tuple_size": 0,
        "output_element": None,
        "per_pass_translator": True,
    },
    "tabdpt": {
        "display_name": "TabDPT",
        "package": "tabdpt",
        "n_layers": 16,
        "d_model": 768,
        "layers_path": "transformer_encoder",
        "max_train_samples": 1500,
        "max_cal_samples": 500,
        "per_feature": False,
        "return_tuple_size": 0,
        "output_element": None,
        "per_pass_translator": True,
    },
    "tabfm": {
        "display_name": "TabFM",
        "package": "tabfm",
        "n_layers": 24,
        "d_model": 2048,
        "layers_path": "icl_predictor.tf_icl.blocks",
        "max_train_samples": 3000,
        "max_cal_samples": 500,
        "max_classes": 10,
        "per_feature": False,
        "return_tuple_size": 0,
        "output_element": None,
    },
    "orion_msp": {
        "display_name": "Orion-MSP",
        "package": "orion_msp",
        "n_layers": 12,
        "d_model": 512,
        "layers_path": "icl_predictor.tf_icl.blocks",
        "max_train_samples": 3000,
        "max_cal_samples": 500,
        "max_classes": 10,
        "per_feature": False,
        "return_tuple_size": 0,
        "output_element": None,
    },
    "orion_bix": {
        "display_name": "Orion-BiX",
        "package": "orion_bix",
        "n_layers": 12,
        "d_model": 512,
        "layers_path": "icl_predictor.tf_icl.blocks",
        "max_train_samples": 3000,
        "max_cal_samples": 500,
        "max_classes": 10,
        "per_feature": False,
        "return_tuple_size": 0,
        "output_element": None,
    },
    # LimiX with its input preprocessing fitted once on the support rows
    # (models.LimiXEstimator), so the columns are the same in the calibration
    # and test passes and a per-feature map is valid.  The key keeps the
    # suffix used in the result files.
    "limix_16m_pf": {
        "display_name": "LimiX-16M",
        "package": "limix",
        "hf_repo": "stable-ai/LimiX-16M",
        "ckpt_file": "LimiX-16M.ckpt",
        "n_layers": 12,
        "d_model": 192,
        "layers_path": "transformer_encoder.layers",
        "max_train_samples": 3000,
        "max_cal_samples": 500,
        "max_classes": 10,
        "per_feature": True,
        "return_tuple_size": 3,
        "output_element": 0,
        "pad_tuple_none": True,
    },
}

ALL_MODELS = list(MODEL_CONFIGS)

#: The fitted translator every reported row uses.
TRANSLATOR = "ridge"

# ---------------------------------------------------------------------------
# Datasets and seeds
# ---------------------------------------------------------------------------

CLINICAL_BINARY = ["diabetes", "heart", "breast_cancer", "support2", "mimic", "eicu"]
CLINICAL_MULTICLASS = ["cardiotocography", "thyroid"]

#: The eight clinical datasets; the OpenML suite datasets are listed in data/dataset_manifest.csv.
PAPER_DATASETS = CLINICAL_BINARY + CLINICAL_MULTICLASS

#: Every experiment runs, and every table aggregates over, these seeds.
SEEDS = [0, 1, 2]

DISPLAY_NAMES = {
    "diabetes": "Diabetes",
    "heart": "Heart",
    "breast_cancer": "Br. Cancer",
    "support2": "SUPPORT2",
    "mimic": "MIMIC",
    "eicu": "eICU",
    "cardiotocography": "CTG",
    "thyroid": "Thyroid",
}

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

#: Maximum number of sequence positions a translator is fitted on.  The
#: calibration forward pass also yields the support rows' activations, so a
#: fixed random subset of at most this many positions is used (the same
#: subset at the source and target depth).  None disables the cap.
TRANSLATOR_FIT_POSITIONS: int | None = 500

#: Datasets above the models' feature ceiling are reduced to the top-K
#: columns by ANOVA F-score computed on the training split, identically for
#: every model and baseline of a given (dataset, seed).
MAX_FEATURES = 500
