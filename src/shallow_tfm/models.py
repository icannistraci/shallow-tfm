"""Uniform access to the eight tabular foundation models.

Each model exposes its transformer stack under a different attribute path and
wraps it in a different sklearn-style estimator.  ``fit_model`` hides that
behind one call returning a ``ModelHandle``: the block list to patch, a
callable that triggers a forward pass, and the device the blocks live on.
"""

from __future__ import annotations

import functools
import importlib
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from shallow_tfm import CONFIG_DIR, PROJECT_ROOT
from shallow_tfm.config import MODEL_CONFIGS

LIMIX_REPO = PROJECT_ROOT / "third_party" / "LimiX"


@dataclass
class ModelHandle:
    """Everything an experiment needs from a fitted foundation model."""

    name: str
    clf: object                 #: The fitted sklearn-style estimator.
    model: nn.Module            #: The underlying torch module.
    layers: nn.ModuleList       #: Transformer blocks, patched in place when skipping.
    predict_fn: Callable        #: X -> probabilities (or predictions, for regressors).
    device: torch.device

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def cfg(self) -> dict:
        return MODEL_CONFIGS[self.name]


def _package_major(package: str) -> int:
    mod = importlib.import_module(package)
    return int(str(getattr(mod, "__version__", "0")).split(".")[0])


def _ens_kwargs() -> dict:
    """Prediction-time ensemble size for the models that expose it.

    A single member is used so that per-block extraction sees exactly one
    forward pass.  ``N_ESTIMATORS`` overrides it; the value ``default`` omits
    the argument so the library default applies.
    """
    v = os.environ.get("N_ESTIMATORS", "1")
    return {} if v == "default" else {"n_estimators": int(v)}


def _resolve_layers(model: nn.Module, name: str) -> nn.ModuleList:
    layers = model
    for attr in MODEL_CONFIGS[name]["layers_path"].split("."):
        layers = getattr(layers, attr)
    return layers


def _check_classes(name: str, y) -> None:
    cap = MODEL_CONFIGS[name].get("max_classes")
    n_classes = len(np.unique(y))
    if cap is not None and n_classes > cap:
        raise ValueError(f"{name} supports at most {cap} classes, got {n_classes}")


# ---------------------------------------------------------------------------
# Per-model constructors: (X, y, device, regression) -> (estimator, module, layers)
# ---------------------------------------------------------------------------


def _fit_tabpfn(X, y, device, regression=False):
    """TabPFNv2 under tabpfn 2.x.  The v2 code path under tabpfn>=8 is not
    numerically identical, so refuse rather than drift."""
    if _package_major("tabpfn") >= 8:
        raise RuntimeError("TabPFNv2 rows need tabpfn 2.x; this environment has tabpfn>=8 "
                           "(use the 'tabpfn3' model here, or the base environment for 'tabpfn').")
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    clf = (TabPFNRegressor if regression else TabPFNClassifier)(
        device=device, ignore_pretraining_limits=True, **_ens_kwargs())
    clf.fit(X, y)
    return clf, clf.model_, _resolve_layers(clf.model_, "tabpfn")


def _fit_tabpfn3(X, y, device, regression=False):
    """TabPFN-3 via tabpfn>=8 (``model_path="auto"`` resolves to v3; the weights
    are license-gated, see the README).  ``TABPFN_MODEL_PATH`` pins a checkpoint."""
    if _package_major("tabpfn") < 8:
        raise RuntimeError("'tabpfn3' needs tabpfn>=8 (see environments/tabpfn3.txt).")
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    model_path = os.environ.get("TABPFN_MODEL_PATH", "auto")
    clf = (TabPFNRegressor if regression else TabPFNClassifier)(
        device=device, ignore_pretraining_limits=True, model_path=model_path, **_ens_kwargs())
    clf.fit(X, y)
    return clf, clf.model_, _resolve_layers(clf.model_, "tabpfn3")


def _fit_tabicl(X, y, device, regression=False):
    # TabICL orders its ensemble members from a Python set, so which members
    # form the first batch (the one the translator is fitted on) depends on
    # PYTHONHASHSEED.  The Makefile pins it to 0, the value used for all
    # released results.
    from tabicl import TabICLClassifier, TabICLRegressor
    clf = (TabICLRegressor if regression else TabICLClassifier)(device=device)
    clf.fit(X, y)
    return clf, clf.model_, _resolve_layers(clf.model_, "tabicl")


def _fit_tabdpt(X, y, device, regression=False):
    from tabdpt import TabDPTClassifier, TabDPTRegressor
    clf = (TabDPTRegressor if regression else TabDPTClassifier)(device=device)
    clf.fit(X, y)
    return clf, clf.model, _resolve_layers(clf.model, "tabdpt")


def _fit_tabfm(X, y, device, regression=False):
    from tabfm import TabFMClassifier, TabFMRegressor
    from tabfm.src.pytorch import tabfm_v1_0_0
    if not regression:
        _check_classes("tabfm", y)
    net = tabfm_v1_0_0.load("regression" if regression else "classification", device=device)
    # TabFM shuffles features with its own random_state, not the global
    # generator, so pass on the seed that fit_model set.
    clf = (TabFMRegressor if regression else TabFMClassifier)(
        model=net, random_state=torch.initial_seed() % (2**31), **_ens_kwargs())
    clf.fit(X, y)
    return clf, net, _resolve_layers(net, "tabfm")


def _fit_orion(name: str):
    """Orion-MSP / Orion-BiX: same block layout as TabICL, classification only."""
    def build(X, y, device, regression=False):
        if regression:
            raise ValueError(f"{name} has no regressor")
        _check_classes(name, y)
        mod = importlib.import_module(name)
        Estimator = getattr(mod, "OrionMSPClassifier" if name == "orion_msp" else "OrionBixClassifier")
        clf = Estimator(device=device, **_ens_kwargs())
        clf.fit(X, y)
        return clf, clf.model_, _resolve_layers(clf.model_, name)
    return build


class LimiXEstimator:
    """sklearn-style facade over ``LimiXPredictor`` with the input preprocessing
    fitted once on the support rows and frozen.

    The stock predictor refits every preprocessing step on the support rows
    concatenated with the query rows of each ``predict`` call (categorical
    detection, categorical encoding, constant-column filter), so the column
    count and order can differ between the calibration pass and the evaluation
    pass.  A per-feature translator indexes columns by position and needs the
    same layout in both passes, so every step is fitted on the support rows
    alone and later calls only transform the query rows.  Regression targets
    are standardised on the support and de-standardised on the way out, as in
    the upstream regression path.
    """

    def __init__(self, clf, X, y, regression):
        import copy
        from itertools import chain, repeat

        from inference.inference_method import InferenceAttentionMap
        from inference.preprocess import CategoricalFeatureEncoder, RebalanceFeatureDistribution, SubSampleData
        from sklearn.preprocessing import LabelEncoder

        self.clf, self.regression = clf, regression
        X, y = np.asarray(X), np.asarray(y)
        if regression:
            self._ym, self._ys = float(y.mean()), float(y.std()) + 1e-12
            y = (y - self._ym) / self._ys
        c = clf
        assert not c.mask_prediction and not getattr(c, "inference_with_DDP", False)
        x_tr, y_tr = c.validate_data(X, y, reset=True, validate_separately=False,
                                     accept_sparse=False, dtype=None, ensure_all_finite=False)
        x_tr = c.convert_category2num(c.convert_x_dtypes(x_tr))
        if not regression:
            x_tr = x_tr.astype(np.float32)
            c.label_encoder = LabelEncoder()
            y_enc = c.label_encoder.fit_transform(y_tr)
            c.classes = c.label_encoder.classes_
            c.n_classes = len(c.classes)
            # Class permutations exactly as LimiXPredictor._predict_cls draws them.
            np_rng = np.random.default_rng(c.seed)
            noise = np_rng.random((c.n_estimators * c.class_shuffle_factor, c.n_classes))
            uniqs = np.unique(np.argsort(noise, axis=1), axis=0)
            perms = list(chain.from_iterable(repeat(e, c.n_estimators // len(uniqs)) for e in uniqs))
            cout = c.n_estimators % len(uniqs)
            if cout > 0:
                perms += [uniqs[i] for i in np_rng.choice(len(uniqs), size=cout)]
            c.class_permutations = perms
        cat_idx = c.get_categorical_features_indices(x_tr)
        self._pipes = []
        for id_pipe, pipe in enumerate(c.preprocess_pipelines):
            steps = copy.deepcopy(pipe)
            x_, cat_ = x_tr.copy(), list(cat_idx)
            y_ = np.asarray(y_tr, dtype=np.float64).copy() if regression else c.class_permutations[id_pipe][y_enc.copy()]
            for id_step, step in enumerate(steps):
                assert not isinstance(step, (InferenceAttentionMap, SubSampleData)), \
                    "frozen preprocessing: retrieval configs are not supported"
                seed = c.seeds[id_pipe * c.preprocess_num + id_step]
                if isinstance(step, RebalanceFeatureDistribution):
                    cat_ = step.fit(x_, cat_, seed)
                    x_, cat_ = step.transform(x_)
                else:
                    x_, cat_ = step.fit_transform(x_, cat_, seed)
            self._pipes.append((steps, np.asarray(x_, dtype=np.float32), y_))
        self._cat_cls = CategoricalFeatureEncoder

    @staticmethod
    def _cat_transform(step, x):
        """``CategoricalFeatureEncoder`` has no ``transform``: apply its fitted column
        transformer and the stored per-column category permutation."""
        if step.transformer is None:
            return x
        xt = np.asarray(step.transformer.transform(x), dtype=np.float64)
        for col_ix, perm in (step.category_mappings or {}).items():
            col = xt[:, col_ix]
            valid = ~np.isnan(col)
            col[valid] = perm[col[valid].astype(int)].astype(col.dtype)
        return xt

    def _run(self, X_test):
        c = self.clf
        x_te = c.validate_data(np.asarray(X_test), reset=False, validate_separately=False,
                               accept_sparse=False, dtype=None, ensure_all_finite=False)
        x_te = c.convert_category2num(c.convert_x_dtypes(x_te))
        if not self.regression:
            x_te = x_te.astype(np.float32)
        outputs = []
        for id_pipe, (steps, x_tr_pre, y_) in enumerate(self._pipes):
            x_ = x_te.copy()
            for step in steps:
                x_ = self._cat_transform(step, x_) if isinstance(step, self._cat_cls) else step.transform(x_)[0]
            x_all = np.concatenate([x_tr_pre, np.asarray(x_, dtype=np.float32)], axis=0)
            x_t = torch.from_numpy(x_all).float().to(c.device).unsqueeze(0)
            y_t = torch.from_numpy(y_).float().to(c.device).unsqueeze(0)
            torch.manual_seed(c.seed)
            torch.cuda.manual_seed_all(c.seed)
            c.model.to(c.device)
            with torch.autocast(device_type=c.device.type, enabled=c.mix_precision), torch.inference_mode():
                if self.regression:
                    outputs.append(c.model(x=x_t, y=y_t, eval_pos=y_t.shape[1], task_type="reg").squeeze(0))
                    continue
                out = c.model(x=x_t, y=y_t, eval_pos=y_t.shape[1], task_type="cls")
                out = out if isinstance(out, dict) else out.squeeze(0)
                if c.softmax_temperature != 1:
                    out = out[:, :c.n_classes].float() / c.softmax_temperature
                out = out[..., c.class_permutations[id_pipe]]
            outputs.append(out)
        if self.regression:
            out = torch.stack(outputs).squeeze(2).mean(dim=0).cpu().numpy()
            return out.astype(np.float64) * self._ys + self._ym
        out = torch.stack([torch.nn.functional.softmax(o, dim=1) for o in outputs]).mean(dim=0)
        out = out.float().cpu().numpy()
        return out / out.sum(axis=1, keepdims=True)

    def predict_proba(self, X_test):
        return self._run(X_test)

    def predict(self, X_test):
        out = self._run(X_test)
        return out if self.regression else out.argmax(axis=1)


def _fit_limix(name: str):
    """LimiX from the repo checkout under ``third_party/LimiX``, with the
    single-member inference config so extraction sees exactly one pass."""
    def build(X, y, device, regression=False):
        if str(LIMIX_REPO) not in sys.path:
            sys.path.insert(0, str(LIMIX_REPO))
        from huggingface_hub import hf_hub_download
        from inference.predictor import LimiXPredictor
        cfg = MODEL_CONFIGS[name]
        if not regression:
            _check_classes(name, y)
        model_path = hf_hub_download(cfg["hf_repo"], cfg["ckpt_file"])
        inference_config = str(CONFIG_DIR / f"limix_{'reg' if regression else 'cls'}_single.json")
        clf = LimiXPredictor(device=torch.device(device), model_path=model_path,
                             inference_config=inference_config)
        return LimiXEstimator(clf, X, y, regression), clf.model, _resolve_layers(clf.model, name)
    return build


_BUILDERS = {
    "tabpfn": _fit_tabpfn,
    "tabpfn3": _fit_tabpfn3,
    "tabicl": _fit_tabicl,
    "tabdpt": _fit_tabdpt,
    "tabfm": _fit_tabfm,
    "orion_msp": _fit_orion("orion_msp"),
    "orion_bix": _fit_orion("orion_bix"),
    "limix_16m_pf": _fit_limix("limix_16m_pf"),
}


def fit_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: str = "cpu",
    seed: int | None = None,
    regression: bool = False,
) -> ModelHandle:
    """Fit ``name`` on the in-context support set and return a ``ModelHandle``.

    Seeding is done here: ICL inference draws on the global torch RNG, so the
    same (model, dataset, seed) drifts between runs unless the RNG state is
    fixed at construction.
    """
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model {name!r}; choose from {list(_BUILDERS)}")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    clf, model, layers = _BUILDERS[name](X_train, y_train, device, regression=regression)
    predict_fn = clf.predict if regression else clf.predict_proba
    if name == "tabdpt":
        # Above 2048 support rows TabDPT's predict silently switches to
        # per-batch kNN retrieval; force the single full-context forward so a
        # context of N rows means N rows.
        predict_fn = functools.partial(predict_fn, context_size=None)
    return ModelHandle(
        name=name,
        clf=clf,
        model=model,
        layers=layers,
        predict_fn=predict_fn,
        device=next(model.parameters()).device,
    )


def is_available(name: str) -> bool:
    """True if the package backing ``name`` is importable in this environment."""
    cfg = MODEL_CONFIGS[name]
    if cfg["package"] == "limix":
        try:
            import kditransform  # noqa: F401  (LimiX hard dependency)
        except ImportError:
            return False
        return (LIMIX_REPO / "inference" / "predictor.py").exists()
    try:
        major = _package_major(cfg["package"])
    except ImportError:
        return False
    if "min_package_major" in cfg and major < cfg["min_package_major"]:
        return False
    if "max_package_major" in cfg and major > cfg["max_package_major"]:
        return False
    return True
