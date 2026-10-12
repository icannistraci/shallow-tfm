"""MIMIC-III in-hospital mortality and length-of-stay cohorts.

Follows the Harutyunyan et al. (2019) extraction: adults with a first ICU stay
of at least 48 hours; features are the mean/std/min/max of eight vitals and
ten labs over the first 48 hours plus age and sex.  Requires the MIMIC-III v1.4
CSVs from PhysioNet under ``data/physionet/mimiciii/1.4`` (or ``MIMIC_DIR``);
the extraction is cached under ``data/cache``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from shallow_tfm.data.clinical import CACHE_DIR, PHYSIONET_DIR, aggregate_long, assemble

MIMIC_DIR = Path(os.environ.get("MIMIC_DIR", PHYSIONET_DIR / "mimiciii" / "1.4"))

VITAL_ITEMIDS = {
    "heart_rate": [211, 220045],
    "sbp": [51, 442, 455, 6701, 220179, 220050],
    "dbp": [8368, 8440, 8441, 8555, 220180, 220051],
    "mbp": [456, 52, 6702, 443, 220052, 220181, 225312],
    "resp_rate": [615, 618, 220210, 224690],
    "temperature": [223761, 678, 223762, 676],
    "spo2": [646, 220277],
    "glucose": [807, 811, 1529, 3745, 3744, 225664, 220621, 226537],
}

LAB_ITEMIDS = {
    "bun": [51006],
    "creatinine": [50912],
    "potassium": [50822, 50971],
    "sodium": [50824, 50983],
    "bicarbonate": [50882],
    "wbc": [51300, 51301],
    "hematocrit": [51221],
    "platelets": [51265],
    "lactate": [50813],
    "ph": [50820],
}


def _read_csv(mimic_dir: Path, name: str, **kwargs) -> pd.DataFrame:
    path = mimic_dir / f"{name}.csv.gz"
    if not path.exists():
        path = mimic_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}.csv(.gz) in {mimic_dir}; download MIMIC-III v1.4 "
                                "from https://physionet.org/content/mimiciii/1.4/")
    return pd.read_csv(path, **kwargs)


def _events_in_window(mimic_dir, name, itemids, id_col, icustays, hours):
    """Numeric events of ``itemids`` within the first ``hours`` of each ICU stay,
    in long format with a canonical ``feature`` name."""
    id_to_name = {item: feat for feat, items in itemids.items() for item in items}
    chunks = []
    try:
        for chunk in _read_csv(mimic_dir, name, chunksize=1_000_000,
                               usecols=[id_col, "ITEMID", "CHARTTIME", "VALUENUM"], parse_dates=["CHARTTIME"]):
            chunk = chunk[chunk["ITEMID"].isin(id_to_name)].dropna(subset=["VALUENUM"])
            chunks.append(chunk)
    except (EOFError, pd.errors.ParserError) as e:
        print(f"  Warning: {name} read ended early ({e}); using {len(chunks)} chunks")
    if not chunks:
        return pd.DataFrame(columns=["ICUSTAY_ID", "feature", "value"])
    events = pd.concat(chunks, ignore_index=True)
    stay_cols = ["ICUSTAY_ID", "INTIME"] if id_col == "ICUSTAY_ID" else ["HADM_ID", "ICUSTAY_ID", "INTIME"]
    events = events.merge(icustays[stay_cols], on=id_col, how="inner")
    hours_in = (events["CHARTTIME"] - events["INTIME"]).dt.total_seconds() / 3600
    events = events[(hours_in >= 0) & (hours_in <= hours)]
    events["feature"] = events["ITEMID"].map(id_to_name)
    return events.rename(columns={"VALUENUM": "value"})


def _extract_raw_features(mimic_dir: Path, hours: int = 48, min_los_hours: int = 48):
    """Returns ``(X_raw with NaNs, mortality, los_hours, patient ids, feature_names)``, cached."""
    stem = CACHE_DIR / f"mimic_h{hours}_los{min_los_hours}"
    cache_file, cache_names = stem.with_suffix(".npz"), stem.with_name(stem.name + "_names.json")
    if cache_file.exists() and cache_names.exists():
        print(f"Loading cached MIMIC features from {cache_file}")
        data = np.load(cache_file, allow_pickle=False)
        with open(cache_names) as f:
            feature_names = json.load(f)
        return data["X"], data["y"], data["los_hours"], data["pids"], feature_names

    print(f"Loading MIMIC-III from {mimic_dir}...")
    admissions = _read_csv(mimic_dir, "ADMISSIONS", parse_dates=["ADMITTIME", "DISCHTIME", "DEATHTIME"])
    admissions["mortality"] = (~admissions["DEATHTIME"].isna()).astype(int)
    patients = _read_csv(mimic_dir, "PATIENTS", parse_dates=["DOB"])
    icustays = _read_csv(mimic_dir, "ICUSTAYS", parse_dates=["INTIME", "OUTTIME"])
    icustays = icustays.sort_values("INTIME").groupby("HADM_ID").first().reset_index()  # first ICU stay per admission

    cohort = icustays.merge(admissions[["HADM_ID", "ADMITTIME", "mortality"]], on="HADM_ID")
    cohort = cohort.merge(patients[["SUBJECT_ID", "GENDER", "DOB"]], on="SUBJECT_ID")
    # Year difference avoids datetime overflow on the shifted DOBs of patients over 89.
    cohort["age"] = (cohort["ADMITTIME"].dt.year - cohort["DOB"].dt.year).astype(float).clip(upper=90)
    cohort = cohort[cohort["age"] >= 18]
    cohort["los_hours"] = (cohort["OUTTIME"] - cohort["INTIME"]).dt.total_seconds() / 3600
    cohort = cohort[cohort["los_hours"] >= min_los_hours]
    print(f"Cohort: {len(cohort)} ICU stays ({cohort['mortality'].mean():.1%} mortality)")

    print("Extracting vitals (CHARTEVENTS is large)...")
    vitals = aggregate_long(_events_in_window(mimic_dir, "CHARTEVENTS", VITAL_ITEMIDS, "ICUSTAY_ID", icustays, hours),
                            "ICUSTAY_ID", "vital")
    print("Extracting labs...")
    labs = aggregate_long(_events_in_window(mimic_dir, "LABEVENTS", LAB_ITEMIDS, "HADM_ID", icustays, hours),
                          "ICUSTAY_ID", "lab")

    features = cohort[["ICUSTAY_ID", "age", "GENDER", "mortality"]].copy()
    features["gender_male"] = (features["GENDER"] == "M").astype(int)
    features = features.drop(columns=["GENDER"])
    for block in (vitals, labs):
        if len(block):
            features = features.merge(block, on="ICUSTAY_ID", how="left")

    y = features["mortality"].values
    los_hours = cohort["los_hours"].values
    pids = cohort["SUBJECT_ID"].astype(str).to_numpy().astype("U")
    features = features.drop(columns=["ICUSTAY_ID", "mortality"])
    feature_names = list(features.columns)
    X = features.values.astype(np.float64)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, X=X, y=y, los_hours=los_hours, pids=pids)
    with open(cache_names, "w") as f:
        json.dump(feature_names, f)
    print(f"Cached to {cache_file}")
    return X, y, los_hours, pids, feature_names


def build_mortality_dataset(mimic_dir=None, hours: int = 48, min_los_hours: int = 48, test_size: float = 0.2,
                            calibration_size: float = 0.15, seed: int = 42) -> dict:
    """In-hospital mortality (binary) in the ``load_dataset`` layout."""
    X_raw, y, _, pids, feature_names = _extract_raw_features(Path(mimic_dir) if mimic_dir else MIMIC_DIR,
                                                             hours, min_los_hours)
    return assemble("mimic", X_raw, y, pids, feature_names, 2, test_size, calibration_size, seed)


def build_los_dataset(mimic_dir=None, hours: int = 48, min_los_hours: int = 48, test_size: float = 0.2,
                      calibration_size: float = 0.15, seed: int = 42) -> dict:
    """ICU length of stay (regression on ``log1p(hours)``), same cohort and features."""
    X_raw, _, los_hours, pids, feature_names = _extract_raw_features(Path(mimic_dir) if mimic_dir else MIMIC_DIR,
                                                                     hours, min_los_hours)
    y = np.log1p(los_hours).astype(np.float32)
    return assemble("mimic_los", X_raw, y, pids, feature_names, 0, test_size, calibration_size, seed)
