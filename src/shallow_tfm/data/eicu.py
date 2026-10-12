"""eICU-CRD in-hospital mortality cohort, mirroring the MIMIC-III loader.

Adults with a first ICU stay of at least 48 hours; features are the
mean/std/min/max of seven vitals and eleven labs over the first 48 hours plus
age and sex (74 features, matching MIMIC-III).  Requires the eICU-CRD v2.0 CSVs
from PhysioNet under ``data/physionet/eicu-crd/2.0`` (or ``EICU_DIR``); the
extraction is cached under ``data/cache``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from shallow_tfm.data.clinical import CACHE_DIR, PHYSIONET_DIR, aggregate_long, assemble

EICU_DIR = Path(os.environ.get("EICU_DIR", PHYSIONET_DIR / "eicu-crd" / "2.0"))

VITAL_COLUMNS = ["heartrate", "systemicsystolic", "systemicdiastolic", "systemicmean",
                 "respiration", "temperature", "sao2"]

LAB_NAMES = {
    "bun": "BUN",
    "creatinine": "creatinine",
    "potassium": "potassium",
    "sodium": "sodium",
    "bicarbonate": "bicarbonate",
    "wbc": "WBC x 1000",
    "hematocrit": "Hct",
    "platelets": "platelets x 1000",
    "lactate": "lactate",
    "ph": "pH",
    "glucose": "glucose",
}

MAX_COHORT_SIZE = 30_000  # subsampled to the MIMIC-III scale


def _read_csv(eicu_dir: Path, name: str, **kwargs) -> pd.DataFrame:
    path = eicu_dir / f"{name}.csv.gz"
    if not path.exists():
        path = eicu_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}.csv(.gz) in {eicu_dir}; download eICU-CRD v2.0 "
                                "from https://physionet.org/content/eicu-crd/2.0/")
    return pd.read_csv(path, **kwargs)


def _load_patients(eicu_dir: Path, min_los_hours: int) -> pd.DataFrame:
    """Adults, first unit stay per hospitalisation, unit stay of at least ``min_los_hours``."""
    patients = _read_csv(eicu_dir, "patient")
    patients["age_numeric"] = pd.to_numeric(patients["age"], errors="coerce")
    patients.loc[patients["age"] == "> 89", "age_numeric"] = 90.0
    patients = patients.dropna(subset=["age_numeric"])
    patients = patients[patients["age_numeric"] >= 18]
    patients = patients.sort_values(["unitvisitnumber", "hospitaladmitoffset"], ascending=[True, False])
    patients = patients.groupby("patienthealthsystemstayid").first().reset_index()
    patients = patients[patients["unitdischargeoffset"] >= min_los_hours * 60]
    patients["mortality"] = (patients["hospitaldischargestatus"] == "Expired").astype(int)
    return patients


def _read_in_window(eicu_dir, name, usecols, stay_ids, offset_col, offset_max):
    chunks = []
    try:
        for chunk in _read_csv(eicu_dir, name, chunksize=1_000_000, usecols=usecols):
            chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)]
            chunk = chunk[(chunk[offset_col] >= 0) & (chunk[offset_col] <= offset_max)]
            if len(chunk):
                chunks.append(chunk)
    except (EOFError, pd.errors.ParserError) as e:
        print(f"  Warning: {name} read ended early ({e}); using {len(chunks)} chunks")
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)


def _extract_vitals(eicu_dir: Path, patients: pd.DataFrame, hours: int) -> pd.DataFrame:
    vitals = _read_in_window(eicu_dir, "vitalPeriodic", ["patientunitstayid", "observationoffset"] + VITAL_COLUMNS,
                             set(patients["patientunitstayid"]), "observationoffset", hours * 60)
    long = vitals.melt(id_vars=["patientunitstayid"], value_vars=VITAL_COLUMNS,
                       var_name="feature", value_name="value").dropna(subset=["value"])
    return aggregate_long(long, "patientunitstayid", "vital")


def _extract_labs(eicu_dir: Path, patients: pd.DataFrame, hours: int) -> pd.DataFrame:
    name_to_canonical = {v: k for k, v in LAB_NAMES.items()}
    labs = _read_in_window(eicu_dir, "lab", ["patientunitstayid", "labname", "labresultoffset", "labresult"],
                           set(patients["patientunitstayid"]), "labresultoffset", hours * 60)
    labs = labs[labs["labname"].isin(name_to_canonical)].copy()
    labs["value"] = pd.to_numeric(labs["labresult"], errors="coerce")
    labs = labs.dropna(subset=["value"])
    labs["feature"] = labs["labname"].map(name_to_canonical)
    return aggregate_long(labs, "patientunitstayid", "lab")


def _extract_raw_features(eicu_dir: Path, hours: int = 48, min_los_hours: int = 48):
    """Returns ``(X_raw with NaNs, mortality, patient ids, feature_names)``, cached."""
    stem = CACHE_DIR / f"eicu_h{hours}_los{min_los_hours}"
    cache_file, cache_names = stem.with_suffix(".npz"), stem.with_name(stem.name + "_names.json")
    if cache_file.exists() and cache_names.exists():
        print(f"Loading cached eICU features from {cache_file}")
        data = np.load(cache_file, allow_pickle=False)
        with open(cache_names) as f:
            feature_names = json.load(f)
        return data["X"], data["y"], data["pids"], feature_names

    print(f"Loading eICU-CRD from {eicu_dir}...")
    patients = _load_patients(eicu_dir, min_los_hours)
    print(f"Cohort: {len(patients)} ICU stays ({patients['mortality'].mean():.1%} mortality)")
    if len(patients) > MAX_COHORT_SIZE:
        idx = np.random.RandomState(42).choice(len(patients), MAX_COHORT_SIZE, replace=False)
        patients = patients.iloc[idx].reset_index(drop=True)
        print(f"Subsampled to {MAX_COHORT_SIZE} stays")

    print("Extracting vitals from vitalPeriodic...")
    vitals = _extract_vitals(eicu_dir, patients, hours)
    print("Extracting labs...")
    labs = _extract_labs(eicu_dir, patients, hours)

    features = patients[["patientunitstayid", "age_numeric", "mortality"]].rename(columns={"age_numeric": "age"})
    features["gender_male"] = (patients["gender"] == "Male").astype(int)
    for block in (vitals, labs):
        if len(block):
            features = features.merge(block, on="patientunitstayid", how="left")

    y = features["mortality"].values
    pids = patients["uniquepid"].astype(str).to_numpy().astype("U")
    features = features.drop(columns=["patientunitstayid", "mortality"])
    feature_names = list(features.columns)
    X = features.values.astype(np.float64)
    print(f"Extracted {X.shape[1]} features for {X.shape[0]} stays (missing rate {np.isnan(X).mean():.1%})")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, X=X, y=y, pids=pids)
    with open(cache_names, "w") as f:
        json.dump(feature_names, f)
    print(f"Cached to {cache_file}")
    return X, y, pids, feature_names


def build_mortality_dataset(eicu_dir=None, hours: int = 48, min_los_hours: int = 48, test_size: float = 0.2,
                            calibration_size: float = 0.15, seed: int = 42) -> dict:
    """In-hospital mortality (binary) in the ``load_dataset`` layout."""
    X_raw, y, pids, feature_names = _extract_raw_features(Path(eicu_dir) if eicu_dir else EICU_DIR,
                                                          hours, min_los_hours)
    return assemble("eicu", X_raw, y, pids, feature_names, 2, test_size, calibration_size, seed)
