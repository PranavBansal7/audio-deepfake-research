#!/usr/bin/env python3
"""
Cluster-bootstrap test: is the Hindi vs English asymmetry in
frequency-intervention deltas (remove_low, remove_mid) statistically real,
or noise at n=100 per language?

Now uses FULL speaker maps for both languages and both labels (real + fake):
    - English: metadata__2_.csv  (pair_id, client_id, real_path, fake_path)
    - Hindi:   hindi_pair_speaker_map.csv  (pair_id, client_id)

This replaces the earlier fallback that had no clustering for real clips.

Run:
    python exp8_freq_intervention_lang_bootstrap.py
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("phase5_outputs")
FREQ = BASE / "frequency_intervention"
OUT = FREQ

# Adjust these paths to wherever you keep the uploaded maps on your machine
ENGLISH_METADATA_PATH = Path("english_griffinlim_eval_final/metadata.csv")          # was metadata__2_.csv
HINDI_MAP_PATH = Path("hindi_pair_speaker_map.csv")

SEED = 18235
N_BOOTSTRAP = 2000
CI = 0.95
CONDITIONS = ["remove_low", "remove_mid"]


def derive_label(sample_key: str) -> str:
    sk = sample_key.lower()
    if "fake" in sk:
        return "fake"
    if "real" in sk:
        return "real"
    return "unknown"


def cluster_bootstrap_diff(en_delta, en_cluster, hi_delta, hi_cluster, seed):
    en_clusters, en_inv = np.unique(en_cluster, return_inverse=True)
    hi_clusters, hi_inv = np.unique(hi_cluster, return_inverse=True)
    n_en, n_hi = len(en_clusters), len(hi_clusters)

    observed = float(np.mean(hi_delta) - np.mean(en_delta))

    rng = np.random.default_rng(seed)
    boot = np.empty(N_BOOTSTRAP, dtype=np.float64)

    for b in range(N_BOOTSTRAP):
        en_mult = np.bincount(
            rng.integers(0, n_en, size=n_en), minlength=n_en
        ).astype(np.float64)
        hi_mult = np.bincount(
            rng.integers(0, n_hi, size=n_hi), minlength=n_hi
        ).astype(np.float64)

        en_w = en_mult[en_inv]
        hi_w = hi_mult[hi_inv]

        en_mean = np.sum(en_delta * en_w) / np.sum(en_w)
        hi_mean = np.sum(hi_delta * hi_w) / np.sum(hi_w)
        boot[b] = hi_mean - en_mean

    alpha = (1 - CI) / 2
    centered = boot - np.mean(boot)
    p_boot = float(np.mean(np.abs(centered) >= np.abs(observed)))
    p_boot = max(p_boot, 1.0 / (len(boot) + 1))

    return {
        "observed_diff_hindi_minus_english": observed,
        "ci_lo": float(np.quantile(boot, alpha)),
        "ci_hi": float(np.quantile(boot, 1 - alpha)),
        "p_bootstrap": p_boot,
        "n_en": int(len(en_delta)),
        "n_hi": int(len(hi_delta)),
        "n_en_clusters": int(n_en),
        "n_hi_clusters": int(n_hi),
        "n_bootstrap": int(len(boot)),
        "seed": int(seed),
    }


def load_english_path_to_speaker():
    meta = pd.read_csv(ENGLISH_METADATA_PATH)
    real = meta[["pair_id", "client_id", "real_path"]].rename(
        columns={"real_path": "path"}
    )
    fake = meta[["pair_id", "client_id", "fake_path"]].rename(
        columns={"fake_path": "path"}
    )
    long = pd.concat([real, fake], ignore_index=True)
    long["path"] = long["path"].astype(str)
    long["client_id"] = long["client_id"].astype(str)
    return long.drop_duplicates("path")


def load_hindi_pair_to_speaker():
    m = pd.read_csv(HINDI_MAP_PATH)
    m["pair_id"] = m["pair_id"].astype(str)
    m["client_id"] = m["client_id"].astype(str)
    return m


def main():
    df = pd.read_csv(FREQ / "frequency_intervention_scores.csv")
    df["pair_id"] = df["pair_id"].astype(str)
    df["path"] = df["path"].astype(str)
    df["derived_label"] = df["sample_key"].apply(derive_label)

    unknown = (df["derived_label"] == "unknown").sum()
    if unknown:
        print(f"WARNING: {unknown} rows had unrecognized sample_key label")

    en_path_map = load_english_path_to_speaker()
    hi_pair_map = load_hindi_pair_to_speaker()

    results = []
    seed = SEED

    for condition in CONDITIONS:
        for label in ["fake", "real"]:
            sub = df[(df.derived_label == label) & (df.condition == condition)]

            en = sub[sub.language == "english"].merge(
                en_path_map, on="path", how="inner"
            )
            hi = sub[sub.language == "hindi"].merge(
                hi_pair_map, on="pair_id", how="inner"
            )

            dropped_en = len(sub[sub.language == "english"]) - len(en)
            dropped_hi = len(sub[sub.language == "hindi"]) - len(hi)
            if dropped_en or dropped_hi:
                print(
                    f"[{condition}/{label}] unmatched to speaker map: "
                    f"en={dropped_en}, hi={dropped_hi}"
                )

            r = cluster_bootstrap_diff(
                en["delta_score"].to_numpy(float),
                en["client_id"].astype(str).to_numpy(),
                hi["delta_score"].to_numpy(float),
                hi["client_id"].astype(str).to_numpy(),
                seed,
            )
            seed += 1
            results.append(
                {"condition": condition, "label": label, "cluster_basis": "speaker", **r}
            )
            print(condition, label, "(speaker-clustered):", json.dumps(r, indent=2))

    out_df = pd.DataFrame(results)
    out_path = OUT / "frequency_intervention_lang_diff_bootstrap.csv"
    out_df.to_csv(out_path, index=False)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
