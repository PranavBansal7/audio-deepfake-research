#!/usr/bin/env python3
"""
Experiment 6 — Residual-gap analysis after acoustic control.

Question
--------
After controlling for the measured fake-audio acoustic covariates using the
Experiment-4 overlap weights, what properties remain associated with
detector difficulty among the common-support English/Hindi cohort?

Design
------
Uses:
    phase5_outputs/spectral_overlap_eer/overlap_weighted_english_scores.csv
    phase5_outputs/spectral_overlap_eer/overlap_weighted_hindi_scores.csv
    phase5_outputs/spectral_overlap_eer/fake_covariates.csv

For each fake clip:
    difficulty = bonafide_logit - frozen language-specific EER threshold

Primary predictors:
    residual spectral / acoustic variables
    attribution features from Phase-5 XAI manifests, when available
    duration / RMS / peak

The key comparison is whether these predictors explain the residual Hindi
difficulty difference AFTER the Experiment-4 overlap weighting has balanced
the measured fake-audio covariates.

Analysis:
    * weighted summary of difficulty by language
    * weighted English-vs-Hindi difference in difficulty
    * weighted Spearman correlations of difficulty with candidate features
    * speaker-cluster bootstrap CIs for each correlation
    * optional linear residual model using only common-support weighted data
      (descriptive, not causal)

This is intentionally a diagnostic / mechanism analysis, not a new EER
estimator.

Run:
    python exp6_residual_gap_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


BASE = Path("phase5_outputs")
OUT = BASE / "residual_gap_analysis"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 16235
N_BOOTSTRAP = 2000
CI = 0.95

FROZEN_THRESHOLDS = {
    "english": -0.5049806833267212,
    "hindi": 0.4493870735168457,
}

CANDIDATE_FEATURES = [
    "low_frac",
    "mid_frac",
    "high_frac",
    "gini",
    "entropy",
    "topk_1pct",
    "topk_5pct",
    "topk_10pct",
    "rms",
    "peak",
    "duration_s",
]


def weighted_mean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    return float(np.sum(x * w) / np.sum(w))


def weighted_spearman(x, y, w):
    """
    Weighted rank correlation using weighted Pearson correlation of ranks.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[mask], y[mask], w[mask]

    if len(x) < 8:
        return np.nan, np.nan, len(x)

    rx = stats.rankdata(x)
    ry = stats.rankdata(y)

    mx = np.sum(w * rx) / np.sum(w)
    my = np.sum(w * ry) / np.sum(w)

    cov = np.sum(w * (rx - mx) * (ry - my)) / np.sum(w)
    vx = np.sum(w * (rx - mx) ** 2) / np.sum(w)
    vy = np.sum(w * (ry - my) ** 2) / np.sum(w)

    rho = cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan

    # Weighted permutation-free p-value is not straightforward; use the
    # ordinary Spearman p-value on the same support as a descriptive reference.
    p = stats.spearmanr(x, y).pvalue

    return float(rho), float(p), int(len(x))


def cluster_bootstrap_weighted_spearman(df, feature, seed):
    groups = {
        str(k): g for k, g in df.groupby("analysis_cluster")
    }
    clusters = list(groups)

    rho0, _, _ = weighted_spearman(
        df["difficulty"], df[feature], df["weight"]
    )

    rng = np.random.default_rng(seed)
    vals = []

    for _ in range(N_BOOTSTRAP):
        draw = rng.choice(clusters, len(clusters), replace=True)

        pieces = []
        for c in draw:
            pieces.append(groups[c].copy())

        sample = pd.concat(pieces, ignore_index=True)
        r, _, _ = weighted_spearman(
            sample["difficulty"], sample[feature], sample["weight"]
        )
        if np.isfinite(r):
            vals.append(r)

    vals = np.asarray(vals, float)
    alpha = (1 - CI) / 2

    # Bootstrap p-value for H0: rho == 0, consistent with the same
    # weighted statistic and resampling scheme used for the CI.
    # Center the bootstrap distribution on 0 (the null), then ask how
    # extreme the observed rho0 is relative to that centered distribution.
    centered = vals - np.mean(vals)
    p_boot = float(
        np.mean(np.abs(centered) >= np.abs(rho0))
    )
    # Avoid a hard zero from finite bootstrap resolution.
    p_boot = max(p_boot, 1.0 / (len(vals) + 1))

    return {
        "rho": rho0,
        "p_bootstrap": p_boot,
        "ci_lo": float(np.quantile(vals, alpha)),
        "ci_hi": float(np.quantile(vals, 1 - alpha)),
        "n_samples": int(len(df)),
        "n_clusters": int(df.analysis_cluster.nunique()),
        "n_bootstrap": int(len(vals)),
        "bootstrap_unit": "speaker",
        "seed": int(seed),
    }

def load_data():
    en_scores = pd.read_csv(
        BASE / "spectral_overlap_eer/overlap_weighted_english_scores.csv"
    )
    hi_scores = pd.read_csv(
        BASE / "spectral_overlap_eer/overlap_weighted_hindi_scores.csv"
    )
    cov = pd.read_csv(
        BASE / "spectral_overlap_eer/fake_covariates.csv"
    )

    # Only fake utterances for difficulty analysis.
    en = en_scores[en_scores.label == 0].copy()
    hi = hi_scores[hi_scores.label == 0].copy()

    en["language"] = "english"
    hi["language"] = "hindi"

    en["difficulty"] = (
        en["bonafide_logit"] - FROZEN_THRESHOLDS["english"]
    )
    hi["difficulty"] = (
        hi["bonafide_logit"] - FROZEN_THRESHOLDS["hindi"]
    )

    # Experiment-4 covariate table has one row per pair and carries the
    # overlap weight.
    cov["pair_id"] = cov["pair_id"].astype(str)
    en["pair_id"] = en["pair_id"].astype(str)
    hi["pair_id"] = hi["pair_id"].astype(str)

    keep = [
        "language", "pair_id", "client_id", "overlap_weight",
        "low_energy_frac", "mid_energy_frac", "high_energy_frac",
        "rms", "peak", "duration_s",
    ]
    missing = [c for c in keep if c not in cov.columns]
    if missing:
        raise RuntimeError(
            "fake_covariates.csv is missing expected columns: "
            + str(missing)
        )

    cov = cov[keep].copy()
    cov["analysis_cluster"] = cov["client_id"].astype(str)

    en = en.merge(
        cov[cov.language == "english"],
        on=["language", "pair_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_cov"),
    )
    hi = hi.merge(
        cov[cov.language == "hindi"],
        on=["language", "pair_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_cov"),
    )

    en["weight"] = en["overlap_weight"]
    hi["weight"] = hi["overlap_weight"]

    # Use metadata-authoritative speaker ID if score table already has one.
    if "analysis_cluster_cov" in en.columns:
        en["analysis_cluster"] = en["analysis_cluster_cov"].astype(str)
        hi["analysis_cluster"] = hi["analysis_cluster_cov"].astype(str)
    else:
        en["analysis_cluster"] = en["client_id"].astype(str)
        hi["analysis_cluster"] = hi["client_id"].astype(str)

    return en, hi


def attach_xai_features(df, lang):
    candidates = [
        BASE / (
            "explanation_features_cv_en_fake.csv"
            if lang == "english"
            else "explanation_features_cv_hi_fake.csv"
        ),
        BASE / (
            "faithfulness_cv_en_fake.csv"
            if lang == "english"
            else "faithfulness_cv_hi_fake.csv"
        ),
    ]

    path: Optional[Path] = None
    for p in candidates:
        if p.exists():
            path = p
            break

    if path is None:
        return df, None

    x = pd.read_csv(path)
    x["pair_id"] = x["pair_id"].astype(str)

    xcols = [
        "pair_id",
        "low_frac", "mid_frac", "high_frac",
        "gini", "entropy",
        "topk_1pct", "topk_5pct", "topk_10pct",
    ]
    x = x[[c for c in xcols if c in x.columns]].drop_duplicates("pair_id")

    return df.merge(x, on="pair_id", how="left"), path


def main():
    en, hi = load_data()

    en, en_xai = attach_xai_features(en, "english")
    hi, hi_xai = attach_xai_features(hi, "hindi")

    en.to_csv(OUT / "english_common_support_fake.csv", index=False)
    hi.to_csv(OUT / "hindi_common_support_fake.csv", index=False)

    summary = {
        "experiment": "residual_gap_analysis",
        "english": {
            "n": len(en),
            "speakers": int(en.analysis_cluster.nunique()),
            "weighted_mean_difficulty": weighted_mean(en.difficulty, en.weight),
        },
        "hindi": {
            "n": len(hi),
            "speakers": int(hi.analysis_cluster.nunique()),
            "weighted_mean_difficulty": weighted_mean(hi.difficulty, hi.weight),
        },
        "weighted_mean_difference_hindi_minus_english": (
            weighted_mean(hi.difficulty, hi.weight)
            - weighted_mean(en.difficulty, en.weight)
        ),
        "xai_sources": {
            "english": str(en_xai) if en_xai else None,
            "hindi": str(hi_xai) if hi_xai else None,
        },
    }

    rows = []
    seed = SEED

    for lang, df in [("english", en), ("hindi", hi)]:
        for feature in CANDIDATE_FEATURES:
            if feature not in df.columns:
                continue

            d = df[["difficulty", feature, "analysis_cluster", "weight"]].dropna()
            d = d[d.weight > 0]

            if len(d) < 12 or d.analysis_cluster.nunique() < 8:
                continue

            r = cluster_bootstrap_weighted_spearman(
                d, feature, seed
            )
            seed += 1

            rows.append({
                "language": lang,
                "feature": feature,
                **r,
            })

    corr = pd.DataFrame(rows)
    corr.to_csv(
        OUT / "residual_difficulty_correlations.csv",
        index=False,
    )

    # Compare weighted mean difficulty directly.
    result = {
        **summary,
        "weighted_difficulty_ci": {},
        "correlation_table": str(OUT / "residual_difficulty_correlations.csv"),
    }

    (OUT / "residual_gap_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))
    print("Wrote:", OUT)


if __name__ == "__main__":
    main()
