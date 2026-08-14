#!/usr/bin/env python3
"""
Fast continuation for Experiment 6.

Reads:
    phase5_outputs/residual_gap_analysis/english_common_support_fake.csv
    phase5_outputs/residual_gap_analysis/hindi_common_support_fake.csv

Does NOT rerun audio loading, model scoring, or XAI.

It recomputes the speaker-cluster bootstrap correlations using NumPy arrays
instead of repeated pandas concatenation, then writes the missing final
Experiment-6 outputs.

Run:
    python finish_exp6_residual_gap.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BASE = Path("phase5_outputs/residual_gap_analysis")
OUT = BASE

SEED = 16235
N_BOOTSTRAP = 2000
CI = 0.95

FEATURES = [
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
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x = x[mask]
    y = y[mask]
    w = w[mask]

    if len(x) < 8:
        return np.nan

    rx = stats.rankdata(x)
    ry = stats.rankdata(y)

    sw = np.sum(w)
    mx = np.sum(w * rx) / sw
    my = np.sum(w * ry) / sw

    cov = np.sum(w * (rx - mx) * (ry - my)) / sw
    vx = np.sum(w * (rx - mx) ** 2) / sw
    vy = np.sum(w * (ry - my) ** 2) / sw

    if vx <= 0 or vy <= 0:
        return np.nan

    return float(cov / np.sqrt(vx * vy))


def prepare(df, feature):
    d = df[
        ["analysis_cluster", "difficulty", "weight", feature]
    ].replace([np.inf, -np.inf], np.nan).dropna()

    d = d[d["weight"] > 0].copy()

    clusters = d["analysis_cluster"].astype(str).to_numpy()
    unique_clusters, inverse = np.unique(clusters, return_inverse=True)

    return (
        d["difficulty"].to_numpy(float),
        d[feature].to_numpy(float),
        d["weight"].to_numpy(float),
        inverse.astype(np.int32),
        len(unique_clusters),
    )


def bootstrap_corr(df, feature, seed):
    x, y, w, inverse, n_clusters = prepare(df, feature)

    if len(x) < 12 or n_clusters < 8:
        return None

    observed = weighted_spearman(x, y, w)

    rng = np.random.default_rng(seed)
    boot = np.empty(N_BOOTSTRAP, dtype=np.float64)

    for b in range(N_BOOTSTRAP):
        multiplicity = np.bincount(
            rng.integers(0, n_clusters, size=n_clusters),
            minlength=n_clusters,
        ).astype(np.float64)

        wb = w * multiplicity[inverse]
        boot[b] = weighted_spearman(x, y, wb)

    boot = boot[np.isfinite(boot)]
    alpha = (1.0 - CI) / 2.0

    # Bootstrap p-value: center the bootstrap distribution on its own mean
    # (approximating the null of rho=0 under the same resampling scheme),
    # then ask how extreme the observed rho is relative to that.
    centered = boot - np.mean(boot)
    p_boot = float(np.mean(np.abs(centered) >= np.abs(observed)))
    p_boot = max(p_boot, 1.0 / (len(boot) + 1))

    return {
        "rho": float(observed),
        "p_bootstrap": p_boot,
        "ci_lo": float(np.quantile(boot, alpha)),
        "ci_hi": float(np.quantile(boot, 1.0 - alpha)),
        "n_samples": int(len(x)),
        "n_clusters": int(n_clusters),
        "n_bootstrap": int(len(boot)),
        "bootstrap_unit": "speaker",
        "seed": int(seed),
    }


def main():
    en_path = OUT / "english_common_support_fake.csv"
    hi_path = OUT / "hindi_common_support_fake.csv"

    if not en_path.exists() or not hi_path.exists():
        raise FileNotFoundError(
            "Expected the already-created common-support files:\n"
            f"  {en_path}\n"
            f"  {hi_path}"
        )

    en = pd.read_csv(en_path)
    hi = pd.read_csv(hi_path)

    print("English common-support rows:", len(en))
    print("Hindi common-support rows  :", len(hi))
    print("English speakers:", en.analysis_cluster.nunique())
    print("Hindi speakers  :", hi.analysis_cluster.nunique())

    rows = []
    seed = SEED

    for language, df in [("english", en), ("hindi", hi)]:
        print(f"\n=== {language.upper()} ===")

        for feature in FEATURES:
            if feature not in df.columns:
                continue

            print(f"  {feature} ...", end="", flush=True)
            result = bootstrap_corr(df, feature, seed)
            seed += 1

            if result is None:
                print(" skipped")
                continue

            print(
                f" rho={result['rho']:.4f} "
                f"CI=[{result['ci_lo']:.4f}, {result['ci_hi']:.4f}]"
            )

            rows.append({
                "language": language,
                "feature": feature,
                **result,
            })

    corr = pd.DataFrame(rows)
    corr.to_csv(
        OUT / "residual_difficulty_correlations.csv",
        index=False,
    )

    # Common-support weighted difficulty summary.
    summary = {
        "experiment": "residual_gap_analysis",
        "english": {
            "n": int(len(en)),
            "speakers": int(en.analysis_cluster.nunique()),
            "weighted_mean_difficulty": weighted_mean(
                en["difficulty"], en["weight"]
            ),
            "weighted_median_unweighted_reference": float(
                en["difficulty"].median()
            ),
        },
        "hindi": {
            "n": int(len(hi)),
            "speakers": int(hi.analysis_cluster.nunique()),
            "weighted_mean_difficulty": weighted_mean(
                hi["difficulty"], hi["weight"]
            ),
            "weighted_median_unweighted_reference": float(
                hi["difficulty"].median()
            ),
        },
        "weighted_mean_difference_hindi_minus_english": (
            weighted_mean(hi["difficulty"], hi["weight"])
            - weighted_mean(en["difficulty"], en["weight"])
        ),
        "correlation_table": str(
            OUT / "residual_difficulty_correlations.csv"
        ),
        "bootstrap_unit": "speaker",
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
    }

    (OUT / "residual_gap_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nWrote:")
    print(OUT / "residual_difficulty_correlations.csv")
    print(OUT / "residual_gap_summary.json")


if __name__ == "__main__":
    main()
