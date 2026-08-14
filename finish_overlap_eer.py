#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("phase5_outputs/spectral_overlap_eer")

N_BOOTSTRAP = 2000
CI = 0.95
SEED = 10435   # 4235 + 6200


def weighted_eer(scores):
    s = scores.sort_values("bonafide_logit", ascending=False).reset_index(drop=True)

    y = s["label"].to_numpy(np.int8)
    w = s["weight"].to_numpy(np.float64)

    total_b = w[y == 1].sum()
    total_s = w[y == 0].sum()

    cum_b = np.cumsum(np.where(y == 1, w, 0.0))
    cum_s = np.cumsum(np.where(y == 0, w, 0.0))

    fnr = (total_b - cum_b) / total_b
    fpr = cum_s / total_s

    i = np.argmin(np.abs(fnr - fpr))

    return float((fnr[i] + fpr[i]) / 2.0)


def prepare(scores):
    s = scores.sort_values("bonafide_logit", ascending=False).reset_index(drop=True)

    labels = s["label"].to_numpy(np.int8)
    weights = s["weight"].to_numpy(np.float64)
    clusters = s["analysis_cluster"].astype(str).to_numpy()

    unique_clusters, inverse = np.unique(clusters, return_inverse=True)

    return {
        "labels": labels,
        "weights": weights,
        "inverse": inverse,
        "n_clusters": len(unique_clusters),
    }


def bootstrap_eer(scores, seed):
    p = prepare(scores)

    labels = p["labels"]
    weights = p["weights"]
    inverse = p["inverse"]
    n_clusters = p["n_clusters"]

    rng = np.random.default_rng(seed)
    vals = np.empty(N_BOOTSTRAP, dtype=np.float64)

    for b in range(N_BOOTSTRAP):
        multiplicity = np.bincount(
            rng.integers(0, n_clusters, size=n_clusters),
            minlength=n_clusters,
        ).astype(np.float64)

        w = weights * multiplicity[inverse]

        total_b = w[labels == 1].sum()
        total_s = w[labels == 0].sum()

        cum_b = np.cumsum(np.where(labels == 1, w, 0.0))
        cum_s = np.cumsum(np.where(labels == 0, w, 0.0))

        fnr = (total_b - cum_b) / total_b
        fpr = cum_s / total_s

        i = np.argmin(np.abs(fnr - fpr))
        vals[b] = 0.5 * (fnr[i] + fpr[i])

    alpha = (1.0 - CI) / 2.0

    return {
        "eer_pct": weighted_eer(scores) * 100.0,
        "bootstrap_mean_eer_pct": vals.mean() * 100.0,
        "ci_lo_pct": np.quantile(vals, alpha) * 100.0,
        "ci_hi_pct": np.quantile(vals, 1.0 - alpha) * 100.0,
        "n_utterances": len(scores),
        "n_clusters": n_clusters,
        "bootstrap_unit": "speaker",
        "n_bootstrap": N_BOOTSTRAP,
        "seed": seed,
    }


def bootstrap_gap(english, hindi, seed):
    e = prepare(english)
    h = prepare(hindi)

    rng = np.random.default_rng(seed)
    vals = np.empty(N_BOOTSTRAP, dtype=np.float64)

    for b in range(N_BOOTSTRAP):
        e_mult = np.bincount(
            rng.integers(0, e["n_clusters"], size=e["n_clusters"]),
            minlength=e["n_clusters"],
        ).astype(np.float64)

        h_mult = np.bincount(
            rng.integers(0, h["n_clusters"], size=h["n_clusters"]),
            minlength=h["n_clusters"],
        ).astype(np.float64)

        def one(p, mult):
            labels = p["labels"]
            w = p["weights"] * mult[p["inverse"]]

            total_b = w[labels == 1].sum()
            total_s = w[labels == 0].sum()

            cum_b = np.cumsum(np.where(labels == 1, w, 0.0))
            cum_s = np.cumsum(np.where(labels == 0, w, 0.0))

            fnr = (total_b - cum_b) / total_b
            fpr = cum_s / total_s

            i = np.argmin(np.abs(fnr - fpr))
            return 0.5 * (fnr[i] + fpr[i])

        vals[b] = one(h, h_mult) - one(e, e_mult)

    observed = weighted_eer(hindi) - weighted_eer(english)
    alpha = (1.0 - CI) / 2.0

    return {
        "observed_hindi_minus_english_pp": observed * 100.0,
        "bootstrap_mean_pp": vals.mean() * 100.0,
        "ci_lo_pp": np.quantile(vals, alpha) * 100.0,
        "ci_hi_pp": np.quantile(vals, 1.0 - alpha) * 100.0,
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_unit": "speaker",
        "seed": seed,
    }


def main():
    en_path = BASE / "overlap_weighted_english_scores.csv"
    hi_path = BASE / "overlap_weighted_hindi_scores.csv"

    if not en_path.exists() or not hi_path.exists():
        raise FileNotFoundError(
            "Expected overlap_weighted_english_scores.csv and "
            "overlap_weighted_hindi_scores.csv"
        )

    english = pd.read_csv(en_path)
    hindi = pd.read_csv(hi_path)

    print("English rows:", len(english))
    print("Hindi rows:", len(hindi))
    print("English speakers:", english.analysis_cluster.nunique())
    print("Hindi speakers:", hindi.analysis_cluster.nunique())

    print("\nComputing English bootstrap...")
    en_result = bootstrap_eer(english, SEED + 1)

    print("Computing Hindi bootstrap...")
    hi_result = bootstrap_eer(hindi, SEED + 2)

    print("Computing Hindi-English gap bootstrap...")
    gap_result = bootstrap_gap(english, hindi, SEED + 3)

    result = {
        "experiment": "spectral_artifact_overlap_weighted_eer",
        "english": en_result,
        "hindi": hi_result,
        "language_gap": gap_result,
    }

    out = BASE / "overlap_weighted_eer_summary.json"
    out.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 70)
    print(json.dumps(result, indent=2))
    print("=" * 70)
    print("\nWrote:", out)


if __name__ == "__main__":
    main()
