#!/usr/bin/env python3
"""
Experiment 7 — Speaker-level heterogeneity.

Question
--------
Is the multilingual gap broad across speakers, or driven by a subset of
speakers?

Uses the FROZEN RAW Phase-5 score CSVs:
    phase5_outputs/english_cv_eer_scores.csv
    phase5_outputs/hindi_eer_scores.csv

For every speaker it computes a speaker-level paired separation score:

    separation = mean(fake score) - mean(real score)

with the bonafide-logit convention, larger positive values meaning better
separation in favor of bonafide speech.

It reports:
    * speaker-level distributions
    * quantiles
    * fraction of speakers with non-positive separation
    * English-vs-Hindi speaker distribution comparison
    * bootstrap CI for median/mean difference
    * optional speaker-level EER-like ranking diagnostic

Run:
    python exp7_speaker_level_heterogeneity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BASE = Path("phase5_outputs")
OUT = BASE / "speaker_heterogeneity"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 17235
N_BOOTSTRAP = 2000
CI = 0.95


def load_score(path):
    df = pd.read_csv(path)
    required = {"pair_id", "label", "bonafide_logit", "analysis_cluster"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    df["pair_id"] = df.pair_id.astype(str)
    df["analysis_cluster"] = df.analysis_cluster.astype(str)
    df["label"] = df.label.astype(int)
    return df


def speaker_table(df, language):
    rows = []

    for spk, g in df.groupby("analysis_cluster"):
        real = g[g.label == 1]
        fake = g[g.label == 0]

        if real.empty or fake.empty:
            continue

        real_mean = real.bonafide_logit.mean()
        fake_mean = fake.bonafide_logit.mean()

        # With higher score = more bonafide-like:
        # positive separation means real is above fake.
        separation = real_mean - fake_mean

        rows.append({
            "language": language,
            "speaker": str(spk),
            "n_real": len(real),
            "n_fake": len(fake),
            "real_mean_logit": float(real_mean),
            "fake_mean_logit": float(fake_mean),
            "separation_real_minus_fake": float(separation),
        })

    return pd.DataFrame(rows)


def bootstrap_distribution_difference(en, hi, col, seed):
    e = en[col].to_numpy(float)
    h = hi[col].to_numpy(float)

    rng = np.random.default_rng(seed)
    mean_vals = []
    median_vals = []

    for _ in range(N_BOOTSTRAP):
        eb = rng.choice(e, len(e), replace=True)
        hb = rng.choice(h, len(h), replace=True)

        mean_vals.append(hb.mean() - eb.mean())
        median_vals.append(np.median(hb) - np.median(eb))

    alpha = (1 - CI) / 2

    return {
        "observed_mean_difference_hindi_minus_english": float(
            h.mean() - e.mean()
        ),
        "mean_ci_lo": float(np.quantile(mean_vals, alpha)),
        "mean_ci_hi": float(np.quantile(mean_vals, 1 - alpha)),
        "observed_median_difference_hindi_minus_english": float(
            np.median(h) - np.median(e)
        ),
        "median_ci_lo": float(np.quantile(median_vals, alpha)),
        "median_ci_hi": float(np.quantile(median_vals, 1 - alpha)),
        "bootstrap_unit": "speaker",
        "n_bootstrap": N_BOOTSTRAP,
        "seed": seed,
    }


def main():
    en_scores = load_score(BASE / "english_cv_eer_scores.csv")
    hi_scores = load_score(BASE / "hindi_eer_scores.csv")

    en = speaker_table(en_scores, "english")
    hi = speaker_table(hi_scores, "hindi")

    en.to_csv(OUT / "english_speaker_metrics.csv", index=False)
    hi.to_csv(OUT / "hindi_speaker_metrics.csv", index=False)

    rows = []

    for lang, d in [("english", en), ("hindi", hi)]:
        x = d.separation_real_minus_fake.to_numpy(float)

        rows.append({
            "language": lang,
            "n_speakers": len(d),
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "sd": float(np.std(x, ddof=1)),
            "q10": float(np.quantile(x, 0.10)),
            "q25": float(np.quantile(x, 0.25)),
            "q75": float(np.quantile(x, 0.75)),
            "q90": float(np.quantile(x, 0.90)),
            "fraction_nonpositive": float(np.mean(x <= 0)),
        })

    desc = pd.DataFrame(rows)
    desc.to_csv(
        OUT / "speaker_heterogeneity_summary.csv",
        index=False,
    )

    boot = bootstrap_distribution_difference(
        en, hi,
        "separation_real_minus_fake",
        SEED,
    )

    # Distribution comparison. This is descriptive/inferential at speaker
    # level, not a replacement for the utterance-level EER.
    u_stat, p_u = stats.mannwhitneyu(
        hi.separation_real_minus_fake,
        en.separation_real_minus_fake,
        alternative="two-sided",
    )

    ks_stat, p_ks = stats.ks_2samp(
        hi.separation_real_minus_fake,
        en.separation_real_minus_fake,
    )

    result = {
        "experiment": "speaker_level_heterogeneity",
        "english_speakers": len(en),
        "hindi_speakers": len(hi),
        "speaker_metric": (
            "mean real bonafide_logit minus mean fake bonafide_logit"
        ),
        "english_summary": desc[desc.language == "english"].iloc[0].to_dict(),
        "hindi_summary": desc[desc.language == "hindi"].iloc[0].to_dict(),
        "bootstrap_difference": boot,
        "mann_whitney_u": {
            "statistic": float(u_stat),
            "p_value": float(p_u),
        },
        "ks_test": {
            "statistic": float(ks_stat),
            "p_value": float(p_ks),
        },
    }

    (OUT / "speaker_heterogeneity_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))
    print("Wrote:", OUT)


if __name__ == "__main__":
    main()
