#!/usr/bin/env python3
"""
Experiment 2 — Language × reconstruction interaction for explanation features.

Question:
    Is the real->fake attribution change larger in Hindi than in English?

For each feature:
    interaction =
        (Hindi_fake - Hindi_real)
        -
        (English_fake - English_real)

This is a difference-in-differences analysis.

Inputs are frozen Phase-5 XAI feature tables:
    explanation_features_cv_en_real.csv
    explanation_features_cv_en_fake.csv
    explanation_features_cv_hi_real.csv
    explanation_features_cv_hi_fake.csv

and, optionally, Phase-5b occlusion feature tables for 20/50/100 ms.

Unit of resampling:
    speaker cluster, while preserving the real/fake pairing within each
    language and speaker.

Outputs:
    phase5_outputs/language_reconstruction_interaction/
        interaction_ig.csv
        interaction_occlusion_20ms.csv
        interaction_occlusion_50ms.csv
        interaction_occlusion_100ms.csv
        interaction_all_methods_windows.csv

No model inference is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from phase5_hindi_eval import CFG, OUT


FEATURES = [
    "low_frac", "mid_frac", "high_frac",
    "speech_frac", "silence_frac",
    "gini", "entropy",
    "topk_1pct", "topk_5pct", "topk_10pct",
]

OUT_DIR = OUT / "language_reconstruction_interaction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = CFG.seed + 4001
N_BOOTSTRAP = CFG.n_bootstrap
CI = CFG.ci


def load_feature_table(name: str) -> pd.DataFrame:
    path = OUT / name
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {"pair_id", "analysis_cluster", "label", *FEATURES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    df["pair_id"] = df["pair_id"].astype(str)
    df["analysis_cluster"] = df["analysis_cluster"].astype(str)
    df["label"] = df["label"].astype(int)

    return df


def paired_language_differences(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    """
    Compute fake-real difference for every paired clip.

    The two tables are expected to contain the same pair IDs in the selected
    XAI sample. Duplicate pair IDs are averaged before differencing.
    """
    r = real_df.groupby(["analysis_cluster", "pair_id"], as_index=False)[feature].mean()
    f = fake_df.groupby(["analysis_cluster", "pair_id"], as_index=False)[feature].mean()

    merged = r.merge(
        f,
        on=["analysis_cluster", "pair_id"],
        suffixes=("_real", "_fake"),
        how="inner",
        validate="one_to_one",
    ).dropna()

    merged["difference_fake_minus_real"] = (
        merged[f"{feature}_fake"] - merged[f"{feature}_real"]
    )
    return merged


def cluster_bootstrap_interaction(
    en_diff: pd.DataFrame,
    hi_diff: pd.DataFrame,
    feature: str,
) -> dict:
    """
    Speaker-cluster bootstrap for:
        mean(Hindi fake-real) - mean(English fake-real)

    Resampling is done independently within language, at speaker level.
    Within each sampled speaker, all their paired clip-level differences are
    retained, preserving repeated-measures structure.
    """
    en_groups = {
        str(s): g["difference_fake_minus_real"].to_numpy(dtype=float)
        for s, g in en_diff.groupby("analysis_cluster")
        if len(g)
    }
    hi_groups = {
        str(s): g["difference_fake_minus_real"].to_numpy(dtype=float)
        for s, g in hi_diff.groupby("analysis_cluster")
        if len(g)
    }

    en_mean = float(en_diff["difference_fake_minus_real"].mean())
    hi_mean = float(hi_diff["difference_fake_minus_real"].mean())
    observed = hi_mean - en_mean

    rng = np.random.default_rng(
        SEED + sum(ord(c) for c in feature)
    )
    vals = []

    for _ in range(N_BOOTSTRAP):
        ed = rng.choice(list(en_groups), size=len(en_groups), replace=True)
        hd = rng.choice(list(hi_groups), size=len(hi_groups), replace=True)

        e_values = np.concatenate([en_groups[s] for s in ed])
        h_values = np.concatenate([hi_groups[s] for s in hd])

        vals.append(float(h_values.mean() - e_values.mean()))

    alpha = (1.0 - CI) / 2.0

    return {
        "observed_hindi_minus_english_interaction": observed,
        "ci_lo": float(np.quantile(vals, alpha)),
        "ci_hi": float(np.quantile(vals, 1 - alpha)),
        "bootstrap_mean": float(np.mean(vals)),
        "n_bootstrap": len(vals),
        "n_english_pairs": int(len(en_diff)),
        "n_hindi_pairs": int(len(hi_diff)),
        "n_english_speakers": int(len(en_groups)),
        "n_hindi_speakers": int(len(hi_groups)),
        "bootstrap_unit": "speaker",
    }


def analyze_method(
    label: str,
    tables: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    en_real = tables["en_real"]
    en_fake = tables["en_fake"]
    hi_real = tables["hi_real"]
    hi_fake = tables["hi_fake"]

    rows = []

    for feature in FEATURES:
        en_diff = paired_language_differences(en_real, en_fake, feature)
        hi_diff = paired_language_differences(hi_real, hi_fake, feature)

        if len(en_diff) < 3 or len(hi_diff) < 3:
            continue

        res = cluster_bootstrap_interaction(en_diff, hi_diff, feature)

        # Standardized descriptive effect using the paired differences pooled
        # across languages. This is descriptive; the cluster bootstrap CI is
        # the primary inferential quantity.
        pooled = np.concatenate(
            [
                en_diff["difference_fake_minus_real"].to_numpy(),
                hi_diff["difference_fake_minus_real"].to_numpy(),
            ]
        )
        pooled_sd = float(np.std(pooled, ddof=1))
        d = res["observed_hindi_minus_english_interaction"] / pooled_sd if pooled_sd > 0 else np.nan

        rows.append(
            {
                "method": label,
                "feature": feature,
                "english_fake_minus_real": float(
                    en_diff["difference_fake_minus_real"].mean()
                ),
                "hindi_fake_minus_real": float(
                    hi_diff["difference_fake_minus_real"].mean()
                ),
                "interaction_hindi_minus_english": res[
                    "observed_hindi_minus_english_interaction"
                ],
                "ci_lo": res["ci_lo"],
                "ci_hi": res["ci_hi"],
                "descriptive_effect_size": float(d),
                "n_english_pairs": res["n_english_pairs"],
                "n_hindi_pairs": res["n_hindi_pairs"],
                "n_english_speakers": res["n_english_speakers"],
                "n_hindi_speakers": res["n_hindi_speakers"],
                "bootstrap_unit": "speaker",
                "n_bootstrap": res["n_bootstrap"],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    # IG features from Phase 5.
    ig = analyze_method(
        "integrated_gradients",
        {
            "en_real": load_feature_table("explanation_features_cv_en_real.csv"),
            "en_fake": load_feature_table("explanation_features_cv_en_fake.csv"),
            "hi_real": load_feature_table("explanation_features_cv_hi_real.csv"),
            "hi_fake": load_feature_table("explanation_features_cv_hi_fake.csv"),
        },
    )
    ig.to_csv(OUT_DIR / "interaction_ig.csv", index=False)

    all_tables = [ig]

    # Occlusion features from Phase 5b, if present.
    occ_dir = OUT / "occlusion_analysis"
    for window in CFG.occlusion_windows_ms:
        prefix = f"occlusion_features_"
        files = {
            "en_real": occ_dir / f"{prefix}cv_en_real_{window}ms.csv",
            "en_fake": occ_dir / f"{prefix}cv_en_fake_{window}ms.csv",
            "hi_real": occ_dir / f"{prefix}cv_hi_real_{window}ms.csv",
            "hi_fake": occ_dir / f"{prefix}cv_hi_fake_{window}ms.csv",
        }

        if not all(p.exists() for p in files.values()):
            continue

        occ = analyze_method(
            f"occlusion_{window}ms",
            {k: pd.read_csv(v) for k, v in files.items()},
        )
        occ.to_csv(
            OUT_DIR / f"interaction_occlusion_{window}ms.csv",
            index=False,
        )
        all_tables.append(occ)

    combined = pd.concat(all_tables, ignore_index=True)
    combined.to_csv(
        OUT_DIR / "interaction_all_methods_windows.csv",
        index=False,
    )

    summary = {
        "question": (
            "Does reconstruction induce a larger attribution change in Hindi "
            "than English? Interaction = (Hindi fake-real) - (English fake-real)."
        ),
        "features": FEATURES,
        "primary_inference": "speaker-cluster bootstrap 95% CI",
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "ci": CI,
        "methods_included": combined["method"].unique().tolist(),
    }
    (OUT_DIR / "interaction_manifest.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(combined.to_string(index=False))
    print(f"\nWrote results to {OUT_DIR}")


if __name__ == "__main__":
    main()
