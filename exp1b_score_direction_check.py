#!/usr/bin/env python3
"""
Experiment 1b — Score-direction reversal check (raw vs. silence-trimmed).

Question:
    exp1_silence_controlled_eer.py shows the *aggregate* Hindi-minus-English
    EER gap survives silence trimming. This script checks whether individual
    utterances actually flip which side of the decision boundary they're on
    when scored raw vs. trimmed, and whether the raw/trimmed scores stay
    rank-correlated. A stable aggregate gap could still hide a lot of
    utterance-level churn, especially if trimming is noisy for short clips.

Requires exp1_silence_controlled_eer.py to have been run first, so that
    phase5_outputs/silence_controlled_eer/{language}_trimmed_scores.csv
exist alongside the frozen Phase-5 raw score files.

Outputs:
    phase5_outputs/silence_controlled_eer/
        score_direction_merged.csv        (per-utterance raw+trimmed+flags)
        score_direction_summary.json      (per-language flip rates, correlations)

Run from repository root:
    python exp1b_score_direction_check.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from phase5_hindi_eval import CFG, OUT, compute_eer

RAW_EER_FILES = {
    "english": OUT / "english_cv_eer_scores.csv",
    "hindi": OUT / "hindi_eer_scores.csv",
}

TRIMMED_DIR = OUT / "silence_controlled_eer"
TRIMMED_FILES = {
    "english": TRIMMED_DIR / "english_trimmed_scores.csv",
    "hindi": TRIMMED_DIR / "hindi_trimmed_scores.csv",
}

OUT_MERGED_CSV = TRIMMED_DIR / "score_direction_merged.csv"
OUT_SUMMARY_JSON = TRIMMED_DIR / "score_direction_summary.json"


def merge_language(language: str) -> pd.DataFrame:
    raw_path = RAW_EER_FILES[language]
    trimmed_path = TRIMMED_FILES[language]

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw score file: {raw_path}")
    if not trimmed_path.exists():
        raise FileNotFoundError(
            f"Missing trimmed score file: {trimmed_path}. "
            "Run exp1_silence_controlled_eer.py first."
        )

    raw = pd.read_csv(raw_path)
    trimmed = pd.read_csv(trimmed_path)

    for df, name in ((raw, "raw"), (trimmed, "trimmed")):
        missing = {"pair_id", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"{name} file for {language} missing columns: {missing}")
        df["pair_id"] = df["pair_id"].astype(str)
        df["label"] = df["label"].astype(int)

    merged = raw.merge(
        trimmed[["pair_id", "label", "analysis_cluster", "bonafide_logit_trimmed",
                 "leading_silence_ms", "trailing_silence_ms", "trimmed_duration_s"]],
        on=["pair_id", "label"],
        how="inner",
        suffixes=("", "_trim"),
    )

    n_raw = len(raw)
    n_trimmed = len(trimmed)
    n_merged = len(merged)
    if n_merged != n_raw or n_merged != n_trimmed:
        raise ValueError(
            f"{language}: row-count mismatch after merge on (pair_id, label) "
            f"raw={n_raw} trimmed={n_trimmed} merged={n_merged}. "
            "Check for duplicate/missing pair_id+label combinations."
        )

    merged["language"] = language
    return merged


def analyze_language(merged: pd.DataFrame, language: str) -> dict:
    bona_raw = merged.loc[merged.label == 1, "bonafide_logit"].to_numpy(dtype=float)
    spoof_raw = merged.loc[merged.label == 0, "bonafide_logit"].to_numpy(dtype=float)
    bona_trim = merged.loc[merged.label == 1, "bonafide_logit_trimmed"].to_numpy(dtype=float)
    spoof_trim = merged.loc[merged.label == 0, "bonafide_logit_trimmed"].to_numpy(dtype=float)

    _, raw_thresh = compute_eer(bona_raw, spoof_raw)
    _, trim_thresh = compute_eer(bona_trim, spoof_trim)

    raw_logit = merged["bonafide_logit"].to_numpy(dtype=float)
    trim_logit = merged["bonafide_logit_trimmed"].to_numpy(dtype=float)

    decision_raw = raw_logit >= raw_thresh          # True = classified bonafide
    decision_trim = trim_logit >= trim_thresh
    threshold_flip = decision_raw != decision_trim

    sign_raw = raw_logit >= 0.0
    sign_trim = trim_logit >= 0.0
    sign_flip = sign_raw != sign_trim

    merged["decision_raw_bonafide"] = decision_raw
    merged["decision_trimmed_bonafide"] = decision_trim
    merged["threshold_flip"] = threshold_flip
    merged["sign_flip"] = sign_flip

    rho_all, p_all = spearmanr(raw_logit, trim_logit)

    by_label = {}
    for label, name in ((1, "real_bonafide"), (0, "fake_spoof")):
        mask = merged.label == label
        rho, p = spearmanr(raw_logit[mask], trim_logit[mask])
        by_label[name] = {
            "n": int(mask.sum()),
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "threshold_flip_rate": float(threshold_flip[mask].mean()),
            "sign_flip_rate": float(sign_flip[mask].mean()),
        }

    return {
        "language": language,
        "n_utterances": int(len(merged)),
        "raw_threshold": float(raw_thresh),
        "trimmed_threshold": float(trim_thresh),
        "spearman_rho_overall": float(rho_all),
        "spearman_p_overall": float(p_all),
        "threshold_flip_rate_overall": float(threshold_flip.mean()),
        "sign_flip_rate_overall": float(sign_flip.mean()),
        "by_label": by_label,
    }


def main() -> None:
    all_merged = []
    summaries = []

    for language in RAW_EER_FILES:
        merged = merge_language(language)
        summary = analyze_language(merged, language)
        all_merged.append(merged)
        summaries.append(summary)

    full = pd.concat(all_merged, ignore_index=True)
    full.to_csv(OUT_MERGED_CSV, index=False)

    result = {
        "config": {
            "sample_rate": CFG.sample_rate,
            "checkpoint_path": CFG.checkpoint_path,
        },
        "note": (
            "threshold_flip uses each condition's own EER-optimal threshold "
            "(raw threshold for raw scores, trimmed threshold for trimmed scores) "
            "to decide bonafide-vs-spoof, then flags utterances whose decision "
            "differs raw-vs-trimmed. sign_flip is a threshold-free reference using "
            "logit >= 0 as the boundary."
        ),
        "per_language": summaries,
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote merged per-utterance table to {OUT_MERGED_CSV}")
    print(f"Wrote summary to {OUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
