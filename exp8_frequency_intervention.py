#!/usr/bin/env python3
"""
Experiment 8 — Direct frequency intervention on the existing 100-pair XAI set.

Question
--------
Does directly perturbing the frequency region associated with the Hindi
fake-audio attribution shift causally change the detector score?

Design
------
Uses the SAME existing 100 fake + 100 real XAI pairs per language from the
Phase-5 feature/faithfulness manifests.

For each waveform, create deterministic band-limited counterfactuals by
zeroing one STFT band at a time:

    low  = 0–1 kHz
    mid  = 1–4 kHz
    high = 4–8 kHz

STFT settings match Phase 5:
    sample_rate = 16000
    n_fft = 512
    hop_length = 160
    center = False

The intervention uses the waveform-length STFT/ISTFT, then feeds the
counterfactual through the SAME frozen model_input()/model scorer used by
Phase 5.

Primary outcome:
    delta_score = counterfactual_bonafide_logit - original_bonafide_logit

Interpretation:
    For fake audio, a large score decrease after removing a band means that
    band contained evidence supporting the model's bonafide/spoof decision
    in the opposite direction; sign must be interpreted relative to the
    model's verified output convention.

This is an intervention analysis, NOT an attribution-faithfulness metric.

Run:
    python exp8_frequency_intervention.py

Expect ~800 model forward passes:
    100 English fake + 100 English real
    100 Hindi fake + 100 Hindi real
    times 4 conditions (original + 3 band interventions).

The script caches results and resumes safely.
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch

from phase5_hindi_eval import CFG, OUT, load_model, model_input, score


BASE = Path("phase5_outputs")
OUT_DIR = BASE / "frequency_intervention"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE = OUT_DIR / "frequency_intervention_scores.csv"

SAMPLE_RATE = CFG.sample_rate
N_FFT = CFG.stft_n_fft
HOP = CFG.stft_hop
CENTER = False

BANDS = {
    "low": (0, 1000),
    "mid": (1000, 4000),
    "high": (4000, 8000),
}

SEED = 18235


def locate_manifest(language, label):
    if language == "english" and label == 1:
        candidates = [
            BASE / "faithfulness_cv_en_real.csv",
        ]
    elif language == "english" and label == 0:
        candidates = [
            BASE / "faithfulness_cv_en_fake.csv",
        ]
    elif language == "hindi" and label == 1:
        candidates = [
            BASE / "faithfulness_cv_hi_real.csv",
        ]
    else:
        candidates = [
            BASE / "faithfulness_cv_hi_fake.csv",
        ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"No frozen Phase-5 manifest found for {language}, label={label}"
    )


def load_existing_samples():
    rows = []

    for language in ["english", "hindi"]:
        for label, kind in [(1, "real"), (0, "fake")]:
            p = locate_manifest(language, label)
            df = pd.read_csv(p)

            if len(df) != 100:
                raise ValueError(
                    f"{p}: expected exactly 100 XAI samples, found {len(df)}"
                )

            required = {"sample_key", "pair_id", "path"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{p}: missing {sorted(missing)}")

            for _, r in df.iterrows():
                rows.append({
                    "language": language,
                    "label": label,
                    "sample_key": str(r.sample_key),
                    "pair_id": str(r.pair_id),
                    "path": str(r.path),
                })

    out = pd.DataFrame(rows)

    if out.sample_key.duplicated().any():
        raise ValueError("Duplicate sample_key across intervention cohorts")

    return out


def band_mask(freqs, lo, hi):
    return (freqs >= lo) & (freqs < hi)


def remove_band(wav, band):
    """
    Remove one frequency band from a real waveform using magnitude STFT
    masking while retaining the original phase.
    """
    S = librosa.stft(
        wav,
        n_fft=N_FFT,
        hop_length=HOP,
        center=CENTER,
    )

    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
    lo, hi = BANDS[band]
    mask = band_mask(freqs, lo, hi)

    S_mod = S.copy()
    S_mod[mask, :] = 0.0

    y = librosa.istft(
        S_mod,
        hop_length=HOP,
        win_length=N_FFT,
        center=CENTER,
        length=len(wav),
    )

    return y.astype(np.float32)


def score_wave(model, wav):
    """
    Use the Phase-5 model_input pathway semantics on an already constructed
    waveform: normalize to fixed length with the same pad() implementation.
    """
    from data_utils_SSL import pad

    fixed = pad(np.asarray(wav, dtype=np.float32), CFG.fixed_len)
    x = torch.tensor(fixed, dtype=torch.float32)

    # Model input shape follows Phase-5 model scoring path.
    with torch.no_grad():
        s = score(model, x)

    return float(s)


def load_original_and_score(model, path):
    wav, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
    )
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"{path}: unexpected sample rate {sr}")

    original = score_wave(model, wav)
    return wav.astype(np.float32), original


def main():
    samples = load_existing_samples()

    existing = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame()
    done = set()

    if not existing.empty:
        done = set(
            existing["sample_key"].astype(str)
            + "::"
            + existing["condition"].astype(str)
        )

    model = load_model()

    rows = existing.to_dict("records") if not existing.empty else []

    for _, r in tqdm(
        samples.iterrows(),
        total=len(samples),
        desc="Frequency intervention",
    ):
        key_base = str(r.sample_key)

        wav, original_score = load_original_and_score(
            model,
            str(r.path),
        )

        conditions = {"original": wav}
        for band in BANDS:
            conditions[f"remove_{band}"] = remove_band(wav, band)

        for condition, signal in conditions.items():
            key = f"{key_base}::{condition}"
            if key in done:
                continue

            if condition == "original":
                new_score = original_score
            else:
                new_score = score_wave(model, signal)

            rows.append({
                "language": r.language,
                "label": int(r.label),
                "sample_key": key_base,
                "pair_id": str(r.pair_id),
                "path": r.path,
                "condition": condition,
                "original_score": original_score,
                "counterfactual_score": new_score,
                "delta_score": new_score - original_score,
            })

            # Incremental save.
            if len(rows) % 25 == 0:
                pd.DataFrame(rows).to_csv(CACHE, index=False)

    results = pd.DataFrame(rows)
    results.to_csv(CACHE, index=False)

    # Summary by language/label/band.
    summary_rows = []

    for (language, label, condition), g in results.groupby(
        ["language", "label", "condition"]
    ):
        if condition == "original":
            continue

        d = g["delta_score"].to_numpy(float)
        summary_rows.append({
            "language": language,
            "label": "real" if label == 1 else "fake",
            "condition": condition,
            "n": len(d),
            "mean_delta": float(d.mean()),
            "median_delta": float(np.median(d)),
            "q25": float(np.quantile(d, 0.25)),
            "q75": float(np.quantile(d, 0.75)),
            "positive_fraction": float(np.mean(d > 0)),
            "negative_fraction": float(np.mean(d < 0)),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUT_DIR / "frequency_intervention_summary.csv",
        index=False,
    )

    # Paired language interaction on the SAME pair IDs, for fake clips.
    # For every intervention, compare Hindi delta against English delta on
    # the phase-5-selected pair IDs only when the same pair_id exists in both
    # languages. Since pair IDs are language-local, this is not a true paired
    # cross-language test; therefore report it only as an independent
    # distribution comparison.
    comparisons = []

    for condition in ["remove_low", "remove_mid", "remove_high"]:
        for label in [0, 1]:
            e = results[
                (results.language == "english")
                & (results.label == label)
                & (results.condition == condition)
            ]["delta_score"].to_numpy(float)

            h = results[
                (results.language == "hindi")
                & (results.label == label)
                & (results.condition == condition)
            ]["delta_score"].to_numpy(float)

            u, p = __import__("scipy").stats.mannwhitneyu(
                e, h, alternative="two-sided"
            )

            comparisons.append({
                "condition": condition,
                "label": "real" if label == 1 else "fake",
                "english_mean_delta": float(e.mean()),
                "hindi_mean_delta": float(h.mean()),
                "hindi_minus_english": float(h.mean() - e.mean()),
                "mannwhitney_p": float(p),
                "n_english": len(e),
                "n_hindi": len(h),
            })

    pd.DataFrame(comparisons).to_csv(
        OUT_DIR / "frequency_intervention_language_comparison.csv",
        index=False,
    )

    cfg = {
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop_length": HOP,
        "center": CENTER,
        "bands_hz": BANDS,
        "fixed_len": CFG.fixed_len,
        "n_samples_per_language_label": 100,
        "model_checkpoint": str(CFG.checkpoint_path),
        "interpretation": (
            "delta_score = counterfactual bonafide_logit - original "
            "bonafide_logit. A large negative delta means removing that band "
            "reduced the bonafide score."
        ),
    }

    (OUT_DIR / "frequency_intervention_config.json").write_text(
        json.dumps(cfg, indent=2),
        encoding="utf-8",
    )

    print("\n=== FREQUENCY INTERVENTION SUMMARY ===")
    print(summary.to_string(index=False))
    print("\nWrote:", OUT_DIR)


if __name__ == "__main__":
    main()
