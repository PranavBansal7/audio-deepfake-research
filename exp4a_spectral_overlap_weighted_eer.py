
#!/usr/bin/env python3
"""
Experiment 4A — Spectral-artifact overlap-weighted EER.

Purpose
-------
Estimate the Hindi-vs-English EER gap after reweighting the two language
cohorts toward the COMMON support of the observed fake-audio covariates.

Why overlap weighting rather than ordinary 1:1 matching?
--------------------------------------------------------
A full 1:1 matching of 1125 English to 1125 Hindi observations without
discarding any pairs would only permute the observations and would NOT change
the EER. The estimand therefore needs either weighting or a prespecified
matched subset. This script uses overlap weighting as the primary sensitivity
analysis and retains every real/fake pair.

Covariates are measured on the EXISTING frozen fake WAV files:
    low_energy_frac
    mid_energy_frac
    high_energy_frac
    RMS
    peak
    duration_s

A logistic language model is fitted on the fake covariates:
    P(language = Hindi | covariates)

Overlap weights:
    English weight = 1 - p
    Hindi weight   = p

The same pair-level weight is applied to that language's real and fake
utterances, preserving the original within-pair design.

Outputs:
    phase5_outputs/spectral_overlap_eer/
        fake_covariates.csv
        propensity_scores.csv
        overlap_weighted_scores.csv
        covariate_balance.csv
        overlap_weighted_eer_summary.json

Run from repository root:
    python exp4a_spectral_overlap_weighted_eer.py

No fake audio is regenerated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from phase5_hindi_eval import CFG, OUT, compute_eer, load_model, model_input, score


EN_DIR = Path("english_griffinlim_eval_final")
HI_DIR = Path("hindi_griffinlim_eval_final")

OUT_DIR = OUT / "spectral_overlap_eer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = CFG.seed + 6001
N_BOOTSTRAP = CFG.n_eer_bootstrap
CI = CFG.ci

SAMPLE_RATE = CFG.sample_rate
N_FFT = CFG.stft_n_fft
HOP = CFG.stft_hop

if hasattr(CFG, "freq_bands"):
    FREQ_BANDS = dict(CFG.freq_bands)
else:
    FREQ_BANDS = {
        "low": (0, 1000),
        "mid": (1000, 4000),
        "high": (4000, 8000),
    }

COVARIATES = [
    "low_energy_frac",
    "mid_energy_frac",
    "high_energy_frac",
    "rms",
    "peak",
    "duration_s",
]


def load_meta(path: Path, language: str) -> pd.DataFrame:
    df = pd.read_csv(path / "metadata.csv")
    required = {"pair_id", "client_id", "real_path", "fake_path", "duration_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}/metadata.csv missing {sorted(missing)}")
    if len(df) != 1125 or df.pair_id.nunique() != 1125:
        raise ValueError(f"{language}: expected exactly 1125 unique pairs")

    df["pair_id"] = df["pair_id"].astype(str)
    df["client_id"] = df["client_id"].astype(str)
    df["analysis_cluster"] = df["client_id"]
    df["language"] = language
    return df


def fake_covariates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Covariates {df.language.iloc[0]}"):
        x, sr = librosa.load(str(r.fake_path), sr=SAMPLE_RATE, mono=True)
        if sr != SAMPLE_RATE or x.size == 0 or not np.isfinite(x).all():
            raise ValueError(f"Invalid audio: {r.fake_path}")

        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))
        peak = float(np.max(np.abs(x)))

        S = np.abs(librosa.stft(
            x, n_fft=N_FFT, hop_length=HOP, center=False
        ))
        freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
        E = S * S
        total = float(E.sum() + 1e-12)

        rec = {
            "language": r.language,
            "pair_id": str(r.pair_id),
            "client_id": str(r.client_id),
            "duration_s": float(len(x) / SAMPLE_RATE),
            "rms": rms,
            "peak": peak,
        }

        for name, (lo, hi) in FREQ_BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            rec[f"{name}_energy_frac"] = float(E[mask].sum() / total)

        rows.append(rec)

    return pd.DataFrame(rows)


def standardized_covariates(cov: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X = scaler.fit_transform(cov[COVARIATES].to_numpy(dtype=float))
    return X, scaler


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w))


def balance_table(cov: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    rows = []
    before_e = cov[cov.language == "english"]
    before_h = cov[cov.language == "hindi"]

    after = cov.copy()
    after["ow"] = weights.to_numpy()

    for stage, e, h, weighted in [
        ("before", before_e, before_h, False),
        ("overlap_weighted", None, None, True),
    ]:
        for c in COVARIATES:
            if not weighted:
                ev = e[c].to_numpy(float)
                hv = h[c].to_numpy(float)
                em, hm = ev.mean(), hv.mean()
                smd = standardized_mean_difference(ev, hv)
            else:
                eg = after[after.language == "english"]
                hg = after[after.language == "hindi"]
                ev = eg[c].to_numpy(float)
                hv = hg[c].to_numpy(float)
                ew = eg["ow"].to_numpy(float)
                hw = hg["ow"].to_numpy(float)
                em = weighted_mean(ev, ew)
                hm = weighted_mean(hv, hw)
                # Weighted SMD using weighted variances.
                evc = ev - em
                hvc = hv - hm
                evw = np.sum(ew * evc * evc) / np.sum(ew)
                hvw = np.sum(hw * hvc * hvc) / np.sum(hw)
                pooled = np.sqrt((evw + hvw) / 2)
                smd = float((em - hm) / pooled) if pooled > 0 else np.nan

            rows.append({
                "stage": stage,
                "covariate": c,
                "english_mean": float(em),
                "hindi_mean": float(hm),
                "difference": float(em - hm),
                "smd": float(smd),
            })

    return pd.DataFrame(rows)


def weighted_eer(scores: pd.DataFrame) -> float:
    """
    Weighted EER using the exact empirical score ranking.

    label=1 is bonafide, label=0 is spoof.
    Increasing bonafide_logit means more bonafide-like.
    """
    s = scores.sort_values("bonafide_logit", ascending=False).reset_index(drop=True)

    y = s.label.to_numpy(dtype=int)
    w = s.weight.to_numpy(dtype=float)

    total_b = float(w[y == 1].sum())
    total_s = float(w[y == 0].sum())
    if total_b <= 0 or total_s <= 0:
        raise ValueError("Invalid label weights.")

    # Threshold moves from +inf to -inf:
    # at each score, the observation switches from spoof to bonafide.
    cum_b = np.cumsum(np.where(y == 1, w, 0.0))
    cum_s = np.cumsum(np.where(y == 0, w, 0.0))

    fnr = (total_b - cum_b) / total_b
    fpr = cum_s / total_s

    d = np.abs(fnr - fpr)
    i = int(np.argmin(d))
    return float((fnr[i] + fpr[i]) / 2.0)


def cluster_weighted_eer_bootstrap(scores: pd.DataFrame, seed: int) -> dict:
    groups = {
        str(c): g
        for c, g in scores.groupby("analysis_cluster")
    }
    clusters = list(groups)

    observed = weighted_eer(scores)
    rng = np.random.default_rng(seed)
    vals = []

    for _ in range(N_BOOTSTRAP):
        draw = rng.choice(clusters, len(clusters), replace=True)
        sample = pd.concat([groups[c] for c in draw], ignore_index=True)
        vals.append(weighted_eer(sample))

    vals = np.asarray(vals, float)
    alpha = (1 - CI) / 2

    return {
        "eer_pct": float(observed * 100),
        "bootstrap_mean_eer_pct": float(vals.mean() * 100),
        "ci_lo_pct": float(np.quantile(vals, alpha) * 100),
        "ci_hi_pct": float(np.quantile(vals, 1 - alpha) * 100),
        "n_utterances": int(len(scores)),
        "n_clusters": int(scores.analysis_cluster.nunique()),
        "bootstrap_unit": "speaker",
        "n_bootstrap": int(len(vals)),
        "seed": seed,
    }


def bootstrap_gap(en: pd.DataFrame, hi: pd.DataFrame, seed: int) -> dict:
    eg = {str(c): g for c, g in en.groupby("analysis_cluster")}
    hg = {str(c): g for c, g in hi.groupby("analysis_cluster")}

    observed = (weighted_eer(hi) - weighted_eer(en)) * 100
    rng = np.random.default_rng(seed)
    vals = []

    for _ in range(N_BOOTSTRAP):
        ed = rng.choice(list(eg), len(eg), replace=True)
        hd = rng.choice(list(hg), len(hg), replace=True)
        es = pd.concat([eg[c] for c in ed], ignore_index=True)
        hs = pd.concat([hg[c] for c in hd], ignore_index=True)
        vals.append((weighted_eer(hs) - weighted_eer(es)) * 100)

    vals = np.asarray(vals, float)
    alpha = (1 - CI) / 2

    return {
        "observed_hindi_minus_english_pp": float(observed),
        "bootstrap_mean_pp": float(vals.mean()),
        "ci_lo_pp": float(np.quantile(vals, alpha)),
        "ci_hi_pp": float(np.quantile(vals, 1 - alpha)),
        "n_bootstrap": int(len(vals)),
        "bootstrap_unit": "speaker",
        "seed": seed,
    }


def score_with_model(
    model: torch.nn.Module,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, r in tqdm(meta.iterrows(), total=len(meta), desc=f"Scoring {meta.language.iloc[0]}"):
        pair_weight = float(r["_overlap_weight"])

        for label, path in ((1, r.real_path), (0, r.fake_path)):
            audio = model_input(str(path))
            logit = float(score(model, audio))
            rows.append({
                "language": r.language,
                "pair_id": str(r.pair_id),
                "analysis_cluster": str(r.client_id),
                "label": int(label),
                "path": str(path),
                "bonafide_logit": logit,
                "weight": pair_weight,
            })

    return pd.DataFrame(rows)


def find_frozen_score_table(language: str) -> Path | None:
    candidates = [
        OUT / "english_cv_eer_scores.csv" if language == "english" else OUT / "hindi_eer_scores.csv",
        OUT / "eer_scores_english_cv.csv" if language == "english" else OUT / "eer_scores_hindi.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    # Conservative fallback: only accept CSVs with the required score columns
    # and exactly 2250 rows, avoiding arbitrary analysis tables.
    for p in OUT.glob("**/*.csv"):
        try:
            df = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        if {"pair_id", "label", "bonafide_logit"}.issubset(df.columns):
            try:
                full_n = sum(1 for _ in open(p, "r", encoding="utf-8")) - 1
            except Exception:
                continue
            if full_n == 2250:
                if language == "english" and "english" in p.name.lower():
                    return p
                if language == "hindi" and "hindi" in p.name.lower():
                    return p
    return None


def main() -> None:
    en_meta = load_meta(EN_DIR, "english")
    hi_meta = load_meta(HI_DIR, "hindi")

    en_cov = fake_covariates(en_meta)
    hi_cov = fake_covariates(hi_meta)
    cov = pd.concat([en_cov, hi_cov], ignore_index=True)

    X, scaler = standardized_covariates(cov)
    y = (cov.language == "hindi").astype(int).to_numpy()

    clf = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
    )
    clf.fit(X, y)

    p_hindi = clf.predict_proba(X)[:, 1]
    # Overlap weights: target density proportional to min(e_density,h_density)
    # in the propensity-score formulation.
    overlap_w = np.where(y == 1, 1.0 - p_hindi, p_hindi)

    cov["propensity_hindi"] = p_hindi
    cov["overlap_weight"] = overlap_w
    cov.to_csv(OUT_DIR / "fake_covariates.csv", index=False)

    en_cov = cov[cov.language == "english"].copy()
    hi_cov = cov[cov.language == "hindi"].copy()

    # Map pair-level weights back to metadata.
    weight_map = cov.set_index(["language", "pair_id"])["overlap_weight"]

    en_meta["_overlap_weight"] = [
        float(weight_map.loc[("english", str(pid))])
        for pid in en_meta.pair_id
    ]
    hi_meta["_overlap_weight"] = [
        float(weight_map.loc[("hindi", str(pid))])
        for pid in hi_meta.pair_id
    ]

    # Covariate balance.
    balance = balance_table(cov.rename(columns={"overlap_weight": "ow"}), cov["overlap_weight"])
    balance.to_csv(OUT_DIR / "covariate_balance.csv", index=False)

    # Save propensity diagnostics.
    propensity_summary = {
        "min_p_hindi": float(p_hindi.min()),
        "q01_p_hindi": float(np.quantile(p_hindi, 0.01)),
        "q05_p_hindi": float(np.quantile(p_hindi, 0.05)),
        "median_p_hindi": float(np.median(p_hindi)),
        "q95_p_hindi": float(np.quantile(p_hindi, 0.95)),
        "q99_p_hindi": float(np.quantile(p_hindi, 0.99)),
        "max_p_hindi": float(p_hindi.max()),
        "effective_english_weight_n": float(
            (en_cov.overlap_weight.sum() ** 2)
            / np.sum(en_cov.overlap_weight ** 2)
        ),
        "effective_hindi_weight_n": float(
            (hi_cov.overlap_weight.sum() ** 2)
            / np.sum(hi_cov.overlap_weight ** 2)
        ),
    }

    (OUT_DIR / "propensity_diagnostics.json").write_text(
        json.dumps(propensity_summary, indent=2),
        encoding="utf-8",
    )

    # Prefer frozen Phase-5 scores. Score from model only if needed.
    model = None
    score_tables = {}
    for lang in ["english", "hindi"]:
        p = find_frozen_score_table(lang)
        if p is not None:
            s = pd.read_csv(p)
            s["pair_id"] = s["pair_id"].astype(str)
            score_tables[lang] = s
        else:
            if model is None:
                model = load_model()
            score_tables[lang] = None

    if score_tables["english"] is None:
        en_scores = score_with_model(model, en_meta)
    else:
        en_scores = score_tables["english"].copy()
        en_scores["pair_id"] = en_scores["pair_id"].astype(str)
        wmap = en_meta.set_index("pair_id")["_overlap_weight"].to_dict()
        cmap = en_meta.set_index("pair_id")["client_id"].to_dict()
        en_scores["weight"] = en_scores.pair_id.map(wmap)
        en_scores["analysis_cluster"] = en_scores.pair_id.map(cmap).astype(str)
        en_scores["language"] = "english"

    if score_tables["hindi"] is None:
        hi_scores = score_with_model(model, hi_meta)
    else:
        hi_scores = score_tables["hindi"].copy()
        hi_scores["pair_id"] = hi_scores["pair_id"].astype(str)
        wmap = hi_meta.set_index("pair_id")["_overlap_weight"].to_dict()
        cmap = hi_meta.set_index("pair_id")["client_id"].to_dict()
        hi_scores["weight"] = hi_scores.pair_id.map(wmap)
        hi_scores["analysis_cluster"] = hi_scores.pair_id.map(cmap).astype(str)
        hi_scores["language"] = "hindi"

    if en_scores.weight.isna().any() or hi_scores.weight.isna().any():
        raise RuntimeError("Missing overlap weights for scored pairs.")

    en_scores.to_csv(OUT_DIR / "overlap_weighted_english_scores.csv", index=False)
    hi_scores.to_csv(OUT_DIR / "overlap_weighted_hindi_scores.csv", index=False)

    en_result = cluster_weighted_eer_bootstrap(en_scores, SEED + 1)
    hi_result = cluster_weighted_eer_bootstrap(hi_scores, SEED + 2)
    gap = bootstrap_gap(en_scores, hi_scores, SEED + 3)

    # Also provide unweighted EERs from exactly these score tables as a check.
    raw_en = compute_eer(
        en_scores[en_scores.label == 1].bonafide_logit.to_numpy(),
        en_scores[en_scores.label == 0].bonafide_logit.to_numpy(),
    )[0] * 100
    raw_hi = compute_eer(
        hi_scores[hi_scores.label == 1].bonafide_logit.to_numpy(),
        hi_scores[hi_scores.label == 0].bonafide_logit.to_numpy(),
    )[0] * 100

    result = {
        "experiment": "spectral_artifact_overlap_weighted_eer",
        "question": (
            "Does the Hindi-vs-English EER gap persist after reweighting both "
            "language cohorts toward common support of measured fake-audio "
            "spectral/acoustic covariates?"
        ),
        "covariates": COVARIATES,
        "matching_method": "propensity-score overlap weighting",
        "weights": {
            "english": "1 - P(Hindi | fake covariates)",
            "hindi": "P(Hindi | fake covariates)",
        },
        "before_balance_max_abs_smd": float(
            balance.loc[balance.stage == "before", "smd"].abs().max()
        ),
        "after_balance_max_abs_smd": float(
            balance.loc[balance.stage == "overlap_weighted", "smd"].abs().max()
        ),
        "propensity": propensity_summary,
        "unweighted_check": {
            "english_eer_pct": float(raw_en),
            "hindi_eer_pct": float(raw_hi),
            "hindi_minus_english_pp": float(raw_hi - raw_en),
        },
        "overlap_weighted_english": en_result,
        "overlap_weighted_hindi": hi_result,
        "overlap_weighted_gap": gap,
        "interpretation_rule": (
            "A materially preserved language gap after strong covariate balance "
            "supports a language-dependent detector difference not explained by "
            "the measured fake-audio covariates. A substantial reduction supports "
            "those measured covariates as an important evaluation confound."
        ),
        "caveat": (
            "Overlap weighting controls measured covariates only. It does not "
            "identify a causal language effect if unmeasured properties of the "
            "fake-generation pipelines differ."
        ),
    }

    (OUT_DIR / "overlap_weighted_eer_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    print(f"Wrote results to {OUT_DIR}")


if __name__ == "__main__":
    main()
