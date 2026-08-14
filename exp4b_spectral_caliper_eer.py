
#!/usr/bin/env python3
"""
Experiment 4B — Caliper-matched spectral sensitivity analysis.

Secondary sensitivity analysis for Experiment 4A.

Instead of weighting all observations, create a prespecified 1:1 matched
SUBSET of English and Hindi fake pairs using a broad caliper on the
standardized multivariate fake-audio covariates.

Why a subset?
-------------
Unlike full 1:1 matching of all 1125 observations, dropping observations that
have no comparable counterpart gives a real conditional-support analysis.

Default caliper:
    Euclidean distance <= 2.0 in pooled-standardized covariate space.

This threshold is fixed before seeing the EER result. The script also reports
results at stricter secondary calipers 1.0 and 1.5 for sensitivity.

For every accepted English/Hindi fake match, the corresponding real clip from
the same pair is retained.

Outputs:
    phase5_outputs/spectral_caliper_eer/
        matched_subset_*.csv
        balance_*.csv
        caliper_eer_summary.json

Run:
    python exp4b_spectral_caliper_eer.py

This script is intentionally secondary to overlap weighting.
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from phase5_hindi_eval import CFG, OUT, compute_eer, load_model, model_input, score


EN_DIR = Path("english_griffinlim_eval_final")
HI_DIR = Path("hindi_griffinlim_eval_final")

OUT_DIR = OUT / "spectral_caliper_eer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CALIPERS = [1.0, 1.5, 2.0]

SEED = CFG.seed + 6101
N_BOOTSTRAP = CFG.n_eer_bootstrap
CI = CFG.ci

SAMPLE_RATE = CFG.sample_rate
N_FFT = CFG.stft_n_fft
HOP = CFG.stft_hop

FREQ_BANDS = dict(CFG.freq_bands) if hasattr(CFG, "freq_bands") else {
    "low": (0, 1000),
    "mid": (1000, 4000),
    "high": (4000, 8000),
}

COVARIATES = [
    "low_energy_frac", "mid_energy_frac", "high_energy_frac",
    "rms", "peak", "duration_s",
]


def meta(directory: Path, language: str) -> pd.DataFrame:
    df = pd.read_csv(directory / "metadata.csv")
    if len(df) != 1125 or df.pair_id.nunique() != 1125:
        raise ValueError(f"{language}: expected 1125 unique pairs")
    required = {"pair_id", "client_id", "real_path", "fake_path", "duration_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{language}: missing {sorted(missing)}")
    df["pair_id"] = df.pair_id.astype(str)
    df["client_id"] = df.client_id.astype(str)
    df["analysis_cluster"] = df.client_id
    df["language"] = language
    return df


def covariates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Covariates {df.language.iloc[0]}"):
        x, sr = librosa.load(str(r.fake_path), sr=SAMPLE_RATE, mono=True)
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))
        peak = float(np.max(np.abs(x)))

        S = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP, center=False))
        E = S * S
        f = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
        total = float(E.sum() + 1e-12)

        rec = {
            "language": r.language,
            "pair_id": r.pair_id,
            "client_id": r.client_id,
            "duration_s": float(len(x) / SAMPLE_RATE),
            "rms": rms,
            "peak": peak,
        }
        for name, (lo, hi) in FREQ_BANDS.items():
            rec[f"{name}_energy_frac"] = float(E[(f >= lo) & (f < hi)].sum() / total)
        rows.append(rec)
    return pd.DataFrame(rows)


def zscore(en: pd.DataFrame, hi: pd.DataFrame):
    pooled = pd.concat([en[COVARIATES], hi[COVARIATES]], ignore_index=True)
    mu = pooled.mean()
    sd = pooled.std(ddof=1)
    en_z = (en[COVARIATES] - mu) / sd
    hi_z = (hi[COVARIATES] - mu) / sd
    return en_z.to_numpy(float), hi_z.to_numpy(float)


def distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    d2 = np.maximum(aa + bb - 2 * A @ B.T, 0)
    return np.sqrt(d2)


def match_caliper(en: pd.DataFrame, hi: pd.DataFrame, caliper: float):
    A, B = zscore(en, hi)
    D = distance_matrix(A, B)
    BIG = 1e6
    cost = np.where(D <= caliper, D, BIG)
    rows, cols = linear_sum_assignment(cost)

    keep = []
    for i, j in zip(rows, cols):
        if cost[i, j] < BIG:
            keep.append({
                "english_row": int(i),
                "hindi_row": int(j),
                "distance": float(D[i, j]),
            })
    return pd.DataFrame(keep)


def scores_for_subset(model, df, ids):
    rows = []
    subset = df[df.pair_id.isin(set(ids))]
    for _, r in tqdm(subset.iterrows(), total=len(subset), desc=f"Score {df.language.iloc[0]} subset"):
        for label, p in ((1, r.real_path), (0, r.fake_path)):
            x = model_input(str(p))
            rows.append({
                "language": r.language,
                "pair_id": str(r.pair_id),
                "analysis_cluster": str(r.client_id),
                "label": label,
                "bonafide_logit": float(score(model, x)),
            })
    return pd.DataFrame(rows)


def eer(scores):
    return compute_eer(
        scores.loc[scores.label == 1, "bonafide_logit"].to_numpy(),
        scores.loc[scores.label == 0, "bonafide_logit"].to_numpy(),
    )[0]


def bootstrap_eer(scores, seed):
    groups = {str(c): g for c, g in scores.groupby("analysis_cluster")}
    rng = np.random.default_rng(seed)
    vals = []
    clusters = list(groups)
    for _ in range(N_BOOTSTRAP):
        draw = rng.choice(clusters, len(clusters), replace=True)
        s = pd.concat([groups[c] for c in draw], ignore_index=True)
        if s.label.nunique() == 2:
            vals.append(eer(s))
    vals = np.asarray(vals)
    a = (1 - CI) / 2
    return {
        "eer_pct": float(eer(scores) * 100),
        "bootstrap_mean_eer_pct": float(vals.mean() * 100),
        "ci_lo_pct": float(np.quantile(vals, a) * 100),
        "ci_hi_pct": float(np.quantile(vals, 1 - a) * 100),
        "n_utterances": int(len(scores)),
        "n_clusters": int(scores.analysis_cluster.nunique()),
        "bootstrap_unit": "speaker",
        "n_bootstrap": len(vals),
        "seed": seed,
    }


def gap_bootstrap(en, hi, seed):
    eg = {str(c): g for c, g in en.groupby("analysis_cluster")}
    hg = {str(c): g for c, g in hi.groupby("analysis_cluster")}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOTSTRAP):
        ed = rng.choice(list(eg), len(eg), replace=True)
        hd = rng.choice(list(hg), len(hg), replace=True)
        es = pd.concat([eg[c] for c in ed], ignore_index=True)
        hs = pd.concat([hg[c] for c in hd], ignore_index=True)
        if es.label.nunique() == 2 and hs.label.nunique() == 2:
            vals.append((eer(hs) - eer(es)) * 100)
    vals = np.asarray(vals)
    a = (1 - CI) / 2
    return {
        "observed_hindi_minus_english_pp": float((eer(hi) - eer(en)) * 100),
        "bootstrap_mean_pp": float(vals.mean()),
        "ci_lo_pp": float(np.quantile(vals, a)),
        "ci_hi_pp": float(np.quantile(vals, 1 - a)),
        "bootstrap_unit": "speaker",
        "n_bootstrap": len(vals),
        "seed": seed,
    }


def main():
    en_meta = meta(EN_DIR, "english")
    hi_meta = meta(HI_DIR, "hindi")
    en_cov = covariates(en_meta)
    hi_cov = covariates(hi_meta)

    (OUT_DIR / "english_fake_covariates.csv").write_text(en_cov.to_csv(index=False))
    (OUT_DIR / "hindi_fake_covariates.csv").write_text(hi_cov.to_csv(index=False))

    model = load_model()
    results = []

    for k, caliper in enumerate(CALIPERS):
        m = match_caliper(en_cov, hi_cov, caliper)

        en_ids = en_cov.iloc[m.english_row].pair_id.astype(str).tolist()
        hi_ids = hi_cov.iloc[m.hindi_row].pair_id.astype(str).tolist()

        m["english_pair_id"] = en_ids
        m["hindi_pair_id"] = hi_ids

        m.to_csv(OUT_DIR / f"matched_subset_{caliper:.1f}sd.csv", index=False)

        en_scores = scores_for_subset(model, en_meta, en_ids)
        hi_scores = scores_for_subset(model, hi_meta, hi_ids)

        en_scores.to_csv(OUT_DIR / f"english_scores_{caliper:.1f}sd.csv", index=False)
        hi_scores.to_csv(OUT_DIR / f"hindi_scores_{caliper:.1f}sd.csv", index=False)

        e = bootstrap_eer(en_scores, SEED + 10 + k)
        h = bootstrap_eer(hi_scores, SEED + 20 + k)
        g = gap_bootstrap(en_scores, hi_scores, SEED + 30 + k)

        results.append({
            "caliper_sd": caliper,
            "matched_pairs": len(m),
            "english": e,
            "hindi": h,
            "gap": g,
            "median_match_distance": float(m.distance.median()) if len(m) else None,
            "p95_match_distance": float(m.distance.quantile(.95)) if len(m) else None,
        })

    summary = {
        "experiment": "spectral_caliper_matched_eer",
        "covariates": COVARIATES,
        "calipers_sd": CALIPERS,
        "results": results,
        "interpretation": (
            "The caliper analysis is a secondary conditional-support sensitivity "
            "analysis. It asks whether the language EER gap persists among pairs "
            "with close multivariate fake-audio acoustic matches."
        ),
        "caveat": (
            "Changing the caliper changes the target population. The primary "
            "sensitivity analysis is overlap weighting, which retains all pairs."
        ),
    }

    (OUT_DIR / "caliper_eer_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote results to {OUT_DIR}")


if __name__ == "__main__":
    main()
