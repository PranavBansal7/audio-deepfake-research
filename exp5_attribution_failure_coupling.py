#!/usr/bin/env python3
"""
Experiment 5 — Attribution–Failure Coupling.

Reads frozen Phase-5 explanation features and RAW detector scores. No model/XAI
inference is performed.

Primary question:
    Among fake clips, are unusual explanation patterns associated with detector
    difficulty, and is that association stronger for Hindi than English?

Fake difficulty:
    bonafide_logit - language-specific raw EER threshold
Higher = harder spoof (more bonafide-like).

Primary explanation features:
    low_frac, mid_frac, high_frac, gini, entropy, top-k fractions.

Outputs:
    phase5_outputs/attribution_failure_coupling/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path("phase5_outputs")
OUT = BASE / "attribution_failure_coupling"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 11235
N_BOOTSTRAP = 2000
CI = 0.95

FEATURES = [
    "low_frac", "mid_frac", "high_frac",
    "gini", "entropy",
    "topk_1pct", "topk_5pct", "topk_10pct",
]
ENERGY_FEATURES = ["low_energy_frac", "mid_energy_frac", "high_energy_frac"]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--english_scores", type=Path, default=None)
    p.add_argument("--hindi_scores", type=Path, default=None)
    p.add_argument("--english_features", type=Path, default=None)
    p.add_argument("--hindi_features", type=Path, default=None)
    p.add_argument("--spectral_energy", action="store_true",
                   help="Merge existing spectral-energy decomposition features when available.")
    return p.parse_args()


def find_feature(lang: str) -> Path:
    if lang == "english":
        names = [
            BASE / "explanation_features_cv_en_fake.csv",
            BASE / "faithfulness_cv_en_fake.csv",
        ]
        keys = ("en", "english")
    else:
        names = [
            BASE / "explanation_features_cv_hi_fake.csv",
            BASE / "faithfulness_cv_hi_fake.csv",
        ]
        keys = ("hi", "hindi")

    for p in names:
        if p.exists():
            return p

    for p in BASE.rglob("*.csv"):
        n = p.name.lower()
        if "fake" not in n or not any(k in n for k in keys):
            continue
        try:
            cols = set(pd.read_csv(p, nrows=2).columns)
        except Exception:
            continue
        if {"pair_id", "analysis_cluster", "low_frac", "mid_frac", "high_frac"} <= cols:
            return p

    raise FileNotFoundError(
        f"Could not find fake explanation features for {lang}. "
        f"Pass --{lang}_features explicitly."
    )


def find_scores(lang: str) -> Path:
    if lang == "english":
        explicit = [
            BASE / "english_cv_eer_scores.csv",
            BASE / "scores_english_cv.csv",
            BASE / "eer_scores_english_cv.csv",
        ]
        keys = ("english", "en")
    else:
        explicit = [
            BASE / "hindi_eer_scores.csv",
            BASE / "scores_hindi_cv.csv",
            BASE / "eer_scores_hindi.csv",
        ]
        keys = ("hindi", "hi")

    for p in explicit:
        if p.exists():
            return p

    candidates = []
    for p in BASE.rglob("*.csv"):
        n = p.name.lower()
        if "overlap_weighted" in n or "weighted_scores" in n:
            continue
        if not any(k in n for k in keys):
            continue
        try:
            df = pd.read_csv(p, nrows=3)
            cols = set(df.columns)
            if not {"pair_id", "label", "bonafide_logit"} <= cols:
                continue
            full = pd.read_csv(p, usecols=["pair_id", "label", "bonafide_logit"])
            candidates.append((abs(len(full) - 2250), p))
        except Exception:
            continue

    if not candidates:
        raise FileNotFoundError(
            f"Could not find RAW score CSV for {lang}. Pass --{lang}_scores explicitly."
        )
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"pair_id", "analysis_cluster", "low_frac", "mid_frac", "high_frac"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    df["pair_id"] = df["pair_id"].astype(str)
    df["analysis_cluster"] = df["analysis_cluster"].astype(str)
    return df


def load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"pair_id", "label", "bonafide_logit"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    df["pair_id"] = df["pair_id"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def eer_threshold(scores: pd.DataFrame) -> float:
    bona = scores.loc[scores.label == 1, "bonafide_logit"].to_numpy(float)
    spoof = scores.loc[scores.label == 0, "bonafide_logit"].to_numpy(float)
    thresholds = np.unique(np.concatenate([bona, spoof]))
    best = None
    for t in thresholds:
        fnr = np.mean(bona < t)
        fpr = np.mean(spoof >= t)
        eer = 0.5 * (fnr + fpr)
        key = (abs(fnr - fpr), eer, t)
        if best is None or key < best:
            best = key
    return float(best[2])


def spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan, np.nan
    r, p = stats.spearmanr(x[mask], y[mask])
    return float(r), float(p)


def bootstrap_rho(df: pd.DataFrame, feature: str, seed: int):
    clusters = list(df.analysis_cluster.astype(str).unique())
    groups = {c: g for c, g in df.groupby(df.analysis_cluster.astype(str))}
    r0, p0 = spearman(df.difficulty.to_numpy(float), df[feature].to_numpy(float))

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOTSTRAP):
        draw = rng.choice(clusters, len(clusters), replace=True)
        s = pd.concat([groups[c] for c in draw], ignore_index=True)
        r, _ = spearman(s.difficulty.to_numpy(float), s[feature].to_numpy(float))
        if np.isfinite(r):
            vals.append(r)

    vals = np.asarray(vals)
    a = (1 - CI) / 2
    return {
        "rho": r0,
        "p_spearman": p0,
        "ci_lo": float(np.quantile(vals, a)),
        "ci_hi": float(np.quantile(vals, 1-a)),
        "n_samples": len(df),
        "n_clusters": len(clusters),
        "n_bootstrap": len(vals),
        "bootstrap_unit": "speaker",
        "seed": seed,
    }


def bootstrap_rho_difference(en: pd.DataFrame, hi: pd.DataFrame, feature: str, seed: int):
    eg = {str(c): g for c, g in en.groupby(en.analysis_cluster.astype(str))}
    hg = {str(c): g for c, g in hi.groupby(hi.analysis_cluster.astype(str))}
    ec, hc = list(eg), list(hg)

    r_en = spearman(en.difficulty.to_numpy(float), en[feature].to_numpy(float))[0]
    r_hi = spearman(hi.difficulty.to_numpy(float), hi[feature].to_numpy(float))[0]
    observed = r_en - r_hi

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOTSTRAP):
        ed = rng.choice(ec, len(ec), replace=True)
        hd = rng.choice(hc, len(hc), replace=True)
        es = pd.concat([eg[c] for c in ed], ignore_index=True)
        hs = pd.concat([hg[c] for c in hd], ignore_index=True)
        re = spearman(es.difficulty.to_numpy(float), es[feature].to_numpy(float))[0]
        rh = spearman(hs.difficulty.to_numpy(float), hs[feature].to_numpy(float))[0]
        if np.isfinite(re) and np.isfinite(rh):
            vals.append(re - rh)

    vals = np.asarray(vals)
    a = (1 - CI) / 2
    # Bootstrap two-sided sign probability around zero.
    p = 2 * min(np.mean(vals <= 0), np.mean(vals >= 0))
    return {
        "english_minus_hindi_rho": float(observed),
        "ci_lo": float(np.quantile(vals, a)),
        "ci_hi": float(np.quantile(vals, 1-a)),
        "p_bootstrap_two_sided": float(min(1.0, p)),
        "n_bootstrap": len(vals),
        "bootstrap_unit": "speaker",
        "seed": seed,
    }


def discover_energy_table() -> Optional[Path]:
    candidate_dirs = [
        BASE / "spectral_energy_decomposition",
        BASE / "spectral_decomposition",
    ]
    for d in candidate_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.csv"):
            try:
                cols = set(pd.read_csv(p, nrows=2).columns)
            except Exception:
                continue
            if {"pair_id", *ENERGY_FEATURES} <= cols:
                return p
    for p in BASE.rglob("*.csv"):
        if "overlap" in p.name.lower():
            continue
        try:
            cols = set(pd.read_csv(p, nrows=2).columns)
        except Exception:
            continue
        if {"pair_id", *ENERGY_FEATURES} <= cols:
            return p
    return None


def partial_spearman(df: pd.DataFrame, feature: str):
    cols = ["difficulty", feature] + ENERGY_FEATURES
    x = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 12:
        return np.nan, np.nan, len(x)

    r = x.rank(method="average")
    Z = np.column_stack([np.ones(len(r)), r[ENERGY_FEATURES].to_numpy(float)])

    def resid(name):
        y = r[name].to_numpy(float)
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        return y - Z @ beta

    rx, ry = resid("difficulty"), resid(feature)
    rho, p = stats.pearsonr(rx, ry)
    return float(rho), float(p), len(x)


def main():
    a = args()

    en_feat_path = a.english_features or find_feature("english")
    hi_feat_path = a.hindi_features or find_feature("hindi")
    en_score_path = a.english_scores or find_scores("english")
    hi_score_path = a.hindi_scores or find_scores("hindi")

    print("English features:", en_feat_path)
    print("Hindi features  :", hi_feat_path)
    print("English scores  :", en_score_path)
    print("Hindi scores    :", hi_score_path)

    en_feat = load_features(en_feat_path)
    hi_feat = load_features(hi_feat_path)
    en_scores = load_scores(en_score_path)
    hi_scores = load_scores(hi_score_path)

    en_thr = eer_threshold(en_scores)
    hi_thr = eer_threshold(hi_scores)

    # Fake-only. Difficulty increases when a fake receives a more
    # bonafide-like score.
    en_sc = en_scores[en_scores.label == 0].copy()
    hi_sc = hi_scores[hi_scores.label == 0].copy()

    en_sc["difficulty"] = en_sc.bonafide_logit - en_thr
    hi_sc["difficulty"] = hi_sc.bonafide_logit - hi_thr

    # Keep score-table speaker clusters if available; otherwise obtain them
    # from the Phase-5 feature manifests.
    en_sc["pair_id"] = en_sc["pair_id"].astype(str)
    hi_sc["pair_id"] = hi_sc["pair_id"].astype(str)

    en = en_feat.merge(
        en_sc[["pair_id", "bonafide_logit", "difficulty"]],
        on="pair_id", how="inner", validate="one_to_one"
    )
    hi = hi_feat.merge(
        hi_sc[["pair_id", "bonafide_logit", "difficulty"]],
        on="pair_id", how="inner", validate="one_to_one"
    )

    en["language"] = "english"
    hi["language"] = "hindi"

    en.to_csv(OUT / "english_fake_merged.csv", index=False)
    hi.to_csv(OUT / "hindi_fake_merged.csv", index=False)

    rows = []
    seed = SEED

    for lang, df in [("english", en), ("hindi", hi)]:
        for feature in FEATURES:
            if feature not in df.columns:
                continue
            d = df[["difficulty", feature, "analysis_cluster"]].dropna()
            if d.analysis_cluster.nunique() < 8:
                continue
            out = bootstrap_rho(d, feature, seed)
            seed += 1
            rows.append({
                "language": lang,
                "feature": feature,
                **out,
            })

    corr = pd.DataFrame(rows)
    corr.to_csv(OUT / "difficulty_explanation_correlations.csv", index=False)

    # Compare the English and Hindi correlation strengths.
    compare_rows = []
    for feature in FEATURES:
        if feature not in en.columns or feature not in hi.columns:
            continue
        e = en[["difficulty", feature, "analysis_cluster"]].dropna()
        h = hi[["difficulty", feature, "analysis_cluster"]].dropna()
        if len(e) < 10 or len(h) < 10:
            continue
        out = bootstrap_rho_difference(e, h, feature, seed)
        seed += 1
        compare_rows.append({"feature": feature, **out})

    comp = pd.DataFrame(compare_rows)
    comp.to_csv(OUT / "english_vs_hindi_rho_difference.csv", index=False)

    # Optional acoustic control: use the existing spectral decomposition table
    # if requested and available.
    partial_rows = []
    energy_path = discover_energy_table() if a.spectral_energy else None

    if energy_path is not None:
        energy = pd.read_csv(energy_path)
        energy["pair_id"] = energy["pair_id"].astype(str)

        for lang, df in [("english", en), ("hindi", hi)]:
            m = df.merge(
                energy[["pair_id"] + ENERGY_FEATURES],
                on="pair_id", how="inner"
            )
            for feature in ["low_frac", "mid_frac", "high_frac"]:
                rho, p, n = partial_spearman(m, feature)
                partial_rows.append({
                    "language": lang,
                    "feature": feature,
                    "partial_spearman_rho": rho,
                    "p_value": p,
                    "n": n,
                    "controls": ",".join(ENERGY_FEATURES),
                    "energy_table": str(energy_path),
                })

    partial = pd.DataFrame(partial_rows)
    if not partial.empty:
        partial.to_csv(OUT / "partial_spearman_acoustic_control.csv", index=False)

    headline = {}
    for lang, df in [("english", en), ("hindi", hi)]:
        d = df.dropna(subset=["difficulty", "high_frac"])
        r, p = spearman(d.difficulty.to_numpy(float), d.high_frac.to_numpy(float))
        headline[lang] = {
            "n_fake": len(df),
            "n_speakers": int(df.analysis_cluster.nunique()),
            "eer_threshold": en_thr if lang == "english" else hi_thr,
            "difficulty_high_frac_rho": r,
            "difficulty_high_frac_p": p,
            "difficulty_mean": float(df.difficulty.mean()),
            "difficulty_median": float(df.difficulty.median()),
        }

    summary = {
        "experiment": "attribution_failure_coupling",
        "question": (
            "Within fake clips, are unusual explanation patterns associated "
            "with detector difficulty, and does that association differ between "
            "English and Hindi?"
        ),
        "difficulty_definition": (
            "bonafide_logit - raw language-specific EER threshold; "
            "higher means the fake is more bonafide-like and harder to detect"
        ),
        "english_threshold": en_thr,
        "hindi_threshold": hi_thr,
        "headline": headline,
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_unit": "speaker",
        "spectral_partial_control": bool(energy_path is not None),
    }

    (OUT / "attribution_failure_coupling_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== HEADLINE ===")
    print(json.dumps(summary, indent=2))
    print("\nOutputs:", OUT)


if __name__ == "__main__":
    main()