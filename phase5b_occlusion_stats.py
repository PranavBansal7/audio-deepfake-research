#!/usr/bin/env python3
"""Phase 5b: statistical rigor pass on the occlusion attribution maps.

phase5_hindi_eval.py's run_xai() already computes and saves an occlusion
attribution map per sample, per window size, for every group it evaluates:

    attributions_{tag}/{sample_key}/occlusion_{window_ms}ms.npy

for tag in {cv_en_real, cv_en_fake, cv_hi_real, cv_hi_fake, asvspoof_a11}
and window_ms in CFG.occlusion_windows_ms (default: 20, 50, 100).

Those maps are never turned into the same statistical objects the
Integrated-Gradients analysis gets (compare_features / explanation_drift_
statistics.csv): no cluster-bootstrap CIs, no significance tests, no BH
correction, and no explicit real-vs-fake / cross-language breakdown. That is
exactly what an ad-hoc script run at the terminal produced -- point estimates
only, with an unclear real/fake subset and no real-audio baseline.

This script closes that gap without re-running the (expensive) occlusion
computation itself. It:

  1. Loads the existing occlusion_{w}ms.npy maps + the faithfulness_{tag}.csv
     manifests phase5_hindi_eval.py already wrote.
  2. Extracts the same feature set used for the IG analysis (low/mid/high
     frequency-band attribution fraction, speech/silence fraction, temporal
     concentration: gini, entropy, top-k%) by reusing freq_proxy /
     speech_silence / temporal directly from phase5_hindi_eval.py, so the
     numbers are computed identically to the IG side and are directly
     comparable.
  3. Reports explicit per-group (language x real/fake) descriptive stats
     with cluster-bootstrap CIs -- the "clear subsets" the ad-hoc script
     was missing (occlusion_group_summary_{w}ms.csv).
  4. Runs the same four comparisons compare_features() runs for IG, at each
     window size, with cluster-bootstrap CIs, Mann-Whitney U / Wilcoxon
     significance tests, and BH-FDR correction per comparison family:
       - cross_language_fake   (English-fake vs Hindi-fake)
       - cross_language_real   (English-real vs Hindi-real)  <- the
         "real audio occlusion baseline": if the language-driven spectral
         shift shows up here too, it isn't fake-specific.
       - english_fake_minus_real, hindi_fake_minus_real (paired, within
         language)
  5. Checks sign/significance consistency across the three window sizes
     (20ms is treated as primary/pre-registered; 50/100ms are robustness
     checks, not additionally corrected against the primary window).
  6. Optionally re-runs the comprehensiveness/sufficiency faithfulness check
     -- previously only computed for IG -- on the occlusion maps themselves,
     so you can see directly whether occlusion is more trustworthy than IG
     for the subgroup where IG failed (Hindi-fake). This requires loading
     the model for new forward passes; skip with --skip-faithfulness if you
     only want the statistics in items 3-5.
  7. Cross-checks the primary-window occlusion drift stats against the
     existing explanation_drift_statistics.csv (IG) for direction/
     significance agreement.

Requires phase5_hindi_eval.py to have been run to completion first (occlusion
maps + faithfulness_*.csv must already exist under CFG.out_dir). Run from the
same repository root as phase5_hindi_eval.py so `from phase5_hindi_eval import
...` resolves, exactly as phase5_hindi_eval.py itself must be run from the
repo root for its own imports (model.py, data_utils_SSL.py, eval_metric_LA.py).

Note on out_dir: CFG.out_dir in the uploaded phase5_hindi_eval.py is
"phase5_outputs_v2". If your occlusion maps actually live under
"phase5_outputs" (an earlier run), either rename the directory or edit
Config.out_dir in phase5_hindi_eval.py before running this script -- it reads
CFG.out_dir directly, so a mismatch here will surface as "missing required
faithfulness_*.csv" below.
"""
from __future__ import annotations

import argparse
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

from phase5_hindi_eval import (
    CFG,
    OUT,
    LOGGER,
    raw_audio,
    model_input,
    freq_proxy,
    speech_silence,
    temporal,
    cluster_bootstrap_mean_diff,
    paired_bootstrap,
    bh,
    faithfulness,
    load_model,
)

OCC_OUT = OUT / "occlusion_analysis"
OCC_OUT.mkdir(parents=True, exist_ok=True)

REQUIRED_GROUPS = ["cv_en_real", "cv_en_fake", "cv_hi_real", "cv_hi_fake"]
OPTIONAL_GROUPS = ["asvspoof_a11"]

PRIMARY_WINDOW_MS = CFG.occlusion_windows_ms[0]

FEATURE_NAMES = [
    "low_frac", "mid_frac", "high_frac",
    "speech_frac", "silence_frac",
    "gini", "entropy",
    "topk_1pct", "topk_5pct", "topk_10pct",
]


# ------------------------------ loading -------------------------------------

def load_group_manifest(tag: str) -> pd.DataFrame | None:
    path = OUT / f"faithfulness_{tag}.csv"
    if not path.exists():
        LOGGER.warning("Skipping group %r: %s not found.", tag, path)
        return None
    df = pd.read_csv(path)
    required = {"sample_key", "pair_id", "analysis_cluster", "label", "path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df


def occlusion_path(tag: str, sample_key: str, window_ms: int):
    return OUT / f"attributions_{tag}" / str(sample_key) / f"occlusion_{window_ms}ms.npy"


# --------------------------- feature extraction ------------------------------

def extract_occlusion_features(manifest: pd.DataFrame, tag: str, window_ms: int) -> pd.DataFrame:
    """Mirrors extract_features() in phase5_hindi_eval.py, but reads
    occlusion_{window_ms}ms.npy instead of integrated_gradients.npy. Reuses
    freq_proxy / speech_silence / temporal unchanged so results are directly
    comparable to the IG-based explanation_features_*.csv files."""
    rows = []
    missing = 0
    bad_shape = 0
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc=f"Occlusion features {tag} @ {window_ms}ms"):
        p = occlusion_path(tag, r.sample_key, window_ms)
        if not p.exists():
            missing += 1
            continue
        attr = np.load(p)
        if attr.shape != (CFG.fixed_len,):
            bad_shape += 1
            continue
        raw = raw_audio(str(r.path))
        padded = model_input(str(r.path))
        rec = {
            "sample_key": str(r.sample_key),
            "pair_id": str(r.pair_id),
            "analysis_cluster": str(r.analysis_cluster),
            "label": int(r.label),
            "path": str(r.path),
            "window_ms": window_ms,
        }
        rec.update(freq_proxy(padded, attr))
        rec.update(speech_silence(raw, attr))
        rec.update(temporal(attr))
        rows.append(rec)
    if missing or bad_shape:
        LOGGER.warning(
            "%s @ %dms: %d/%d samples missing occlusion_%dms.npy, %d with unexpected shape (both skipped). "
            "Re-run phase5_hindi_eval.py's run_xai() to backfill missing maps.",
            tag, window_ms, missing, len(manifest), window_ms, bad_shape,
        )
    out = pd.DataFrame(rows)
    out.to_csv(OCC_OUT / f"occlusion_features_{tag}_{window_ms}ms.csv", index=False)
    return out


# --------------------------- descriptive subsets ------------------------------

def group_summary(features: Dict[str, pd.DataFrame], window_ms: int) -> pd.DataFrame:
    """Explicit per-group (language x real/fake) descriptive stats with
    cluster-bootstrap CIs on the mean. This is the "clear subsets" piece the
    ad-hoc terminal script didn't have: every number here is unambiguously
    tagged by language and real/fake."""
    rows = []
    offset = window_ms * 10_000
    for tag, df in features.items():
        if df.empty:
            continue
        for feat in FEATURE_NAMES:
            offset += 1
            vals = df[feat].dropna()
            if len(vals) < 2:
                continue
            groups = {
                str(s): g[feat].dropna().to_numpy()
                for s, g in df.groupby("analysis_cluster")
                if g[feat].notna().any()
            }
            keys = list(groups)
            rng = np.random.default_rng(CFG.seed + offset)
            boots = []
            for _ in range(CFG.n_bootstrap):
                draw = rng.choice(keys, len(keys), replace=True)
                parts = [groups[k] for k in draw if len(groups[k])]
                if not parts:
                    continue
                boots.append(float(np.concatenate(parts).mean()))
            alpha = (1 - CFG.ci) / 2
            rows.append({
                "group": tag, "window_ms": window_ms, "feature": feat,
                "n_samples": int(len(vals)), "n_clusters": len(keys),
                "mean": float(vals.mean()), "median": float(vals.median()),
                "ci_lo": float(np.quantile(boots, alpha)) if boots else np.nan,
                "ci_hi": float(np.quantile(boots, 1 - alpha)) if boots else np.nan,
            })
    out = pd.DataFrame(rows)
    out.to_csv(OCC_OUT / f"occlusion_group_summary_{window_ms}ms.csv", index=False)
    return out


# ------------------------------ significance ----------------------------------

def compare_occlusion_features(features: Dict[str, pd.DataFrame], window_ms: int) -> pd.DataFrame:
    """Same four comparisons compare_features() runs for IG, applied to the
    occlusion features at a single window size: cluster-bootstrap CIs,
    Mann-Whitney U (independent groups) or Wilcoxon signed-rank (paired),
    and BH-FDR correction within each comparison family.

    cross_language_real is the real-audio occlusion baseline: it tells you
    whether the English-vs-Hindi spectral shift seen in cross_language_fake
    is fake-specific or just a general property of the two languages'
    acoustics (in which case it would also show up in real audio)."""
    independent = [
        ("cross_language_fake", "cv_en_fake", "cv_hi_fake"),
        ("cross_language_real", "cv_en_real", "cv_hi_real"),  # real-audio baseline
    ]
    paired = [
        ("english_fake_minus_real", "cv_en_real", "cv_en_fake"),
        ("hindi_fake_minus_real", "cv_hi_real", "cv_hi_fake"),
    ]
    rows = []
    for comp, akey, bkey in independent:
        if akey not in features or bkey not in features:
            continue
        a, b = features[akey], features[bkey]
        if a.empty or b.empty:
            continue
        for feat in FEATURE_NAMES:
            av, bv = a[feat].dropna(), b[feat].dropna()
            if len(av) < 2 or len(bv) < 2:
                continue
            boot = cluster_bootstrap_mean_diff(a, b, feat, offset=window_ms * 1000 + len(rows))
            _, p = stats.mannwhitneyu(av, bv, alternative="two-sided")
            pooled = math.sqrt((np.var(av, ddof=1) + np.var(bv, ddof=1)) / 2)
            d = boot["observed"] / pooled if pooled > 0 else np.nan
            rows.append({
                "window_ms": window_ms, "comparison": comp, "feature": feat,
                "mean_a": float(av.mean()), "mean_b": float(bv.mean()),
                "mean_difference_b_minus_a": boot["observed"],
                "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"],
                "p_value": float(p), "effect_size": float(d),
                "test": "Mann-Whitney U + cluster bootstrap",
                "bootstrap_unit": boot["bootstrap_unit"],
            })
    for comp, real_key, fake_key in paired:
        if real_key not in features or fake_key not in features:
            continue
        if features[real_key].empty or features[fake_key].empty:
            continue
        merged = pd.concat([features[real_key], features[fake_key]], ignore_index=True)
        for feat in FEATURE_NAMES:
            wide = merged.pivot_table(index="pair_id", columns="label", values=feat, aggfunc="mean").dropna()
            if len(wide) < 3:
                continue
            diff = wide[0].to_numpy() - wide[1].to_numpy()
            try:
                _, p = stats.wilcoxon(diff, alternative="two-sided")
            except ValueError:
                p = 1.0
            boot = paired_bootstrap(merged, feat, offset=window_ms * 1000 + len(rows) + 1000)
            d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else np.nan
            rows.append({
                "window_ms": window_ms, "comparison": comp, "feature": feat,
                "mean_a": float(wide[1].mean()), "mean_b": float(wide[0].mean()),
                "mean_difference_b_minus_a": boot["observed_fake_minus_real"],
                "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"],
                "p_value": float(p), "effect_size": float(d),
                "test": "Wilcoxon signed-rank + pair bootstrap",
                "bootstrap_unit": "pair",
            })
    out = pd.DataFrame(rows)
    out["q_value_bh"] = np.nan
    for comp in out.comparison.unique():
        m = out.comparison == comp
        out.loc[m, "q_value_bh"] = bh(out.loc[m, "p_value"].to_numpy())
    out.to_csv(OCC_OUT / f"occlusion_drift_statistics_{window_ms}ms.csv", index=False)
    return out


# --------------------------- cross-window consistency --------------------------

def cross_window_consistency(combined: pd.DataFrame) -> pd.DataFrame:
    """Is the direction (and BH-significance) of each comparison x feature
    effect stable across the 20/50/100ms occlusion window sizes, or is it an
    artifact of one particular window choice? 20ms is treated as primary;
    the others are robustness checks, not additionally corrected against it."""
    rows = []
    for (comp, feat), g in combined.groupby(["comparison", "feature"]):
        g = g.sort_values("window_ms")
        signs = np.sign(g.mean_difference_b_minus_a.to_numpy())
        rows.append({
            "comparison": comp, "feature": feat, "n_windows": len(g),
            "sign_consistent_across_windows": bool(np.all(signs == signs[0])) if len(signs) else False,
            "significant_q05_at_all_windows": bool((g.q_value_bh < 0.05).all()),
            "windows_ms": g.window_ms.tolist(),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OCC_OUT / "occlusion_cross_window_consistency.csv", index=False)
    return out


# --------------------- occlusion attribution faithfulness ----------------------

def occlusion_faithfulness_for_group(model, manifest: pd.DataFrame, tag: str, window_ms: int) -> pd.DataFrame:
    """Comprehensiveness/sufficiency of the occlusion map itself, using the
    same faithfulness() routine phase5_hindi_eval.py applies to IG. Included
    because IG failed this check for cv_hi_fake (~0 comprehensiveness, ~50%
    positive) -- this tells you directly whether occlusion is more
    trustworthy for that subgroup before leaning on it as the primary
    explanation method.

    Caveat: comprehensiveness is close to true by construction for occlusion
    (the top-ranked positions are exactly those whose removal changed the
    score the most, at this same window size), so a strong score here is
    expected and less informative than the same score is for IG. Included
    for completeness/parity, not as fully independent validation.

    Crash-resilient / resumable, same pattern as run_xai() and
    silence_dataset() in phase5_hindi_eval.py."""
    csv_path = OCC_OUT / f"occlusion_faithfulness_{tag}_{window_ms}ms.csv"
    old = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    done = set(old.sample_key.astype(str)) if not old.empty else set()
    rows = old.to_dict("records") if not old.empty else []
    pending = 0
    for _, r in tqdm(manifest.iterrows(), total=len(manifest), desc=f"Occlusion faithfulness {tag} @ {window_ms}ms"):
        key = str(r.sample_key)
        if key in done:
            continue
        p = occlusion_path(tag, key, window_ms)
        if not p.exists():
            continue
        attr = np.load(p)
        if attr.shape != (CFG.fixed_len,):
            LOGGER.warning("%s: unexpected occlusion map shape %s for %s @ %dms; skipping.", tag, attr.shape, key, window_ms)
            continue
        audio = model_input(str(r.path))
        rec = {
            "sample_key": key, "pair_id": str(r.pair_id),
            "analysis_cluster": str(r.analysis_cluster), "label": int(r.label),
            "path": str(r.path), "window_ms": window_ms,
        }
        for frac in CFG.faithfulness_fracs:
            c, s = faithfulness(model, audio, attr, frac)
            pct = int(frac * 100)
            rec[f"occ_comp_{pct}"] = c
            rec[f"occ_suff_{pct}"] = s
        rows.append(rec)
        pending += 1
        if pending >= 25:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            pending = 0
    out = pd.DataFrame(rows).drop_duplicates("sample_key", keep="last").reset_index(drop=True)
    out.to_csv(csv_path, index=False)
    return out


def summarize_occlusion_faithfulness(all_faith: Dict[Tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (tag, w), df in all_faith.items():
        if df.empty:
            continue
        for frac in CFG.faithfulness_fracs:
            pct = int(frac * 100)
            c, s = df[f"occ_comp_{pct}"], df[f"occ_suff_{pct}"]
            rows.append({
                "group": tag, "window_ms": w, "frac_pct": pct,
                "comp_mean": float(c.mean()), "comp_median": float(c.median()),
                "comp_frac_positive": float((c > 0).mean()),
                "suff_mean": float(s.mean()), "suff_median": float(s.median()),
                "suff_frac_positive": float((s > 0).mean()),
                "n": int(len(df)),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OCC_OUT / "occlusion_faithfulness_summary.csv", index=False)
    return out


# ---------------------------- IG cross-check ------------------------------------

def compare_to_ig(occlusion_primary: pd.DataFrame) -> pd.DataFrame | None:
    """Does the primary-window (20ms) occlusion result agree in direction and
    significance with the existing IG-based explanation_drift_statistics.csv?
    IG failed its own faithfulness check for cv_hi_fake, so this tells you
    whether the IG story (larger spectral shift for Hindi fakes) survives
    switching to a method that doesn't share IG's failure mode."""
    ig_path = OUT / "explanation_drift_statistics.csv"
    if not ig_path.exists():
        LOGGER.warning("IG comparison table not found at %s; skipping IG-vs-occlusion cross-check.", ig_path)
        return None
    ig = pd.read_csv(ig_path)
    merged = occlusion_primary.merge(
        ig[["comparison", "feature", "mean_difference_b_minus_a", "ci_lo", "ci_hi", "p_value", "q_value_bh"]],
        on=["comparison", "feature"], suffixes=("_occlusion", "_ig"), how="inner",
    )
    merged["direction_agrees"] = np.sign(merged.mean_difference_b_minus_a_occlusion) == np.sign(merged.mean_difference_b_minus_a_ig)
    merged["both_significant_q05"] = (merged.q_value_bh_occlusion < 0.05) & (merged.q_value_bh_ig < 0.05)
    merged.to_csv(OCC_OUT / "occlusion_vs_ig_comparison.csv", index=False)
    n, agree = len(merged), int(merged.direction_agrees.sum())
    LOGGER.info("IG vs occlusion (window=%dms) direction agreement: %d/%d feature-comparisons.", PRIMARY_WINDOW_MS, agree, n)
    return merged


# ------------------------------------ main --------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 5b: occlusion statistical analysis")
    parser.add_argument("--skip-faithfulness", action="store_true",
                         help="Skip the occlusion comprehensiveness/sufficiency check (skips model loading; "
                              "the CI/significance/subset/baseline analysis still runs).")
    parser.add_argument("--no-a11", action="store_true",
                         help="Exclude the ASVspoof-A11 sampled reference group even if its manifest exists.")
    args = parser.parse_args()

    LOGGER.info("=== Phase 5b: occlusion statistical analysis (reads %s) ===", OUT)
    if not OUT.exists():
        raise RuntimeError(
            f"{OUT} does not exist. This script reuses artifacts written by phase5_hindi_eval.py's "
            "run_xai(); run that first, or check Config.out_dir in phase5_hindi_eval.py if your results "
            "live under a different directory name (e.g. 'phase5_outputs' vs 'phase5_outputs_v2')."
        )

    groups_to_load = list(REQUIRED_GROUPS)
    if not args.no_a11:
        groups_to_load += OPTIONAL_GROUPS

    manifests: Dict[str, pd.DataFrame] = {}
    for tag in groups_to_load:
        m = load_group_manifest(tag)
        if m is not None:
            manifests[tag] = m

    missing_required = [g for g in REQUIRED_GROUPS if g not in manifests]
    if missing_required:
        raise RuntimeError(
            f"Missing required faithfulness_*.csv for groups {missing_required} under {OUT}. "
            "Run phase5_hindi_eval.py to completion first (it writes these alongside the "
            "occlusion_{w}ms.npy attribution maps this script reads)."
        )

    model = None
    if not args.skip_faithfulness:
        try:
            model = load_model()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not load model for occlusion faithfulness check (%s); skipping that section.", exc)
            model = None

    all_drift = []
    all_faith: Dict[Tuple[str, int], pd.DataFrame] = {}
    for w in CFG.occlusion_windows_ms:
        LOGGER.info("--- occlusion window = %d ms%s ---", w, " (primary)" if w == PRIMARY_WINDOW_MS else "")
        features = {tag: extract_occlusion_features(m, tag, w) for tag, m in manifests.items()}
        group_summary(features, w)
        drift = compare_occlusion_features(features, w)
        drift["is_primary_window"] = drift.window_ms == PRIMARY_WINDOW_MS
        all_drift.append(drift)

        if model is not None:
            for tag, m in manifests.items():
                all_faith[(tag, w)] = occlusion_faithfulness_for_group(model, m, tag, w)

    combined = pd.concat(all_drift, ignore_index=True)
    combined.to_csv(OCC_OUT / "occlusion_drift_statistics_all_windows.csv", index=False)

    cross_window_consistency(combined)

    if model is not None and all_faith:
        summarize_occlusion_faithfulness(all_faith)

    primary_drift = combined[combined.window_ms == PRIMARY_WINDOW_MS].copy()
    compare_to_ig(primary_drift)

    LOGGER.info("Phase 5b occlusion analysis complete: %s", OCC_OUT)


if __name__ == "__main__":
    main()
