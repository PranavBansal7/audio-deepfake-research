#!/usr/bin/env python3
"""
Experiment 3 — Decompose the frequency-attribution proxy against ordinary
spectral energy.

Motivation:
    Phase 5's "frequency attribution" is explicitly an energy-weighted proxy,
    not true frequency-domain attribution. This experiment asks whether the
    observed English/Hindi attribution shift can be explained simply by
    differences in ordinary spectral energy.

For each XAI sample and band:
    energy_frac = global STFT energy in band
    proxy_frac  = Phase-5 frequency attribution proxy in band
    excess_frac = proxy_frac - energy_frac

Interpretation:
    - If proxy shift >> energy shift, the attribution proxy contains information
      beyond simple global spectral composition.
    - If proxy and energy shifts are nearly identical, the attribution finding
      is strongly confounded by acoustics.

Analyses:
    1. Explicit group summaries for real/fake x language.
    2. Cross-language fake differences in energy vs proxy.
    3. Real->fake differences within language.
    4. Difference-in-differences for energy and proxy.
    5. Speaker-cluster bootstrap CIs.
    6. Spearman correlation between energy_frac and proxy_frac.
    7. Excess attribution = proxy_frac - energy_frac.

Uses the SAME 100-pair XAI samples and SAME STFT settings as Phase 5:
    n_fft=512, hop=160, sample_rate=16000, center=False.

No model inference and no XAI recomputation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

from phase5_hindi_eval import CFG, OUT, model_input, raw_audio, freq_proxy


FEATURES = ["low", "mid", "high"]
BANDS = {
    "low": (0, 1000),
    "mid": (1000, 4000),
    "high": (4000, 8000),
}

OUT_DIR = OUT / "spectral_energy_decomposition"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = CFG.seed + 5001
N_BOOTSTRAP = CFG.n_bootstrap
CI = CFG.ci


def spectral_energy_fractions(audio: np.ndarray) -> dict:
    S = np.abs(
        librosa.stft(
            audio,
            n_fft=CFG.stft_n_fft,
            hop_length=CFG.stft_hop,
            center=False,
        )
    )
    freqs = librosa.fft_frequencies(
        sr=CFG.sample_rate,
        n_fft=CFG.stft_n_fft,
    )

    energy = S * S
    total = float(np.sum(energy) + 1e-12)

    out = {}
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[f"{band}_energy_frac"] = float(np.sum(energy[mask]) / total)

    return out


def load_manifest(tag: str) -> pd.DataFrame:
    path = OUT / f"faithfulness_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)

    required = {"sample_key", "pair_id", "analysis_cluster", "label", "path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    df["sample_key"] = df["sample_key"].astype(str)
    df["pair_id"] = df["pair_id"].astype(str)
    df["analysis_cluster"] = df["analysis_cluster"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def extract_group(tag: str) -> pd.DataFrame:
    manifest = load_manifest(tag)
    attr_dir = OUT / f"attributions_{tag}"

    rows = []

    for _, r in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc=f"Spectral decomposition {tag}",
    ):
        attr_path = attr_dir / r.sample_key / "integrated_gradients.npy"
        if not attr_path.exists():
            raise FileNotFoundError(attr_path)

        attr = np.load(attr_path)
        if attr.shape != (CFG.fixed_len,):
            raise ValueError(
                f"{attr_path}: unexpected shape {attr.shape}, "
                f"expected {(CFG.fixed_len,)}"
            )

        raw = raw_audio(str(r.path))
        padded = model_input(str(r.path))

        energy = spectral_energy_fractions(padded)
        proxy = freq_proxy(padded, attr)

        rec = {
            "tag": tag,
            "sample_key": r.sample_key,
            "pair_id": r.pair_id,
            "analysis_cluster": r.analysis_cluster,
            "label": int(r.label),
            "path": str(r.path),
        }

        for band in FEATURES:
            rec[f"{band}_energy_frac"] = energy[f"{band}_energy_frac"]
            rec[f"{band}_proxy_frac"] = proxy[f"{band}_frac"]
            rec[f"{band}_excess_frac"] = (
                proxy[f"{band}_frac"] - energy[f"{band}_energy_frac"]
            )

        rows.append(rec)

    return pd.DataFrame(rows)


def bootstrap_group_mean(
    df: pd.DataFrame,
    column: str,
    seed_offset: int,
) -> dict:
    groups = {
        str(s): g[column].dropna().to_numpy(dtype=float)
        for s, g in df.groupby("analysis_cluster")
        if g[column].notna().any()
    }

    clusters = list(groups)
    observed = float(df[column].mean())

    rng = np.random.default_rng(SEED + seed_offset)
    vals = []

    for _ in range(N_BOOTSTRAP):
        draw = rng.choice(clusters, len(clusters), replace=True)
        x = np.concatenate([groups[s] for s in draw])
        vals.append(float(x.mean()))

    alpha = (1.0 - CI) / 2.0

    return {
        "mean": observed,
        "ci_lo": float(np.quantile(vals, alpha)),
        "ci_hi": float(np.quantile(vals, 1 - alpha)),
        "n_clusters": len(clusters),
        "n_samples": int(df[column].notna().sum()),
        "bootstrap_unit": "speaker",
    }


def bootstrap_difference(
    a: pd.DataFrame,
    b: pd.DataFrame,
    column: str,
    seed_offset: int,
) -> dict:
    ag = {
        str(s): g[column].dropna().to_numpy(dtype=float)
        for s, g in a.groupby("analysis_cluster")
        if g[column].notna().any()
    }
    bg = {
        str(s): g[column].dropna().to_numpy(dtype=float)
        for s, g in b.groupby("analysis_cluster")
        if g[column].notna().any()
    }

    observed = float(b[column].mean() - a[column].mean())

    rng = np.random.default_rng(SEED + seed_offset)
    vals = []

    for _ in range(N_BOOTSTRAP):
        ad = rng.choice(list(ag), len(ag), replace=True)
        bd = rng.choice(list(bg), len(bg), replace=True)

        av = np.concatenate([ag[s] for s in ad])
        bv = np.concatenate([bg[s] for s in bd])
        vals.append(float(bv.mean() - av.mean()))

    alpha = (1.0 - CI) / 2.0
    return {
        "observed": observed,
        "ci_lo": float(np.quantile(vals, alpha)),
        "ci_hi": float(np.quantile(vals, 1 - alpha)),
        "n_bootstrap": len(vals),
    }


def paired_diff(df_real: pd.DataFrame, df_fake: pd.DataFrame, column: str) -> pd.DataFrame:
    r = (
        df_real.groupby(["analysis_cluster", "pair_id"], as_index=False)[column]
        .mean()
    )
    f = (
        df_fake.groupby(["analysis_cluster", "pair_id"], as_index=False)[column]
        .mean()
    )
    m = r.merge(
        f,
        on=["analysis_cluster", "pair_id"],
        suffixes=("_real", "_fake"),
        how="inner",
        validate="one_to_one",
    ).dropna()

    m["fake_minus_real"] = m[f"{column}_fake"] - m[f"{column}_real"]
    return m


def bootstrap_interaction(
    en_diff: pd.DataFrame,
    hi_diff: pd.DataFrame,
    column_name: str,
    seed_offset: int,
) -> dict:
    eg = {
        str(s): g["fake_minus_real"].to_numpy(dtype=float)
        for s, g in en_diff.groupby("analysis_cluster")
        if len(g)
    }
    hg = {
        str(s): g["fake_minus_real"].to_numpy(dtype=float)
        for s, g in hi_diff.groupby("analysis_cluster")
        if len(g)
    }

    observed = float(
        hi_diff["fake_minus_real"].mean()
        - en_diff["fake_minus_real"].mean()
    )

    rng = np.random.default_rng(SEED + seed_offset)
    vals = []

    for _ in range(N_BOOTSTRAP):
        ed = rng.choice(list(eg), len(eg), replace=True)
        hd = rng.choice(list(hg), len(hg), replace=True)

        ev = np.concatenate([eg[s] for s in ed])
        hv = np.concatenate([hg[s] for s in hd])
        vals.append(float(hv.mean() - ev.mean()))

    alpha = (1.0 - CI) / 2.0
    return {
        "interaction": observed,
        "ci_lo": float(np.quantile(vals, alpha)),
        "ci_hi": float(np.quantile(vals, 1 - alpha)),
        "n_bootstrap": len(vals),
        "bootstrap_unit": "speaker",
    }


def main() -> None:
    tags = {
        "english_real": "cv_en_real",
        "english_fake": "cv_en_fake",
        "hindi_real": "cv_hi_real",
        "hindi_fake": "cv_hi_fake",
    }

    groups = {name: extract_group(tag) for name, tag in tags.items()}

    all_features = pd.concat(groups.values(), ignore_index=True)
    all_features.to_csv(
        OUT_DIR / "spectral_energy_decomposition_per_sample.csv",
        index=False,
    )

    # 1. Descriptive group-level summaries.
    summary_rows = []

    for name, df in groups.items():
        for band in FEATURES:
            for kind in ("energy_frac", "proxy_frac", "excess_frac"):
                col = f"{band}_{kind}"
                boot = bootstrap_group_mean(
                    df,
                    col,
                    seed_offset=10_000 + len(summary_rows),
                )
                summary_rows.append(
                    {
                        "group": name,
                        "band": band,
                        "quantity": kind,
                        **boot,
                    }
                )

    group_summary = pd.DataFrame(summary_rows)
    group_summary.to_csv(
        OUT_DIR / "spectral_energy_group_summary.csv",
        index=False,
    )

    # 2. Cross-language fake difference and real baseline.
    comparison_rows = []

    comparisons = [
        ("cross_language_fake", groups["english_fake"], groups["hindi_fake"]),
        ("cross_language_real", groups["english_real"], groups["hindi_real"]),
    ]

    for comp, a, b in comparisons:
        for band in FEATURES:
            for kind in ("energy_frac", "proxy_frac", "excess_frac"):
                col = f"{band}_{kind}"
                boot = bootstrap_difference(
                    a,
                    b,
                    col,
                    seed_offset=20_000 + len(comparison_rows),
                )
                comparison_rows.append(
                    {
                        "comparison": comp,
                        "band": band,
                        "quantity": kind,
                        "mean_a": float(a[col].mean()),
                        "mean_b": float(b[col].mean()),
                        "b_minus_a": boot["observed"],
                        "ci_lo": boot["ci_lo"],
                        "ci_hi": boot["ci_hi"],
                        "n_a": int(a[col].notna().sum()),
                        "n_b": int(b[col].notna().sum()),
                        "bootstrap_unit": "speaker",
                    }
                )

    # 3. Within-language real->fake and difference-in-differences.
    for band in FEATURES:
        for kind in ("energy_frac", "proxy_frac", "excess_frac"):
            col = f"{band}_{kind}"

            en = paired_diff(groups["english_real"], groups["english_fake"], col)
            hi = paired_diff(groups["hindi_real"], groups["hindi_fake"], col)

            en_boot = bootstrap_group_mean(
                en.rename(columns={"fake_minus_real": "value"}),
                "value",
                seed_offset=30_000 + len(comparison_rows),
            )
            hi_boot = bootstrap_group_mean(
                hi.rename(columns={"fake_minus_real": "value"}),
                "value",
                seed_offset=31_000 + len(comparison_rows),
            )

            interaction = bootstrap_interaction(
                en.rename(columns={"fake_minus_real": "value"})
                .rename(columns={"value": "fake_minus_real"}),
                hi.rename(columns={"fake_minus_real": "value"})
                .rename(columns={"value": "fake_minus_real"}),
                "fake_minus_real",
                seed_offset=32_000 + len(comparison_rows),
            )

            comparison_rows.extend(
                [
                    {
                        "comparison": "english_fake_minus_real",
                        "band": band,
                        "quantity": kind,
                        "mean_a": float(en["fake_minus_real"].mean() * 0 + 0),
                        "mean_b": float(en["fake_minus_real"].mean()),
                        "b_minus_a": float(en["fake_minus_real"].mean()),
                        "ci_lo": en_boot["ci_lo"],
                        "ci_hi": en_boot["ci_hi"],
                        "n_a": int(len(en)),
                        "n_b": int(len(en)),
                        "bootstrap_unit": "speaker",
                    },
                    {
                        "comparison": "hindi_fake_minus_real",
                        "band": band,
                        "quantity": kind,
                        "mean_a": 0.0,
                        "mean_b": float(hi["fake_minus_real"].mean()),
                        "b_minus_a": float(hi["fake_minus_real"].mean()),
                        "ci_lo": hi_boot["ci_lo"],
                        "ci_hi": hi_boot["ci_hi"],
                        "n_a": int(len(hi)),
                        "n_b": int(len(hi)),
                        "bootstrap_unit": "speaker",
                    },
                    {
                        "comparison": "language_x_reconstruction_interaction",
                        "band": band,
                        "quantity": kind,
                        "mean_a": float(en["fake_minus_real"].mean()),
                        "mean_b": float(hi["fake_minus_real"].mean()),
                        "b_minus_a": float(interaction["interaction"]),
                        "ci_lo": interaction["ci_lo"],
                        "ci_hi": interaction["ci_hi"],
                        "n_a": int(len(en)),
                        "n_b": int(len(hi)),
                        "bootstrap_unit": "speaker",
                    },
                ]
            )

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(
        OUT_DIR / "spectral_energy_comparisons.csv",
        index=False,
    )

    # 4. Correlation between ordinary energy and attribution proxy.
    corr_rows = []
    for name, df in groups.items():
        for band in FEATURES:
            e = df[f"{band}_energy_frac"]
            p = df[f"{band}_proxy_frac"]
            mask = e.notna() & p.notna()

            rho, pval = stats.spearmanr(e[mask], p[mask])

            corr_rows.append(
                {
                    "group": name,
                    "band": band,
                    "spearman_rho": float(rho),
                    "p_value": float(pval),
                    "n": int(mask.sum()),
                }
            )

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(
        OUT_DIR / "spectral_energy_proxy_correlations.csv",
        index=False,
    )

    manifest = {
        "question": (
            "Does the frequency-attribution proxy shift exceed what can be "
            "explained by ordinary spectral-energy differences?"
        ),
        "stft_n_fft": CFG.stft_n_fft,
        "stft_hop": CFG.stft_hop,
        "sample_rate": CFG.sample_rate,
        "bands_hz": BANDS,
        "samples_per_group": {
            k: int(len(v)) for k, v in groups.items()
        },
        "speaker_clusters": {
            k: int(v.analysis_cluster.nunique()) for k, v in groups.items()
        },
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "ci": CI,
        "primary_quantity": "proxy_frac",
        "secondary_quantity": "excess_frac = proxy_frac - energy_frac",
        "warning": (
            "The proxy is not true frequency-domain attribution. It combines "
            "temporal attribution with spectral energy. This experiment tests "
            "the extent to which ordinary energy composition accounts for the "
            "observed proxy shift."
        ),
    }

    (OUT_DIR / "spectral_energy_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Wrote results to", OUT_DIR)


if __name__ == "__main__":
    main()
