#!/usr/bin/env python3
"""
Experiment 1 — Silence-controlled EER sensitivity analysis.

Question:
    Does the English-vs-Hindi detector performance gap survive after
    removing leading/trailing silence from each utterance?

This script reuses:
    - frozen metadata.csv files under the two finalized evaluation datasets
    - the frozen Phase-5 checkpoint/config via phase5_hindi_eval.py

It does NOT modify or regenerate the Phase-5 datasets.

Outputs:
    phase5_outputs/silence_controlled_eer/
        silence_controlled_scores.csv
        silence_controlled_eer_summary.json
        silence_controlled_bootstrap.csv

Primary trimming policy:
    librosa.effects.trim(top_db=40, frame_length=2048, hop_length=512),
    matching Phase 5's silence analysis. Fully-silent clips fall back to
    the original waveform, matching Phase 5.

Inference:
    - same checkpoint
    - same 16 kHz loading
    - same pad() / fixed length
    - same bonafide logit
    - speaker-cluster bootstrap for EER CIs
    - paired raw-vs-trimmed comparison within language
    - Hindi-minus-English EER difference before and after trimming

Run from repository root:
    python exp1_silence_controlled_eer.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from phase5_hindi_eval import (
    CFG,
    OUT,
    compute_eer,
    load_model,
    model_input,
    pad,
)


RAW_EER_FILES = {
    "english": OUT / "english_cv_eer_scores.csv",
    "hindi": OUT / "hindi_eer_scores.csv",
}

DATASETS = {
    "english": CFG.english_cv_dir,
    "hindi": CFG.hindi_dir,
}

SILENCE_TOP_DB = CFG.silence_top_db
FRAME_LENGTH = CFG.silence_frame_length
HOP_LENGTH = CFG.silence_hop_length

OUT_DIR = OUT / "silence_controlled_eer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = CFG.seed + 3001
N_BOOTSTRAP = CFG.n_eer_bootstrap
CI = CFG.ci
SAMPLE_RATE = CFG.sample_rate
DEVICE = CFG.device


def load_metadata(directory: str) -> pd.DataFrame:
    path = Path(directory) / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {"pair_id", "real_path", "fake_path", "duration_s", "client_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    if len(df) != 1125:
        raise ValueError(f"{path}: expected 1125 pairs, found {len(df)}")

    if df.pair_id.duplicated().any():
        raise ValueError(f"{path}: duplicate pair_id")

    valid_client = (
        df.client_id.notna()
        & (df.client_id.astype(str).str.strip() != "")
        & (df.client_id.astype(str).str.lower() != "nan")
    )
    if not valid_client.all():
        raise ValueError(f"{path}: missing client_id values")

    df["pair_id"] = df["pair_id"].astype(str)
    df["analysis_cluster"] = df["client_id"].astype(str)

    return df


def trimmed_waveform(path: str) -> tuple[np.ndarray, int, float, float]:
    x, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    x = np.asarray(x, dtype=np.float32)

    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite samples: {path}")

    trimmed, idx = librosa.effects.trim(
        x,
        top_db=SILENCE_TOP_DB,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )

    if trimmed.size == 0:
        trimmed = x
        idx = np.array([0, len(x)], dtype=int)

    leading_ms = 1000.0 * idx[0] / SAMPLE_RATE
    trailing_ms = 1000.0 * (len(x) - idx[1]) / SAMPLE_RATE
    return trimmed.astype(np.float32), sr, leading_ms, trailing_ms


def build_trimmed_score_rows(model: torch.nn.Module, meta: pd.DataFrame, language: str) -> pd.DataFrame:
    rows = []

    for _, r in tqdm(meta.iterrows(), total=len(meta), desc=f"Trimmed scoring {language}"):
        for label, path in ((1, str(r.real_path)), (0, str(r.fake_path))):
            trimmed, sr, lead_ms, trail_ms = trimmed_waveform(path)
            if sr != SAMPLE_RATE:
                raise ValueError(f"{path}: {sr=} != {SAMPLE_RATE}")

            padded = np.asarray(pad(trimmed, CFG.fixed_len), dtype=np.float32)
            if len(padded) != CFG.fixed_len:
                raise RuntimeError(f"Unexpected padded length for {path}")

            audio = torch.tensor(padded, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                logits = model(audio)

            rows.append(
                {
                    "language": language,
                    "pair_id": str(r.pair_id),
                    "analysis_cluster": str(r.analysis_cluster),
                    "label": int(label),
                    "path": path,
                    "leading_silence_ms": lead_ms,
                    "trailing_silence_ms": trail_ms,
                    "trimmed_duration_s": len(trimmed) / SAMPLE_RATE,
                    "bonafide_logit_trimmed": float(logits[0, 1].item()),
                }
            )

    return pd.DataFrame(rows)


def cluster_eer_bootstrap(
    scores: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> dict:
    groups: Dict[str, Dict[int, np.ndarray]] = {}
    for cluster, g in scores.groupby("analysis_cluster"):
        groups[str(cluster)] = {
            1: g[g.label == 1].bonafide_logit_trimmed.to_numpy(dtype=float),
            0: g[g.label == 0].bonafide_logit_trimmed.to_numpy(dtype=float),
        }

    clusters = list(groups)
    bona = scores[scores.label == 1].bonafide_logit_trimmed.to_numpy(dtype=float)
    spoof = scores[scores.label == 0].bonafide_logit_trimmed.to_numpy(dtype=float)
    observed, threshold = compute_eer(bona, spoof)

    rng = np.random.default_rng(seed)
    vals = []

    for _ in range(n_bootstrap):
        draw = rng.choice(clusters, size=len(clusters), replace=True)
        b = np.concatenate([groups[c][1] for c in draw if groups[c][1].size])
        s = np.concatenate([groups[c][0] for c in draw if groups[c][0].size])

        if b.size == 0 or s.size == 0:
            continue

        vals.append(float(compute_eer(b, s)[0]))

    if not vals:
        raise RuntimeError("No valid bootstrap draws.")

    alpha = (1.0 - CI) / 2.0
    return {
        "eer_pct": float(observed * 100.0),
        "threshold": float(threshold),
        "bootstrap_mean_eer_pct": float(np.mean(vals) * 100.0),
        "ci_lo_pct": float(np.quantile(vals, alpha) * 100.0),
        "ci_hi_pct": float(np.quantile(vals, 1 - alpha) * 100.0),
        "n_utterances": int(len(scores)),
        "n_clusters": int(len(clusters)),
        "bootstrap_unit": "speaker",
        "bootstrap_seed": seed,
        "n_bootstrap": len(vals),
    }


def save_language_summary(scores: pd.DataFrame, language: str) -> dict:
    result = cluster_eer_bootstrap(scores, N_BOOTSTRAP, SEED + (1 if language == "english" else 2))
    result["language"] = language
    return result


def bootstrap_language_gap(
    english: pd.DataFrame,
    hindi: pd.DataFrame,
    trimmed: bool = True,
) -> dict:
    col = "bonafide_logit_trimmed" if trimmed else "bonafide_logit"

    def groups(df: pd.DataFrame):
        return {
            str(s): {
                1: g[g.label == 1][col].to_numpy(dtype=float),
                0: g[g.label == 0][col].to_numpy(dtype=float),
            }
            for s, g in df.groupby("analysis_cluster")
        }

    eg = groups(english)
    hg = groups(hindi)

    eb = english[english.label == 1][col].to_numpy(dtype=float)
    es = english[english.label == 0][col].to_numpy(dtype=float)
    hb = hindi[hindi.label == 1][col].to_numpy(dtype=float)
    hs = hindi[hindi.label == 0][col].to_numpy(dtype=float)

    observed_e = compute_eer(eb, es)[0]
    observed_h = compute_eer(hb, hs)[0]
    observed_diff = (observed_h - observed_e) * 100.0

    rng = np.random.default_rng(SEED + (101 if trimmed else 102))
    diffs = []

    for _ in range(N_BOOTSTRAP):
        ed = rng.choice(list(eg), size=len(eg), replace=True)
        hd = rng.choice(list(hg), size=len(hg), replace=True)

        e_b = np.concatenate([eg[c][1] for c in ed if eg[c][1].size])
        e_s = np.concatenate([eg[c][0] for c in ed if eg[c][0].size])
        h_b = np.concatenate([hg[c][1] for c in hd if hg[c][1].size])
        h_s = np.concatenate([hg[c][0] for c in hd if hg[c][0].size])

        if min(e_b.size, e_s.size, h_b.size, h_s.size) == 0:
            continue

        e_eer = compute_eer(e_b, e_s)[0]
        h_eer = compute_eer(h_b, h_s)[0]
        diffs.append(float((h_eer - e_eer) * 100.0))

    alpha = (1.0 - CI) / 2.0

    return {
        "trimmed": trimmed,
        "observed_hindi_minus_english_eer_pp": observed_diff,
        "ci_lo_pp": float(np.quantile(diffs, alpha)),
        "ci_hi_pp": float(np.quantile(diffs, 1 - alpha)),
        "bootstrap_mean_pp": float(np.mean(diffs)),
        "n_bootstrap": len(diffs),
        "bootstrap_unit": "speaker",
    }


def main() -> None:
    print(f"Using device={DEVICE}")
    model = load_model()

    trimmed_scores = {}
    summaries = []

    for language, directory in DATASETS.items():
        meta = load_metadata(directory)
        scored = build_trimmed_score_rows(model, meta, language)
        trimmed_scores[language] = scored
        scored.to_csv(
            OUT_DIR / f"{language}_trimmed_scores.csv",
            index=False,
        )
        summaries.append(save_language_summary(scored, language))

    english_gap = bootstrap_language_gap(
        trimmed_scores["english"],
        trimmed_scores["hindi"],
        trimmed=True,
    )

    # Raw-vs-trimmed comparison for the already-existing Phase-5 scores.
    raw_summaries = {}
    raw_gap = None

    for language, score_file in RAW_EER_FILES.items():
        if not score_file.exists():
            raise FileNotFoundError(
                f"Missing Phase-5 score file: {score_file}. "
                "Do not rerun Phase 5 automatically; locate the frozen scores first."
            )

        raw = pd.read_csv(score_file)
        if len(raw) != 2250:
            raise ValueError(f"{score_file}: expected 2250 rows, found {len(raw)}")
        raw["analysis_cluster"] = raw["analysis_cluster"].astype(str)
        raw_summaries[language] = raw

    # Raw gap from Phase-5 scores.
    def raw_eer_gap() -> dict:
        e = raw_summaries["english"]
        h = raw_summaries["hindi"]
        e_eer = compute_eer(
            e[e.label == 1].bonafide_logit.to_numpy(),
            e[e.label == 0].bonafide_logit.to_numpy(),
        )[0]
        h_eer = compute_eer(
            h[h.label == 1].bonafide_logit.to_numpy(),
            h[h.label == 0].bonafide_logit.to_numpy(),
        )[0]
        return {
            "english_raw_eer_pct": float(e_eer * 100.0),
            "hindi_raw_eer_pct": float(h_eer * 100.0),
            "raw_hindi_minus_english_pp": float((h_eer - e_eer) * 100.0),
        }

    raw_gap = raw_eer_gap()

    result = {
        "config": {
            "sample_rate": SAMPLE_RATE,
            "top_db": SILENCE_TOP_DB,
            "frame_length": FRAME_LENGTH,
            "hop_length": HOP_LENGTH,
            "fixed_len": CFG.fixed_len,
            "checkpoint_path": CFG.checkpoint_path,
            "seed": SEED,
            "n_bootstrap": N_BOOTSTRAP,
            "ci": CI,
        },
        "raw_phase5_gap": raw_gap,
        "trimmed_language_summaries": summaries,
        "trimmed_hindi_minus_english_gap": english_gap,
        "interpretation": (
            "Primary sensitivity question: whether the language gap persists "
            "after the same deterministic leading/trailing-silence trim is applied "
            "independently to every utterance."
        ),
    }

    (OUT_DIR / "silence_controlled_eer_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))
    print(f"Wrote results to {OUT_DIR}")


if __name__ == "__main__":
    main()
