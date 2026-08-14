#!/usr/bin/env python3
"""
Phase 4/5: English dataset sanity check + artifact generation
==============================================================

Run AFTER prepare_english_griffinlim.py on the completed English
CommonVoice + Griffin-Lim dataset.

This script is the English counterpart of the Hindi quality-check script.
It scans the audio folders directly -- it does not trust or require a
pre-existing metadata.csv -- so the artifacts describe the data that
actually exists on disk.

Expected layout
---------------
    <data_dir>/
        real_english/
        fake_english_griffinlim/
        metadata.csv                  <- regenerated from the audio
        prepare.log                   <- full verification log
        quality_check/
            quality_flags.csv         <- per-pair quality flags for ALL pairs
            pair_XXXX.png              <- waveform + log-mel grids for a sample

Quality model
-------------
Fakes are RMS-matched to their paired real clip UNLESS that match would
exceed the 0.99 digital ceiling, in which case they are smoothly
peak-normalized instead.

Therefore:

  * rms_rel_mismatch > 5% is EXPECTED for peak-normalized pairs. It is
    reported as the inferred normalization method, not as a defect.
    The amplitude_mismatch_flag column is retained for compatibility with
    older analysis scripts.
  * The true defect signal is CLIPPING: flat-topped samples at the waveform
    ceiling or any samples at/above 0.99.
  * Near-silent, non-finite, and duration-mismatch conditions are also
    reported.

Usage
-----
    python sanity_check_english.py \
        --data_dir ./english_griffinlim_eval_final

    # Optionally carry over descriptive fields from the synthesis metadata:
    python sanity_check_english.py \
        --data_dir ./english_griffinlim_eval_final \
        --old_metadata ./english_griffinlim_eval_final/metadata.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe on DGX / SSH sessions

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def clip_ratio(x: np.ndarray, thresh: float = 0.99) -> float:
    return float(np.mean(np.abs(x) >= thresh))


def count_flat_tops(
    audio: np.ndarray,
    threshold: float = 0.989,
    tolerance: float = 1e-5,
) -> int:
    """Count samples stuck at the absolute peak ceiling."""
    peaks = np.abs(audio[np.abs(audio) >= threshold])
    if len(peaks) == 0:
        return 0
    max_val = np.max(peaks)
    return int(np.sum(np.abs(peaks - max_val) < tolerance))


def compute_pair_metrics(
    real: np.ndarray,
    fake: np.ndarray,
    sr: int,
    rms_tolerance: float,
    flat_top_tolerance: int,
) -> dict:
    real_rms, fake_rms = rms(real), rms(fake)
    rms_rel_mismatch = abs(fake_rms - real_rms) / max(real_rms, 1e-8)

    real_flats = count_flat_tops(real)
    fake_flats = count_flat_tops(fake)

    inferred_norm = (
        "rms_matched"
        if rms_rel_mismatch <= rms_tolerance
        else "peak_normalized"
    )

    real_clip = clip_ratio(real)
    fake_clip = clip_ratio(fake)

    return {
        "real_rms": real_rms,
        "fake_rms": fake_rms,
        "rms_rel_mismatch": rms_rel_mismatch,
        "amplitude_mismatch_flag": rms_rel_mismatch > rms_tolerance,
        "inferred_normalization": inferred_norm,
        "real_peak": float(np.max(np.abs(real))),
        "fake_peak": float(np.max(np.abs(fake))),
        "real_clip_ratio": real_clip,
        "fake_clip_ratio": fake_clip,
        "real_flat_tops": real_flats,
        "fake_flat_tops": fake_flats,
        "clipping_flag": bool(
            fake_flats > flat_top_tolerance or fake_clip > 0.0
        ),
        "real_near_silent": real_rms < 1e-3,
        "fake_near_silent": fake_rms < 1e-3,
        "duration_mismatch_s": abs(len(real) - len(fake)) / sr,
        "real_nonfinite": bool(not np.all(np.isfinite(real))),
        "fake_nonfinite": bool(not np.all(np.isfinite(fake))),
    }


def spectral_distance(
    real: np.ndarray,
    fake: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> float:
    n = min(len(real), len(fake))
    lm_r = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=real[:n],
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
    )
    lm_f = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=fake[:n],
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
    )
    m = min(lm_r.shape[1], lm_f.shape[1])
    return float(np.abs(lm_r[:, :m] - lm_f[:, :m]).mean())


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def plot_pair(
    real: np.ndarray,
    fake: np.ndarray,
    sr: int,
    title: str,
    out_path: Path,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))

    t_r = np.arange(len(real)) / sr
    t_f = np.arange(len(fake)) / sr

    axes[0, 0].plot(t_r, real, linewidth=0.5)
    axes[0, 0].set_title("Real -- waveform")
    axes[0, 0].set_xlabel("s")

    axes[0, 1].plot(t_f, fake, linewidth=0.5, color="tab:orange")
    axes[0, 1].set_title("Griffin-Lim fake -- waveform")
    axes[0, 1].set_xlabel("s")

    lm_r = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=real,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
    )
    lm_f = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=fake,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
    )

    img0 = librosa.display.specshow(
        lm_r,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Real -- log-mel")
    fig.colorbar(img0, ax=axes[1, 0], format="%+2.0f dB")

    img1 = librosa.display.specshow(
        lm_f,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Griffin-Lim fake -- log-mel")
    fig.colorbar(img1, ax=axes[1, 1], format="%+2.0f dB")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("phase4_sanity_english")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        "%H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def log_environment(logger: logging.Logger) -> None:
    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "librosa": librosa.__version__,
        "soundfile": sf.__version__,
    }

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
        info["cuda_version"] = torch.version.cuda
    except ImportError:
        info["torch"] = None

    logger.info("Environment: " + json.dumps(info))


def scan_pairs(data_dir: Path, logger: logging.Logger) -> list:
    """Scan English real/fake folders and return paired files.

    Hard-fails on unpaired files so the dataset cannot silently proceed with
    inconsistent real/fake membership.
    """
    real_dir = data_dir / "real_english"
    fake_dir = data_dir / "fake_english_griffinlim"

    for d in (real_dir, fake_dir):
        if not d.exists():
            raise FileNotFoundError(
                f"Expected directory {d} -- check --data_dir."
            )

    def index_by_id(folder: Path, prefix: str) -> dict:
        out = {}
        for f in sorted(folder.glob(f"{prefix}_*.wav")):
            m = re.match(rf"{prefix}_(\d+)\.wav$", f.name)
            if m:
                out[int(m.group(1))] = f
        return out

    reals = index_by_id(real_dir, "english_real")
    fakes = index_by_id(fake_dir, "english_fake")

    only_real = sorted(set(reals) - set(fakes))
    only_fake = sorted(set(fakes) - set(reals))

    if only_real or only_fake:
        raise RuntimeError(
            f"Unpaired files on disk: {len(only_real)} real-only ids "
            f"{only_real[:10]}{'...' if len(only_real) > 10 else ''}, "
            f"{len(only_fake)} fake-only ids "
            f"{only_fake[:10]}{'...' if len(only_fake) > 10 else ''}. "
            f"Fix the dataset before generating artifacts."
        )

    if not reals:
        raise RuntimeError(
            f"No english_real_*.wav files found in {real_dir}."
        )

    logger.info(
        f"Found {len(reals)} paired real/fake files on disk "
        f"(ids {min(reals):04d}-{max(reals):04d})."
    )
    return [(pid, reals[pid], fakes[pid]) for pid in sorted(reals)]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Dataset dir containing real_english/ and fake_english_griffinlim/.",
    )
    p.add_argument(
        "--old_metadata",
        type=Path,
        default=None,
        help=(
            "Optional metadata.csv from the original English synthesis run. "
            "Descriptive fields (client_id, sentence, gender, age, accents, "
            "vote_margin, source_clip) are carried over; measured fields are "
            "recomputed from audio on disk."
        ),
    )
    p.add_argument(
        "--n_plot_samples",
        type=int,
        default=8,
        help="How many pairs to render waveform/log-mel PNG grids for.",
    )
    p.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Expected sample rate; mismatch is flagged.",
    )
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument(
        "--rms_tolerance",
        type=float,
        default=0.05,
        help=(
            "Relative RMS gap above which a pair is classified as "
            "peak_normalized (expected fallback, not a defect)."
        ),
    )
    p.add_argument(
        "--flat_top_tolerance",
        type=int,
        default=2,
        help=(
            "Max samples allowed at the waveform ceiling before a fake "
            "is flagged as clipped."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    qc_dir = args.data_dir / "quality_check"
    qc_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(args.data_dir / "prepare.log")

    logger.info(
        f"Config: {json.dumps(vars(args), default=str, indent=2)}"
    )
    log_environment(logger)

    pairs = scan_pairs(args.data_dir, logger)

    old_meta = None
    if args.old_metadata is not None:
        if not args.old_metadata.exists():
            raise FileNotFoundError(
                f"--old_metadata {args.old_metadata} not found."
            )
        old_meta = pd.read_csv(args.old_metadata).set_index("pair_id")
        logger.info(
            f"Loaded descriptive fields from {args.old_metadata} "
            f"({len(old_meta)} rows)."
        )
    else:
        logger.info(
            "No --old_metadata given; metadata.csv will contain measured "
            "fields only (descriptive columns empty)."
        )

    descriptive_cols = [
        "client_id",
        "sentence",
        "gender",
        "age",
        "accents",
        "vote_margin",
        "source_clip",
    ]

    flag_rows = []
    meta_rows = []
    t_start = time.time()

    for pair_id, real_path, fake_path in tqdm(
        pairs,
        desc="Analyzing English pairs",
    ):
        real, sr_r = sf.read(str(real_path))
        fake, sr_f = sf.read(str(fake_path))

        if sr_r != args.sample_rate or sr_f != args.sample_rate:
            logger.error(
                f"pair {pair_id}: sample-rate mismatch "
                f"(real={sr_r}, fake={sr_f}, expected={args.sample_rate})"
            )

        if sr_r != sr_f:
            logger.error(
                f"pair {pair_id}: real/fake sample rates differ -- "
                f"check pipeline"
            )

        m = compute_pair_metrics(
            real,
            fake,
            sr_r,
            args.rms_tolerance,
            args.flat_top_tolerance,
        )
        m["pair_id"] = pair_id
        m["spectral_l1_db"] = spectral_distance(
            real,
            fake,
            sr_r,
            args.n_fft,
            args.hop_length,
            args.n_mels,
        )
        flag_rows.append(m)

        meta_row = {
            "pair_id": pair_id,
            "real_path": str(real_path),
            "fake_path": str(fake_path),
            "duration_s": len(real) / sr_r,
            "real_duration_s": len(real) / sr_r,
            "fake_duration_s": len(fake) / sr_f,
            "real_rms": m["real_rms"],
            "fake_rms": m["fake_rms"],
            "real_peak": m["real_peak"],
            "fake_peak": m["fake_peak"],
            "inferred_normalization": m["inferred_normalization"],
        }

        if old_meta is not None and pair_id in old_meta.index:
            for c in descriptive_cols:
                meta_row[c] = (
                    old_meta.loc[pair_id, c]
                    if c in old_meta.columns
                    else ""
                )
        else:
            for c in descriptive_cols:
                meta_row[c] = ""

        meta_rows.append(meta_row)

    elapsed = time.time() - t_start

    logger.info(
        f"Analyzed {len(pairs)} pairs in {elapsed:.1f}s "
        f"({len(pairs) / elapsed:.1f} pairs/sec)."
    )

    # ---- artifacts -------------------------------------------------------- #

    meta_cols = (
        [
            "pair_id",
            "client_id",
            "real_path",
            "fake_path",
            "duration_s",
            "sentence",
            "gender",
            "age",
            "accents",
            "vote_margin",
            "source_clip",
        ]
        + [
            "real_duration_s",
            "fake_duration_s",
            "real_rms",
            "fake_rms",
            "real_peak",
            "fake_peak",
            "inferred_normalization",
        ]
    )

    metadata = pd.DataFrame(meta_rows)
    metadata = metadata.reindex(columns=meta_cols)
    metadata.to_csv(args.data_dir / "metadata.csv", index=False)

    logger.info(
        f"Wrote {args.data_dir / 'metadata.csv'} ({len(metadata)} rows)."
    )

    flags_df = pd.DataFrame(flag_rows)
    flags_df.to_csv(qc_dir / "quality_flags.csv", index=False)

    logger.info(
        f"Wrote {qc_dir / 'quality_flags.csv'} "
        f"({len(flags_df)} rows, ALL pairs)."
    )

    rng = np.random.RandomState(args.seed)
    plot_ids = set(
        rng.choice(
            [pid for pid, _, _ in pairs],
            size=min(args.n_plot_samples, len(pairs)),
            replace=False,
        ).tolist()
    )

    for pair_id, real_path, fake_path in tqdm(
        [p for p in pairs if p[0] in plot_ids],
        desc="Plotting samples",
    ):
        real, sr = sf.read(str(real_path))
        fake, _ = sf.read(str(fake_path))

        plot_pair(
            real,
            fake,
            sr,
            f"pair_{pair_id:04d}",
            qc_dir / f"pair_{pair_id:04d}.png",
            args.n_fft,
            args.hop_length,
            args.n_mels,
        )

    logger.info(
        f"Wrote {len(plot_ids)} waveform/log-mel grids to {qc_dir}."
    )

    # ---- summary ---------------------------------------------------------- #

    n_peak_norm = int(
        (flags_df["inferred_normalization"] == "peak_normalized").sum()
    )
    n_clipped = int(flags_df["clipping_flag"].sum())
    n_near_silent = int(
        flags_df["real_near_silent"].sum()
        + flags_df["fake_near_silent"].sum()
    )
    n_nonfinite = int(
        flags_df["real_nonfinite"].sum()
        + flags_df["fake_nonfinite"].sum()
    )
    n_dur_mismatch = int(
        (flags_df["duration_mismatch_s"] > 0.05).sum()
    )
    n_defects = (
        n_clipped
        + n_near_silent
        + n_nonfinite
        + n_dur_mismatch
    )

    logger.info("=" * 70)
    logger.info(
        f"DONE. Verified {len(flags_df)} English pairs directly from disk."
    )
    logger.info(
        f"Mean real<->fake log-mel L1 distance: "
        f"{flags_df['spectral_l1_db'].mean():.2f} dB "
        f"(max {flags_df['spectral_l1_db'].max():.2f} dB)"
    )
    logger.info(
        f"Normalization: {len(flags_df) - n_peak_norm} RMS-matched, "
        f"{n_peak_norm} peak-normalized fallback "
        f"(expected, not defects)."
    )

    rms_matched = flags_df[
        flags_df["inferred_normalization"] == "rms_matched"
    ]

    if len(rms_matched) > 0:
        logger.info(
            "Mean RMS mismatch among RMS-matched pairs: "
            f"{rms_matched['rms_rel_mismatch'].mean() * 100:.2f}%"
        )
    else:
        logger.info(
            "Mean RMS mismatch among RMS-matched pairs: N/A "
            "(no RMS-matched pairs found)."
        )

    logger.info(
        f"Defects: {n_clipped} clipped (flat tops / samples >= 0.99), "
        f"{n_near_silent} near-silent, {n_nonfinite} non-finite, "
        f"{n_dur_mismatch} duration mismatches > 50ms."
    )

    if n_clipped == 0 and n_defects == 0:
        logger.info(
            "English dataset is clean: no clipping or other defects detected. "
            "Ready for Phase 5."
        )
    else:
        logger.warning(
            f"{n_defects} defect(s) detected -- inspect "
            f"{qc_dir / 'quality_flags.csv'} and exclude or re-synthesize "
            f"affected pairs before Phase 5."
        )


if __name__ == "__main__":
    main()
