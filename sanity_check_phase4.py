#!/usr/bin/env python3
"""
Phase 4 deliverable: "Audio quality sanity-checked"
=====================================================

Run after prepare_hindi_griffinlim.py. Produces:
  - A waveform + log-mel spectrogram comparison grid (real vs. Griffin-Lim fake)
    for a random sample of pairs, saved as PNG.
  - A CSV of per-pair quality flags (near-silent, clipped, duration mismatch,
    non-finite samples) so problem pairs can be excluded before Phase 5.

Usage:
    python sanity_check.py --data_dir ./hindi_griffinlim_eval --n_samples 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


def compute_flags(real: np.ndarray, fake: np.ndarray, sr: int,
                    real_dur: float, fake_dur: float) -> dict:
    def rms(x):
        return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))

    def clip_ratio(x, thresh=0.99):
        return float(np.mean(np.abs(x) >= thresh))

    real_rms, fake_rms = rms(real), rms(fake)
    # prepare_hindi_griffinlim.py RMS-matches each fake to its own paired real
    # clip (see match_amplitude in that script), so these two numbers should be
    # close by construction. A large gap here is a regression signal -- either
    # match_amplitude isn't running as intended, or the safety clamp is engaging
    # unusually often (which would itself be worth investigating: it only
    # triggers when the RMS-matched fake would clip).
    rms_rel_mismatch = abs(fake_rms - real_rms) / max(real_rms, 1e-8)

    return {
        "real_rms": real_rms,
        "fake_rms": fake_rms,
        "rms_rel_mismatch": rms_rel_mismatch,
        "amplitude_mismatch_flag": rms_rel_mismatch > 0.05,
        "real_near_silent": real_rms < 1e-3,
        "fake_near_silent": fake_rms < 1e-3,
        "real_clip_ratio": clip_ratio(real),
        "fake_clip_ratio": clip_ratio(fake),
        "duration_mismatch_s": abs(real_dur - fake_dur),
        "real_nonfinite": bool(not np.all(np.isfinite(real))),
        "fake_nonfinite": bool(not np.all(np.isfinite(fake))),
    }


def spectral_distance(real: np.ndarray, fake: np.ndarray, sr: int,
                        n_fft: int, hop_length: int, n_mels: int) -> float:
    n = min(len(real), len(fake))
    lm_r = librosa.power_to_db(librosa.feature.melspectrogram(
        y=real[:n], sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels))
    lm_f = librosa.power_to_db(librosa.feature.melspectrogram(
        y=fake[:n], sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels))
    m = min(lm_r.shape[1], lm_f.shape[1])
    return float(np.abs(lm_r[:, :m] - lm_f[:, :m]).mean())


def plot_pair(real: np.ndarray, fake: np.ndarray, sr: int, title: str, out_path: Path,
               n_fft: int, hop_length: int, n_mels: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    t_r = np.arange(len(real)) / sr
    t_f = np.arange(len(fake)) / sr

    axes[0, 0].plot(t_r, real, linewidth=0.5)
    axes[0, 0].set_title("Real -- waveform")
    axes[0, 0].set_xlabel("s")

    axes[0, 1].plot(t_f, fake, linewidth=0.5, color="tab:orange")
    axes[0, 1].set_title("Griffin-Lim fake -- waveform")
    axes[0, 1].set_xlabel("s")

    lm_r = librosa.power_to_db(librosa.feature.melspectrogram(
        y=real, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels))
    lm_f = librosa.power_to_db(librosa.feature.melspectrogram(
        y=fake, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels))

    img0 = librosa.display.specshow(lm_r, sr=sr, hop_length=hop_length,
                                      x_axis="time", y_axis="mel", ax=axes[1, 0])
    axes[1, 0].set_title("Real -- log-mel")
    fig.colorbar(img0, ax=axes[1, 0], format="%+2.0f dB")

    img1 = librosa.display.specshow(lm_f, sr=sr, hop_length=hop_length,
                                      x_axis="time", y_axis="mel", ax=axes[1, 1])
    axes[1, 1].set_title("Griffin-Lim fake -- log-mel")
    fig.colorbar(img1, ax=axes[1, 1], format="%+2.0f dB")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, required=True,
                    help="Output dir from prepare_hindi_griffinlim.py (contains metadata.csv)")
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    meta_path = args.data_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found -- run prepare_hindi_griffinlim.py first.")

    metadata = pd.read_csv(meta_path)
    out_dir = args.data_dir / "quality_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    flag_rows = []
    sample = metadata.sample(n=min(args.n_samples, len(metadata)), random_state=args.seed)

    for _, row in sample.iterrows():
        real, sr = sf.read(row["real_path"])
        fake, sr_f = sf.read(row["fake_path"])
        assert sr == sr_f, "sample-rate mismatch between real and fake -- check pipeline"

        flags = compute_flags(real, fake, sr,
                                real_dur=len(real) / sr, fake_dur=len(fake) / sr)
        flags["pair_id"] = row["pair_id"]
        flags["spectral_l1_db"] = spectral_distance(
            real, fake, sr, args.n_fft, args.hop_length, args.n_mels)
        flag_rows.append(flags)

        plot_pair(real, fake, sr, f"pair_{row['pair_id']:04d} (client {row['client_id']})",
                    out_dir / f"pair_{row['pair_id']:04d}.png",
                    args.n_fft, args.hop_length, args.n_mels)

    flags_df = pd.DataFrame(flag_rows)
    flags_df.to_csv(out_dir / "quality_flags.csv", index=False)

    n_problems = (
        flags_df["real_near_silent"].sum() + flags_df["fake_near_silent"].sum()
        + flags_df["real_nonfinite"].sum() + flags_df["fake_nonfinite"].sum()
        + flags_df["amplitude_mismatch_flag"].sum()
        + (flags_df["duration_mismatch_s"] > 0.05).sum()
    )
    print(f"Checked {len(flags_df)} pairs -> {out_dir}")
    print(f"Mean real<->fake log-mel L1 distance: {flags_df['spectral_l1_db'].mean():.2f} dB")
    print(f"Mean real<->fake RMS mismatch: {flags_df['rms_rel_mismatch'].mean() * 100:.1f}% "
          f"(should be small -- fakes are RMS-matched to their paired real clip)")
    print(f"Flagged issues across checked pairs: {int(n_problems)} "
          f"(see quality_flags.csv for detail)")
    if n_problems > 0:
        print("Review flagged pairs before proceeding to Phase 5 -- either exclude them "
              "or loosen/tighten the filters in prepare_hindi_griffinlim.py and regenerate.")


if __name__ == "__main__":
    main()
