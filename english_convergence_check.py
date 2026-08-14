#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sweep_english_inv_mel_iters.py

English equivalent of the Hindi InverseMelScale convergence sweep.

Uses the EXISTING English 1,125-example selection:
    <eval_dir>/selected_sources.csv

For a fixed random subset of those clips, compare:

    GPU:
        torchaudio MelSpectrogram
        -> InverseMelScale (SGD)
        -> GriffinLim

against the independent librosa reference:

        librosa melspectrogram
        -> librosa.feature.inverse.mel_to_audio

Metric:
    mean absolute difference between the two reconstructed
    log-mel spectrograms in dB.

Usage:

    python sweep_english_inv_mel_iters.py \
        --cv_dir /path/to/cv_english_subset/en \
        --eval_dir /path/to/english_griffinlim_eval_final \
        --n_samples 10

For a full plateau check:

    python sweep_english_inv_mel_iters.py \
        --cv_dir /path/to/cv_english_subset/en \
        --eval_dir /path/to/english_griffinlim_eval_final \
        --n_samples 30 \
        --iters 1000 2000 2250 2500 3000 3500 4000 5000
"""

import argparse
import csv
import json
import random
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio.transforms as T


def logmel_l1_db(
    a: np.ndarray,
    b: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    n_mels: int,
) -> float:
    """
    Mean absolute difference between log-mel spectrograms in dB.

    Uses librosa.power_to_db with default ref=1.0,
    matching the Hindi convergence validation.
    """

    mel_a = librosa.feature.melspectrogram(
        y=a,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        power=2.0,
    )

    mel_b = librosa.feature.melspectrogram(
        y=b,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        power=2.0,
    )

    db_a = librosa.power_to_db(mel_a)
    db_b = librosa.power_to_db(mel_b)

    m = min(db_a.shape[1], db_b.shape[1])

    return float(
        np.mean(np.abs(db_a[:, :m] - db_b[:, :m]))
    )


def librosa_reference(
    wav: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    n_mels: int,
    gl_iters: int,
) -> np.ndarray:
    """
    Independent librosa pseudo-inverse reference.
    """

    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        power=2.0,
    )

    return librosa.feature.inverse.mel_to_audio(
        mel,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_iter=gl_iters,
        power=2.0,
        length=len(wav),
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cv_dir",
        type=Path,
        required=True,
        help="English Common Voice directory containing clips/",
    )

    parser.add_argument(
        "--eval_dir",
        type=Path,
        required=True,
        help="english_griffinlim_eval_final directory",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=30,
        help="Number of English examples to use",
    )

    parser.add_argument(
        "--iters",
        type=int,
        nargs="+",
        default=[1000, 2000, 2250, 2500, 3000, 3500, 4000, 5000],
        help="InverseMelScale iteration budgets",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    config_path = args.eval_dir / "config.json"

    with open(config_path) as f:
        cfg = json.load(f)

    sr = cfg["sample_rate"]
    n_fft = cfg["n_fft"]
    hop = cfg["hop_length"]
    n_mels = cfg["n_mels"]

    inv_mel_lr = cfg["inv_mel_lr"]
    inv_mel_momentum = cfg["inv_mel_momentum"]

    gl_iters = cfg["gl_n_iter"]
    gl_momentum = cfg["gl_momentum"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---------------------------------------------------------
    # Load EXISTING 1,125-example English selection
    # ---------------------------------------------------------

    selection_csv = args.eval_dir / "selected_sources.csv"

    with open(selection_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Existing English selection: {len(rows)} clips")

    if args.n_samples > len(rows):
        raise ValueError(
            f"--n_samples={args.n_samples}, "
            f"but only {len(rows)} selected English clips exist."
        )

    # Same fixed subset for every iteration budget
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[:args.n_samples]

    # ---------------------------------------------------------
    # Transform objects
    # ---------------------------------------------------------

    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        power=2.0,
    ).to(device)

    # ---------------------------------------------------------
    # Load waveforms + compute independent references ONCE
    # ---------------------------------------------------------

    waveforms = []
    references = []
    names = []

    for row in rows:

        wav_path = Path(row["path"])

        if not wav_path.exists():
            print(f"WARNING: missing {wav_path}")
            continue

        wav, _ = librosa.load(
            str(wav_path),
            sr=sr,
            mono=True,
        )

        wav = wav.astype(np.float32)

        reference = librosa_reference(
            wav,
            sr,
            n_fft,
            hop,
            n_mels,
            gl_iters,
        )

        waveforms.append(wav)
        references.append(reference)
        names.append(row["path"])

    print(
        f"Using {len(waveforms)} English clips "
        f"on device={device}"
    )

    print()
    print(
        f"{'max_iter':>10} | "
        f"{'mean logmel L1 dB':>20} | "
        f"{'min':>10} | "
        f"{'max':>10}"
    )
    print("-" * 62)

    # ---------------------------------------------------------
    # Sweep iteration budgets
    # ---------------------------------------------------------

    all_results = []

    for max_iter in args.iters:

        inv_mel = T.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sr,
            max_iter=max_iter,
            sgdargs={
                "lr": inv_mel_lr,
                "momentum": inv_mel_momentum,
            },
        ).to(device)

        griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            hop_length=hop,
            power=2.0,
            n_iter=gl_iters,
            momentum=gl_momentum,
        ).to(device)

        distances = []

        for wav, reference, name in zip(
            waveforms,
            references,
            names,
        ):

            # Waveform → tensor
            x = torch.tensor(
                wav,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            # Production reconstruction path
            mel = mel_transform(x)

            linear_spec = inv_mel(mel)

            # Safety clamp, matching your Hindi implementation
            linear_spec = linear_spec.clamp(min=0.0)

            reconstruction = griffin_lim(
                linear_spec
            )[0]

            reconstruction = (
                reconstruction
                .detach()
                .cpu()
                .numpy()
            )

            # Match original/reference length
            n = min(
                len(reconstruction),
                len(reference),
            )

            distance = logmel_l1_db(
                reconstruction[:n],
                reference[:n],
                sr,
                n_fft,
                hop,
                n_mels,
            )

            distances.append(distance)

            all_results.append(
                {
                    "clip": name,
                    "inv_mel_max_iter": max_iter,
                    "logmel_l1_db": f"{distance:.4f}",
                }
            )

        mean_d = float(np.mean(distances))
        min_d = float(np.min(distances))
        max_d = float(np.max(distances))

        print(
            f"{max_iter:>10} | "
            f"{mean_d:>20.3f} | "
            f"{min_d:>10.3f} | "
            f"{max_d:>10.3f}"
        )

        # Explicitly release iteration-specific transforms
        del inv_mel
        del griffin_lim

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    output_csv = (
        args.eval_dir / "english_convergence_check_result.csv"
    )

    with open(
        output_csv,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip",
                "inv_mel_max_iter",
                "logmel_l1_db",
            ],
        )

        writer.writeheader()
        writer.writerows(all_results)

    print()
    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()
