#!/usr/bin/env python3
"""
Phase 4: Hindi Data Preparation via Griffin-Lim Copy-Synthesis
================================================================

Builds a paired (real, Griffin-Lim-resynthesized-fake) Hindi evaluation set from
Mozilla Common Voice, in the same family of method as ASVspoof 2019 LA attack A11.

What's verified vs. what's a reasonable choice
------------------------------------------------
Per the ASVspoof 2019 database paper (Wang et al., arXiv:1911.01601), A11 shares
its Tacotron-2-style acoustic model with A10, differing only in using the
Griffin-Lim algorithm instead of a neural vocoder to generate the waveform.
Tacotron 2 canonically predicts a *mel* spectrogram, so a mel-based Griffin-Lim
pipeline is the right family of method -- unlike a naive first draft that feeds a
mel spectrogram straight into `librosa.griffinlim()`, which expects a linear
magnitude/power spectrogram and silently misinterprets 128 mel channels as 128
linear-frequency bins.

What is NOT independently verified: the exact mel-to-linear step A11's own
implementation used before Griffin-Lim. That detail isn't in the public
description. This script uses `InverseMelScale`'s SGD-based pseudo-inverse for
that step, which is a standard, defensible choice (and the correct fix for the
naive snippet's dimensionality bug) -- but treat this as "a faithful mel-based
Griffin-Lim copy-synthesis consistent with A11's documented approach," not as a
verified bit-exact reproduction of A11's internal pipeline.

Method:
    real waveform
      -> mel spectrogram                          (torchaudio.transforms.MelSpectrogram)
      -> pseudo-inverse to linear magnitude spec   (torchaudio.transforms.InverseMelScale)
      -> phase reconstruction                      (torchaudio.transforms.GriffinLim)
      -> "fake" waveform

Processing is batched (see --batch_size) so the mel/InverseMelScale/GriffinLim
chain actually exploits GPU parallelism instead of running sample-by-sample;
actual measured throughput is logged at the end of each run rather than assumed.

Usage
-----
    python prepare_hindi_griffinlim.py \
        --commonvoice_dir /path/to/cv-corpus-XX.0-YYYY-MM-DD/hi \
        --output_dir ./hindi_griffinlim_eval \
        --n_pairs 200 \
        --device cuda

Expected CommonVoice directory layout (standard CV release):
    <commonvoice_dir>/
        validated.tsv
        clips/
            common_voice_hi_XXXXXXXX.mp3
            ...

Output layout:
    <output_dir>/
        real_hindi/hindi_real_0000.wav ...
        fake_hindi_griffinlim/hindi_fake_0000.wav ...
        protocol.txt        <- ASVspoof-LA-style protocol (space-delimited, no header)
        metadata.csv         <- rich metadata for analysis / error slicing
        config.json           <- exact config used for this run
        environment.json       <- package/hardware versions for reproducibility
        prepare.log              <- full run log
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

try:
    import soundfile as sf
except ImportError:
    sf = None

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    commonvoice_dir: Path
    output_dir: Path
    n_pairs: int = 200
    oversample_factor: float = 3.0  # pool size = n_pairs * oversample_factor
    sample_rate: int = 16000        # matches ASVspoof 2019 LA convention
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    gl_iters: int = 32              # matches Phase 4 plan's griffinlim n_iter
    gl_momentum: float = 0.99
    inv_mel_max_iter: int = 200     # SGD steps for mel->linear pseudo-inverse
    inv_mel_lr: float = 0.1
    batch_size: int = 16            # samples processed together per GPU forward pass
    min_duration_s: float = 2.0
    max_duration_s: float = 8.0
    min_vote_margin: int = 1        # up_votes - down_votes, quality filter
    max_clips_per_speaker: int = 5  # speaker-diversity cap
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cross_check_n: int = 5          # how many samples to cross-validate against librosa
    dry_run: bool = False


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    fh = logging.FileHandler(output_dir / "prepare.log", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def save_environment_info(output_dir: Path) -> None:
    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    with open(output_dir / "environment.json", "w") as f:
        json.dump(info, f, indent=2)


# --------------------------------------------------------------------------- #
# Robust audio I/O
#
# Common Voice clips are .mp3. torchaudio's mp3 support depends on the backend it
# was built against (sox/ffmpeg), which is a common silent-failure point on older
# torchaudio builds (e.g. 0.13.1, paired with torch 1.13.1). librosa.load falls
# back through soundfile/audioread/ffmpeg automatically, so it's used here for
# both duration probing and decoding instead of relying on torchaudio for I/O.
# torchaudio is still used for the actual mel/InverseMelScale/GriffinLim math,
# which is where the GPU benefit is.
# --------------------------------------------------------------------------- #


def probe_duration(path: str) -> Optional[float]:
    """Fast, header-only duration probe with fallbacks for mp3 edge cases."""
    try:
        info = torchaudio.info(path)
        if info.num_frames > 0:
            return info.num_frames / info.sample_rate
    except Exception:
        pass
    if sf is not None:
        try:
            info = sf.info(path)
            if info.frames > 0:
                return info.frames / info.samplerate
        except Exception:
            pass
    try:
        return float(librosa.get_duration(path=path))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# CommonVoice metadata handling
# --------------------------------------------------------------------------- #


def load_commonvoice_metadata(cv_dir: Path) -> pd.DataFrame:
    tsv_path = cv_dir / "validated.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(
            f"Could not find {tsv_path}. Expected a standard Common Voice release "
            f"directory containing validated.tsv and a clips/ subfolder. Common Voice "
            f"requires accepting Mozilla's terms before download, so this script does "
            f"not fetch it automatically -- see README.md."
        )
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    required = {"client_id", "path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"validated.tsv is missing expected columns: {missing}")
    return df


def build_candidate_pool(
    df: pd.DataFrame, cfg: Config, clips_dir: Path, pool_size: int, logger: logging.Logger
) -> pd.DataFrame:
    """
    Shuffle + scan validated.tsv, stopping early once `pool_size` clips pass all
    filters. Avoids probing every file in large releases when filters are already
    satisfied by a subset. Applies: vote-margin quality filter, duration window,
    per-speaker cap (for speaker diversity in a small target set), file existence.
    """
    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

    if "up_votes" in df.columns and "down_votes" in df.columns:
        df["vote_margin"] = df["up_votes"] - df["down_votes"]
    else:
        df["vote_margin"] = None
        logger.warning("validated.tsv has no up_votes/down_votes columns; skipping vote filter.")

    speaker_counts: dict[str, int] = {}
    rows = []
    scanned = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scanning CommonVoice metadata"):
        scanned += 1
        if len(rows) >= pool_size:
            break

        if row["vote_margin"] is not None and not pd.isna(row["vote_margin"]):
            if row["vote_margin"] < cfg.min_vote_margin:
                continue

        spk = row["client_id"]
        if speaker_counts.get(spk, 0) >= cfg.max_clips_per_speaker:
            continue

        fp = clips_dir / row["path"]
        if not fp.exists():
            continue

        dur = probe_duration(str(fp))
        if dur is None:
            continue
        if not (cfg.min_duration_s <= dur <= cfg.max_duration_s):
            continue

        row = row.copy()
        row["duration_s"] = dur
        row["src_path"] = str(fp)
        rows.append(row)
        speaker_counts[spk] = speaker_counts.get(spk, 0) + 1

    pool = pd.DataFrame(rows)
    logger.info(
        f"Scanned {scanned}/{len(df)} rows, collected pool of {len(pool)} candidates "
        f"from {len(speaker_counts)} unique speakers (target pool size: {pool_size})."
    )
    if len(pool) < cfg.n_pairs:
        logger.warning(
            f"Candidate pool ({len(pool)}) is smaller than requested n_pairs "
            f"({cfg.n_pairs}). Loosen --min_duration_s/--max_duration_s/--min_vote_margin/"
            f"--max_clips_per_speaker, or fall back to L2-ARCTIC per the risk-mitigation plan."
        )
    return pool


# --------------------------------------------------------------------------- #
# Audio processing
# --------------------------------------------------------------------------- #


class GriffinLimSynthesizer:
    """Wraps the mel -> inverse-mel -> Griffin-Lim chain as reusable GPU modules."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            power=2.0,
        ).to(self.device)

        self.inv_mel = T.InverseMelScale(
            n_stft=cfg.n_fft // 2 + 1,
            n_mels=cfg.n_mels,
            sample_rate=cfg.sample_rate,
            max_iter=cfg.inv_mel_max_iter,
            sgdargs={"lr": cfg.inv_mel_lr, "momentum": 0.9},
        ).to(self.device)

        self.gl_transform = T.GriffinLim(
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            power=2.0,
            n_iter=cfg.gl_iters,
            momentum=cfg.gl_momentum,
        ).to(self.device)

    def load_audio(self, path: str) -> torch.Tensor:
        # librosa.load handles mp3 decoding + mono-downmix + resampling in one call
        # via soundfile/audioread/ffmpeg fallbacks, avoiding torchaudio mp3-backend
        # issues on older builds (see the module-level note above probe_duration).
        # Returns a 1D (T,) tensor -- batch dim is added in synthesize_batch.
        wav_np, _ = librosa.load(path, sr=self.cfg.sample_rate, mono=True)
        return torch.from_numpy(wav_np).float().to(self.device)

    def synthesize_batch(self, wavs: list) -> list:
        """
        wavs: list of 1D (T_i,) tensors on self.device, possibly different lengths.
        Returns a list of 1D reconstructions, each trimmed/padded back to its own
        input's length. Processing the whole list as one padded batch (rather than
        looping sample-by-sample) is what actually gives the GPU something to
        parallelize -- MelSpectrogram/InverseMelScale/GriffinLim all treat leading
        dims as batch dims, so this is a straightforward zero-pad-to-max-length batch.
        Padding is silence appended past each clip's real content; since these ops
        are computed independently per batch element (no cross-sample mixing), the
        padding of one clip cannot influence another clip's reconstruction, and the
        padded tail is trimmed off before saving.
        """
        lengths = [w.shape[-1] for w in wavs]
        max_len = max(lengths)
        batch = torch.zeros(len(wavs), max_len, device=self.device)
        for i, w in enumerate(wavs):
            batch[i, : w.shape[-1]] = w

        mel_spec = self.mel_transform(batch)                  # (B, n_mels, frames)
        linear_spec = self.inv_mel(mel_spec).clamp(min=0.0)     # SGD can yield tiny negatives
        recon = self.gl_transform(linear_spec)                    # (B, T')

        return [self.match_length(recon[i], L) for i, L in enumerate(lengths)]

    @staticmethod
    def match_length(recon: torch.Tensor, target_len: int) -> torch.Tensor:
        cur_len = recon.shape[-1]
        if cur_len == target_len:
            return recon
        if cur_len > target_len:
            return recon[..., :target_len]
        pad = target_len - cur_len
        return torch.nn.functional.pad(recon, (0, pad))

    @staticmethod
    def match_amplitude(recon: torch.Tensor, reference: torch.Tensor, safety_peak: float = 0.99) -> torch.Tensor:
        """
        Scale the Griffin-Lim reconstruction to match the RMS of its *own paired*
        real clip -- not a fixed global target. Griffin-Lim's output level is
        otherwise fairly arbitrary, so some normalization of the fake is
        legitimate; but peak-normalizing every fake to the same fixed level (the
        previous version of this function) would collapse natural loudness
        variation across the fake set into one flat level regardless of each
        source clip's loudness -- handing any downstream classifier or attribution
        method a trivial amplitude cue that has nothing to do with language or
        vocoder, which is exactly the kind of shortcut this project exists to
        catch. Matching pairwise instead preserves the real clip's loudness
        relationship in its paired fake. The real clip itself is never rescaled.
        A safety clamp only engages if the matched result would clip.
        """
        ref_rms = reference.pow(2).mean().sqrt().clamp(min=1e-8)
        recon_rms = recon.pow(2).mean().sqrt().clamp(min=1e-8)
        matched = recon * (ref_rms / recon_rms)
        peak = matched.abs().max()
        if peak > safety_peak:
            matched = matched * (safety_peak / peak)
        return matched


def cross_check_against_librosa(
    synth: GriffinLimSynthesizer, wav: torch.Tensor, cfg: Config, logger: logging.Logger, idx: int
) -> None:
    """
    Sanity check: reconstruct the same clip with librosa's independent implementation
    (pseudo-inverse mel->linear + Griffin-Lim) and compare log-mel distance to the
    GPU/torchaudio reconstruction. Large divergence would indicate a bug in the
    torchaudio pipeline rather than an expected implementation-difference.
    """
    try:
        import librosa
    except ImportError:
        logger.warning("librosa not installed; skipping cross-validation check.")
        return

    wav_np = wav.detach().cpu().numpy()
    mel = librosa.feature.melspectrogram(
        y=wav_np, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_mels=cfg.n_mels, power=2.0,
    )
    librosa_recon = librosa.feature.inverse.mel_to_audio(
        mel, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_iter=cfg.gl_iters, power=2.0,
    )

    torch_recon = synth.synthesize_batch([wav])[0]
    torch_recon_np = torch_recon.detach().cpu().numpy()

    n = min(len(librosa_recon), len(torch_recon_np))
    lm_a = librosa.power_to_db(
        librosa.feature.melspectrogram(y=librosa_recon[:n], sr=cfg.sample_rate,
                                        n_fft=cfg.n_fft, hop_length=cfg.hop_length, n_mels=cfg.n_mels)
    )
    lm_b = librosa.power_to_db(
        librosa.feature.melspectrogram(y=torch_recon_np[:n], sr=cfg.sample_rate,
                                        n_fft=cfg.n_fft, hop_length=cfg.hop_length, n_mels=cfg.n_mels)
    )
    min_frames = min(lm_a.shape[1], lm_b.shape[1])
    l1_dist = np.abs(lm_a[:, :min_frames] - lm_b[:, :min_frames]).mean()
    logger.info(
        f"[cross-check {idx}] torchaudio-GPU vs librosa-CPU reconstruction, "
        f"mean log-mel L1 distance = {l1_dist:.3f} dB (expect small; large values -> investigate)"
    )


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #


def run(cfg: Config) -> None:
    output_dir = Path(cfg.output_dir)
    real_dir = output_dir / "real_hindi"
    fake_dir = output_dir / "fake_hindi_griffinlim"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Config: {json.dumps(dataclasses.asdict(cfg), default=str, indent=2)}")

    set_seed(cfg.seed)
    save_environment_info(output_dir)
    with open(output_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, default=str, indent=2)

    cv_dir = Path(cfg.commonvoice_dir)
    clips_dir = cv_dir / "clips"
    if not clips_dir.exists():
        raise FileNotFoundError(f"Expected clips/ directory at {clips_dir}")

    df = load_commonvoice_metadata(cv_dir)
    logger.info(f"Loaded validated.tsv with {len(df)} rows.")

    n_pairs = 5 if cfg.dry_run else cfg.n_pairs
    pool_size = int(n_pairs * cfg.oversample_factor)
    pool = build_candidate_pool(df, cfg, clips_dir, pool_size, logger)

    if len(pool) == 0:
        raise RuntimeError("No candidates survived filtering -- check filter thresholds.")

    n_take = min(n_pairs, len(pool))
    selected = pool.sample(n=min(len(pool), n_take * 2), random_state=cfg.seed).reset_index(drop=True)

    synth = GriffinLimSynthesizer(cfg)
    device = torch.device(cfg.device)
    logger.info(f"Running on device: {device}")

    records = []
    n_cross_checked = 0
    pair_idx = 0
    batch_wavs: list = []
    batch_rows: list = []

    def flush_batch():
        nonlocal pair_idx
        if not batch_wavs:
            return
        recons = synth.synthesize_batch(batch_wavs)
        for wav, recon, row in zip(batch_wavs, recons, batch_rows):
            try:
                # Real audio keeps its natural loudness (only a safety clamp for
                # true [-1, 1] range violations, not a rescale). The fake is
                # RMS-matched to its own paired real clip -- see match_amplitude's
                # docstring for why this is deliberate rather than a fixed target.
                real_out = wav.clamp(-1.0, 1.0).cpu()
                fake_out = GriffinLimSynthesizer.match_amplitude(recon, wav).cpu()

                if not (torch.isfinite(real_out).all() and torch.isfinite(fake_out).all()):
                    raise ValueError("non-finite samples in output audio")

                real_name = f"hindi_real_{pair_idx:04d}.wav"
                fake_name = f"hindi_fake_{pair_idx:04d}.wav"
                torchaudio.save(str(real_dir / real_name), real_out.unsqueeze(0), cfg.sample_rate)
                torchaudio.save(str(fake_dir / fake_name), fake_out.unsqueeze(0), cfg.sample_rate)

                records.append({
                    "pair_id": pair_idx,
                    "client_id": row["client_id"],
                    "real_path": str(real_dir / real_name),
                    "fake_path": str(fake_dir / fake_name),
                    "duration_s": row["duration_s"],
                    "sentence": row.get("sentence", ""),
                    "gender": row.get("gender", ""),
                    "age": row.get("age", ""),
                    "accents": row.get("accents", ""),
                    "vote_margin": row.get("vote_margin", None),
                    "source_clip": row["path"],
                })
                pair_idx += 1
            except Exception as e:
                logger.warning(f"Skipping candidate ({row.get('path', '?')}): {e}")
        batch_wavs.clear()
        batch_rows.clear()

    t_start = time.time()
    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Loading + synthesizing"):
        if pair_idx + len(batch_wavs) >= n_take:
            break
        try:
            wav = synth.load_audio(row["src_path"])
            if wav.shape[-1] < 100:
                raise ValueError("clip too short after load")

            if n_cross_checked < cfg.cross_check_n:
                cross_check_against_librosa(synth, wav, cfg, logger, n_cross_checked)
                n_cross_checked += 1

            batch_wavs.append(wav)
            batch_rows.append(row)
        except Exception as e:
            logger.warning(f"Skipping candidate ({row.get('path', '?')}): {e}")
            continue

        if len(batch_wavs) >= cfg.batch_size:
            flush_batch()

    flush_batch()  # remaining partial batch
    elapsed = time.time() - t_start
    throughput = pair_idx / elapsed if elapsed > 0 else float("nan")
    logger.info(
        f"Synthesis wall-clock: {elapsed:.1f}s for {pair_idx} pairs "
        f"({throughput:.1f} pairs/sec, batch_size={cfg.batch_size}, device={cfg.device})."
    )

    if pair_idx < n_take:
        logger.warning(
            f"Only produced {pair_idx}/{n_take} pairs (pool exhausted or too many "
            f"failures). Re-run with a larger --oversample_factor or looser filters."
        )

    metadata = pd.DataFrame(records)
    metadata.to_csv(output_dir / "metadata.csv", index=False)

    # ASVspoof-LA-style protocol file: SPEAKER_ID UTT_ID - ATTACK_ID LABEL
    protocol_lines = []
    for r in records:
        real_utt = Path(r["real_path"]).stem
        fake_utt = Path(r["fake_path"]).stem
        protocol_lines.append(f"{r['client_id']} {real_utt} - - bonafide")
        protocol_lines.append(f"{r['client_id']} {fake_utt} - GL spoof")
    (output_dir / "protocol.txt").write_text("\n".join(protocol_lines) + "\n")

    # Integrity summary
    logger.info("=" * 70)
    logger.info(f"DONE. Produced {len(records)} real/fake pairs ({2 * len(records)} files).")
    if len(records) > 0:
        logger.info(f"Mean duration: {metadata['duration_s'].mean():.2f}s "
                    f"(min {metadata['duration_s'].min():.2f}s, max {metadata['duration_s'].max():.2f}s)")
        logger.info(f"Unique speakers: {metadata['client_id'].nunique()}")
        if len(records) <= 200:
            logger.info(
                f"NOTE: n={len(records)} pairs is a reasonable initial diagnostic pass "
                f"but is underpowered for formal significance testing on Phase 5's "
                f"explanation-stability metrics (cosine similarity, top-K overlap, "
                f"Spearman correlation) -- report those with bootstrap confidence "
                f"intervals, or scale --n_pairs up (cheap on an A100; see README)."
            )
    logger.info(f"Outputs: {output_dir}")
    logger.info("Next: run sanity_check.py to visually/statistically spot-check pairs "
                "before moving to Phase 5.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--commonvoice_dir", type=Path, required=True,
                    help="Path to CV language dir containing validated.tsv and clips/")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--n_pairs", type=int, default=200)
    p.add_argument("--oversample_factor", type=float, default=3.0)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--gl_iters", type=int, default=32)
    p.add_argument("--inv_mel_max_iter", type=int, default=200,
                    help="SGD steps for mel->linear pseudo-inverse. Raise (e.g. 1000) "
                         "for higher-fidelity reconstructions -- cheap on an A100.")
    p.add_argument("--batch_size", type=int, default=16,
                    help="Clips processed together per GPU forward pass through "
                         "mel/InverseMelScale/GriffinLim. Raise on an A100 (e.g. 32-64) "
                         "if you scale up --n_pairs.")
    p.add_argument("--min_duration_s", type=float, default=2.0)
    p.add_argument("--max_duration_s", type=float, default=8.0)
    p.add_argument("--min_vote_margin", type=int, default=1)
    p.add_argument("--max_clips_per_speaker", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cross_check_n", type=int, default=5,
                    help="Number of samples to cross-validate against librosa's independent implementation.")
    p.add_argument("--dry_run", action="store_true",
                    help="Process only 5 pairs to sanity-check the pipeline before a full run.")
    args = p.parse_args()
    return Config(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
