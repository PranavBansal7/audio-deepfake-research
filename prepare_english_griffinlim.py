#!/usr/bin/env python3
"""Prepare the CommonVoice English Griffin-Lim control for Phase 5.

This is the English counterpart of the finalized Hindi Phase-4 dataset. It
deliberately mirrors prepare_hindi_griffinlim.py (the script that produced the
finalized Hindi dataset, including the verified step7 amplitude fix), so that
the English and Hindi conditions differ only in language, not in procedure:

  * 16 kHz, n_fft=1024, hop=256, n_mels=128, power=2
  * InverseMelScale: max_iter=5000, lr=0.1, momentum=0.99   (== finalized Hindi)
  * Griffin-Lim: 32 iterations, momentum=0.99                (== finalized Hindi)
  * Batched mel -> InverseMelScale -> Griffin-Lim on zero-padded batches
    (per-sample independence; same scheme as the Hindi production run)
  * CommonVoice vote margin >=2, duration 2-8 s, max 5 clips/speaker
  * 1125 pairs, seed 42, oversample factor 1.05
  * RMS matching with the 0.99 dynamic-peak fallback         (== finalized Hindi)
  * PCM_16 WAV output (what torchaudio.save wrote for the Hindi dataset)
  * librosa-based MP3 probing/decoding with soundfile fallback
    (torchaudio 0.13.1's mp3 support is backend-dependent and was deliberately
    avoided in the Hindi pipeline for the same reason)

IMPORTANT -- verify parity before the production run:
    This script's synthesis/filtering parameters must match the values recorded
    in hindi_griffinlim_eval_final/config.json. If that file exists next to this
    script, main() compares the two automatically and warns on any mismatch.

The Hindi dataset is already finalized and is NOT regenerated here.

Output layout (mirrors the Hindi dataset layout):
    english_griffinlim_eval_final_v2/
        real_english/english_real_0000.wav ...
        fake_english_griffinlim/english_fake_0000.wav ...
        metadata.csv          <- per-pair fields + amplitude diagnostics +
                                 inferred_normalization (rms_matched|peak_normalized)
        protocol.txt          <- ASVspoof-LA-style, attack id "GL" (same as Hindi)
        config.json           <- exact config used for this run
        environment.json      <- package/hardware versions
        manifest.json         <- config hash; guards against mixed-config reruns
        selected_sources.csv  <- the clips that became pairs (provenance)
        prepare.log
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import logging
import platform
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm


@dataclasses.dataclass(frozen=True)
class Config:
    cv_dir: str = "cv_english_subset/en"  # CHANGE THIS
    out_dir: str = "english_griffinlim_eval_final"
    # Path to the finalized Hindi run's config.json, used for an automatic
    # parity check (warning only). Set to "" to skip.
    hindi_config_path: str = "hindi_griffinlim_eval_final/config.json"
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    power: float = 2.0
    gl_n_iter: int = 32
    gl_momentum: float = 0.99
    # Finalized Hindi production values (confirmed against the actual run).
    # Do NOT change these without regenerating and re-finalizing BOTH languages.
    inv_mel_max_iter: int = 5000
    inv_mel_lr: float = 0.1
    inv_mel_momentum: float = 0.99
    min_vote_margin: int = 2
    max_clips_per_speaker: int = 5
    min_duration_s: float = 2.0
    max_duration_s: float = 8.0
    target_pairs: int = 1125
    oversample_factor: float = 1.05  # == Hindi: pool size = target_pairs * oversample_factor
    batch_size: int = 256            # clips per GPU forward pass (A100-40GB safe)
    cross_check_n: int = 30          # == Hindi: samples cross-validated against librosa
    seed: int = 42
    peak_limit: float = 0.99
    wav_subtype: str = "PCM_16"     # == Hindi (torchaudio.save's WAV default)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()
OUT = Path(CFG.out_dir)


def setup_logging() -> logging.Logger:
    OUT.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("prepare_english_griffinlim")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); log.addHandler(sh)
    fh = logging.FileHandler(OUT / "prepare.log"); fh.setFormatter(fmt); log.addHandler(fh)
    return log


LOGGER = setup_logging()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def cfg_hash(cfg: Config) -> str:
    payload = json.dumps(dataclasses.asdict(cfg), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def save_environment_info() -> None:
    info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchaudio": getattr(torchaudio, "__version__", "unknown"),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "librosa": librosa.__version__,
        "soundfile": sf.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    (OUT / "environment.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def ensure_manifest() -> None:
    path = OUT / "manifest.json"
    payload = {
        "script": "prepare_english_griffinlim.py",
        "config": dataclasses.asdict(CFG),
        "config_sha256": cfg_hash(CFG),
        "torch": torch.__version__,
        "torchaudio": getattr(torchaudio, "__version__", "unknown"),
        "librosa": librosa.__version__,
    }
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("config_sha256") != payload["config_sha256"]:
            raise RuntimeError("Existing English output has a different configuration; use a fresh output directory.")
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def check_hindi_parity() -> None:
    """Warn if this script's synthesis/filter parameters differ from the ones
    recorded in the finalized Hindi run's config.json."""
    p = Path(CFG.hindi_config_path)
    if not p.exists():
        LOGGER.warning(
            "Hindi config.json not found at %s -- cannot auto-verify parity. "
            "Manually confirm these match the finalized Hindi production run: "
            "inv_mel_max_iter=%d, inv_mel_lr=%s, inv_mel_momentum=%s, gl_iters=%d, "
            "gl_momentum=%s, n_fft=%d, hop_length=%d, n_mels=%d, sample_rate=%d, "
            "min_duration_s=%s, max_duration_s=%s, min_vote_margin=%d, "
            "max_clips_per_speaker=%d, oversample_factor=%s, seed=%d.",
            p, CFG.inv_mel_max_iter, CFG.inv_mel_lr, CFG.inv_mel_momentum,
            CFG.gl_n_iter, CFG.gl_momentum, CFG.n_fft, CFG.hop_length, CFG.n_mels,
            CFG.sample_rate, CFG.min_duration_s, CFG.max_duration_s,
            CFG.min_vote_margin, CFG.max_clips_per_speaker, CFG.oversample_factor, CFG.seed)
        return
    hindi = json.loads(p.read_text(encoding="utf-8"))
    mapping = {  # english field -> hindi field in prepare_hindi_griffinlim.py's Config
        "sample_rate": "sample_rate", "n_fft": "n_fft", "hop_length": "hop_length",
        "n_mels": "n_mels", "gl_n_iter": "gl_iters", "gl_momentum": "gl_momentum",
        "inv_mel_max_iter": "inv_mel_max_iter", "inv_mel_lr": "inv_mel_lr",
        "inv_mel_momentum": "inv_mel_momentum", "min_duration_s": "min_duration_s",
        "max_duration_s": "max_duration_s", "min_vote_margin": "min_vote_margin",
        "max_clips_per_speaker": "max_clips_per_speaker",
        "oversample_factor": "oversample_factor", "seed": "seed",
    }
    mismatches = []
    for en_key, hi_key in mapping.items():
        en_val = getattr(CFG, en_key)
        hi_val = hindi.get(hi_key, "<missing>")
        if hi_val == "<missing>" or float(en_val) != float(hi_val):
            mismatches.append(f"{en_key}: english={en_val} vs hindi({hi_key})={hi_val}")
    if mismatches:
        LOGGER.warning(
            "PARITY MISMATCH with the finalized Hindi run -- the cross-language "
            "comparison is only controlled if these match:\n  %s",
            "\n  ".join(mismatches))
    else:
        LOGGER.info("Parity check passed: English config matches the finalized Hindi config.json on all synthesis/filter parameters.")


class GriffinLimSynthesizer:
    """Batched mel -> InverseMelScale -> Griffin-Lim chain, mirroring the
    finalized Hindi pipeline (prepare_hindi_griffinlim.py)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.mel = T.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, power=cfg.power).to(self.device)
        self.inv_mel = T.InverseMelScale(
            n_stft=cfg.n_fft // 2 + 1, n_mels=cfg.n_mels, sample_rate=cfg.sample_rate,
            max_iter=cfg.inv_mel_max_iter,
            sgdargs={"lr": cfg.inv_mel_lr, "momentum": cfg.inv_mel_momentum}).to(self.device)
        self.gl = T.GriffinLim(
            n_fft=cfg.n_fft, hop_length=cfg.hop_length, power=cfg.power,
            n_iter=cfg.gl_n_iter, momentum=cfg.gl_momentum).to(self.device)

    def synthesize_batch(self, wavs: List[np.ndarray]) -> List[np.ndarray]:
        """One padded batch through mel/InverseMelScale/Griffin-Lim.

        All three ops are computed independently per batch element, so the
        zero-padding of one clip cannot influence another clip's reconstruction;
        each reconstruction is trimmed/padded back to its own true length
        afterwards. This is exactly the batching scheme used for the finalized
        Hindi dataset.
        """
        lengths = [len(w) for w in wavs]
        max_len = max(lengths)
        batch = torch.zeros(len(wavs), max_len, device=self.device)
        for i, w in enumerate(wavs):
            batch[i, :len(w)] = torch.from_numpy(w).to(self.device)
        mel_spec = self.mel(batch)                       # (B, n_mels, frames)
        # InverseMelScale's internal SGD needs gradients; Griffin-Lim does not.
        linear_spec = self.inv_mel(mel_spec).clamp(min=0.0)
        with torch.no_grad():
            recon = self.gl(linear_spec)                 # (B, T')
        out = []
        for i, L in enumerate(lengths):
            r = recon[i]
            if r.shape[-1] > L: r = r[:L]
            elif r.shape[-1] < L: r = torch.nn.functional.pad(r, (0, L - r.shape[-1]))
            out.append(r)
        return out

    def match_amplitude(self, recon: torch.Tensor, ref: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """RMS-match the fake to its own paired real clip; if that match would
        exceed the 0.99 ceiling, smoothly peak-normalize instead (the verified
        step7 dynamic fallback, identical to the finalized Hindi rule)."""
        eps = 1e-8
        ref_rms = torch.sqrt(torch.mean(ref ** 2) + eps)
        raw_rms = torch.sqrt(torch.mean(recon ** 2) + eps)
        raw_peak = torch.max(torch.abs(recon)).clamp(min=eps)
        requested = ref_rms / raw_rms
        predicted_peak = raw_peak * requested
        fallback = bool(predicted_peak.item() > self.cfg.peak_limit)
        if fallback:
            final_scale = torch.tensor(self.cfg.peak_limit, device=recon.device, dtype=recon.dtype) / raw_peak
        else:
            final_scale = requested
        out = recon * final_scale
        final_rms = torch.sqrt(torch.mean(out ** 2) + eps)
        final_peak = torch.max(torch.abs(out))
        diag = {
            "source_rms": float(ref_rms.item()),
            "raw_recon_rms": float(raw_rms.item()),
            "raw_recon_peak": float(raw_peak.item()),
            "requested_rms_scale": float(requested.item()),
            "predicted_peak_after_rms": float(predicted_peak.item()),
            "peak_fallback_used": int(fallback),
            "final_scale": float(final_scale.item()),
            "final_rms": float(final_rms.item()),
            "final_peak": float(final_peak.item()),
            "rms_relative_error": float(abs(final_rms.item() - ref_rms.item()) / max(ref_rms.item(), eps)),
            "clipping_fraction": float((torch.abs(out) >= 1.0 - 1e-7).float().mean().item()),
        }
        return out, diag


# --------------------------------------------------------------------------- #
# Robust audio I/O (mirrors the Hindi pipeline: torchaudio's mp3 support on
# 0.13.1 depends on the backend it was built against; librosa.load falls back
# through soundfile/audioread/ffmpeg automatically)
# --------------------------------------------------------------------------- #


def probe_duration(path: str) -> Optional[float]:
    try:
        info = torchaudio.info(path)
        if info.num_frames > 0:
            return info.num_frames / info.sample_rate
    except Exception:
        pass
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


def load_audio(path: str) -> np.ndarray:
    wav, _ = librosa.load(path, sr=CFG.sample_rate, mono=True)
    audio = np.asarray(wav, dtype=np.float32)
    if not np.isfinite(audio).all(): raise RuntimeError(f"Non-finite source: {path}")
    return audio


# --------------------------------------------------------------------------- #
# Source selection (mirrors the Hindi pipeline: shuffle once with the seed,
# then apply the vote-margin filter, duration window and per-speaker cap while
# scanning, counting only clips that pass every filter; stop at the pool size)
# --------------------------------------------------------------------------- #


def select_sources() -> pd.DataFrame:
    base = Path(CFG.cv_dir); tsv = base / "validated.tsv"; clips = base / "clips"
    if not tsv.exists(): raise FileNotFoundError(f"Missing validated.tsv: {tsv}")
    if not clips.exists(): raise FileNotFoundError(f"Missing clips/: {clips}")
    df = pd.read_csv(tsv, sep="\t", low_memory=False)
    required = {"path", "client_id", "up_votes", "down_votes"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"validated.tsv missing: {sorted(missing)}")
    df = df.sample(frac=1.0, random_state=CFG.seed).reset_index(drop=True)

    pool_size = int(np.ceil(CFG.target_pairs * CFG.oversample_factor))
    speaker_counts: Dict[str, int] = {}
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Filtering CommonVoice"):
        if len(rows) >= pool_size: break
        if (r["up_votes"] - r["down_votes"]) < CFG.min_vote_margin: continue
        spk = str(r["client_id"])
        if speaker_counts.get(spk, 0) >= CFG.max_clips_per_speaker: continue
        p = clips / str(r["path"])
        if not p.exists(): continue
        dur = probe_duration(str(p))
        if dur is None: continue
        if not (CFG.min_duration_s <= dur <= CFG.max_duration_s): continue
        rows.append({"client_id": spk, "path": str(p), "duration_s": float(dur)})
        speaker_counts[spk] = speaker_counts.get(spk, 0) + 1

    pool = pd.DataFrame(rows)
    LOGGER.info("Candidate pool: %d clips from %d speakers (pool target %d)", len(pool), len(speaker_counts), pool_size)
    if len(pool) < CFG.target_pairs:
        raise RuntimeError(
            f"Candidate pool ({len(pool)}) is smaller than target_pairs ({CFG.target_pairs}). "
            f"Loosen the duration/vote filters or raise oversample_factor.")
    pool.to_csv(OUT / "selected_sources.csv", index=False)
    return pool.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# QC helpers
# --------------------------------------------------------------------------- #


def logmel_l1(a: np.ndarray, b: np.ndarray) -> float:
    ma = librosa.feature.melspectrogram(y=a, sr=CFG.sample_rate, n_fft=CFG.n_fft, hop_length=CFG.hop_length, n_mels=CFG.n_mels, power=2.0)
    mb = librosa.feature.melspectrogram(y=b, sr=CFG.sample_rate, n_fft=CFG.n_fft, hop_length=CFG.hop_length, n_mels=CFG.n_mels, power=2.0)
    da = librosa.power_to_db(ma, ref=np.max); db = librosa.power_to_db(mb, ref=np.max)
    m = min(da.shape[1], db.shape[1])
    return float(np.mean(np.abs(da[:, :m] - db[:, :m])))


def librosa_reference_recon(wav: np.ndarray) -> np.ndarray:
    """Independent librosa reconstruction for cross-validation, identical to
    the Hindi pipeline's cross_check_against_librosa()."""
    mel = librosa.feature.melspectrogram(
        y=wav, sr=CFG.sample_rate, n_fft=CFG.n_fft, hop_length=CFG.hop_length,
        n_mels=CFG.n_mels, power=2.0)
    return librosa.feature.inverse.mel_to_audio(
        mel, sr=CFG.sample_rate, n_fft=CFG.n_fft, hop_length=CFG.hop_length,
        n_iter=CFG.gl_n_iter, power=2.0)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

META_COLS = ["pair_id", "client_id", "real_path", "fake_path", "duration_s",
             "inferred_normalization", "logmel_l1_real_fake_db",
             "source_rms", "raw_recon_rms", "raw_recon_peak", "requested_rms_scale",
             "predicted_peak_after_rms", "peak_fallback_used", "final_scale",
             "final_rms", "final_peak", "rms_relative_error", "clipping_fraction"]


def generate(pool: pd.DataFrame) -> None:
    real_dir = OUT / "real_english"; fake_dir = OUT / "fake_english_griffinlim"
    real_dir.mkdir(parents=True, exist_ok=True); fake_dir.mkdir(parents=True, exist_ok=True)
    meta = OUT / "metadata.csv"; protocol = OUT / "protocol.txt"
    if not meta.exists(): pd.DataFrame(columns=META_COLS).to_csv(meta, index=False)
    completed = set(pd.read_csv(meta)["pair_id"].astype(str))
    if completed:
        LOGGER.info("Resuming: %d pairs already present in metadata.csv", len(completed))

    synth = GriffinLimSynthesizer(CFG)
    records = pool.to_dict("records")
    start_time = time.time()
    produced_this_run = 0
    n_cross_checked = 0
    ref_distances: List[float] = []

    batch_wavs: List[np.ndarray] = []
    batch_rows: List[dict] = []

    def flush_batch():
        nonlocal produced_this_run, n_cross_checked
        if not batch_wavs:
            return
        wavs, rows_in = batch_wavs, batch_rows
        try:
            recons = synth.synthesize_batch(wavs)
        except Exception as e:
            # A whole-batch failure should not sink every clip in it: retry
            # sample-by-sample and skip only the clips that genuinely fail.
            LOGGER.warning("Batch synthesis failed (%s); retrying sample-by-sample.", e)
            recons, wavs_keep, rows_keep = [], [], []
            for w, r in zip(wavs, rows_in):
                try:
                    recons.extend(synth.synthesize_batch([w]))
                    wavs_keep.append(w); rows_keep.append(r)
                except Exception as e2:
                    LOGGER.warning("Skipping clip %s: %s", r["path"], e2)
            wavs, rows_in = wavs_keep, rows_keep
        with meta.open("a", newline="", encoding="utf-8") as fm, protocol.open("a", encoding="utf-8") as fp:
            writer = csv.writer(fm)
            for wav, recon, row in zip(wavs, recons, rows_in):
                pid = row["pair_id"]  # deterministic: pool position, stable across resumes
                real = np.clip(wav, -1.0, 1.0).astype(np.float32)  # same clamp as the Hindi pipeline
                fake_t, diag = synth.match_amplitude(recon, row["wav_tensor"])
                fake = fake_t.detach().cpu().numpy().astype(np.float32)
                if not np.isfinite(fake).all():
                    LOGGER.warning("Non-finite fake for %s; skipping.", row["path"]); continue
                method = "peak_normalized" if diag["peak_fallback_used"] else "rms_matched"
                rpath = real_dir / f"english_real_{pid}.wav"; fpath = fake_dir / f"english_fake_{pid}.wav"
                sf.write(rpath, real, CFG.sample_rate, subtype=CFG.wav_subtype)
                sf.write(fpath, fake, CFG.sample_rate, subtype=CFG.wav_subtype)
                dist = logmel_l1(real, fake)
                if n_cross_checked < CFG.cross_check_n:
                    ref_distances.append(logmel_l1(fake, librosa_reference_recon(real)))
                    n_cross_checked += 1
                writer.writerow([pid, row["client_id"], str(rpath), str(fpath), row["duration_s"],
                                 method, dist, diag["source_rms"], diag["raw_recon_rms"],
                                 diag["raw_recon_peak"], diag["requested_rms_scale"],
                                 diag["predicted_peak_after_rms"], diag["peak_fallback_used"],
                                 diag["final_scale"], diag["final_rms"], diag["final_peak"],
                                 diag["rms_relative_error"], diag["clipping_fraction"]])
                fp.write(f"{row['client_id']} english_real_{pid} - - bonafide\n")
                fp.write(f"{row['client_id']} english_fake_{pid} - GL spoof\n")
                produced_this_run += 1
        batch_wavs.clear(); batch_rows.clear()

    n_done = len(completed)
    for pos, row in enumerate(tqdm(records, desc="Synthesizing English")):
        if n_done >= CFG.target_pairs:
            break
        pid = f"{pos:04d}"
        if pid in completed:
            continue
        try:
            wav = load_audio(row["path"])
            if len(wav) < 100:
                raise ValueError("clip too short after load")
        except Exception as e:
            LOGGER.warning("Skipping candidate (%s): %s", row["path"], e)
            continue
        row = dict(row); row["pair_id"] = pid
        row["wav_tensor"] = torch.from_numpy(wav).to(synth.device)
        batch_wavs.append(wav); batch_rows.append(row)
        n_done += 1
        if len(batch_wavs) >= CFG.batch_size:
            flush_batch()
    flush_batch()

    final = pd.read_csv(meta)
    if len(final) != CFG.target_pairs:
        raise RuntimeError(
            f"Expected {CFG.target_pairs} pairs, found {len(final)}. "
            f"Re-run to consume more of the pool (resume is supported), or raise oversample_factor.")
    LOGGER.info("Generated %d pairs, %d speakers, mean duration %.3fs",
                len(final), final.client_id.nunique(), final.duration_s.mean())
    LOGGER.info("Amplitude normalization: %d RMS-matched, %d peak-limited fallback",
                int((final.inferred_normalization == "rms_matched").sum()),
                int((final.inferred_normalization == "peak_normalized").sum()))
    if ref_distances:
        LOGGER.info("Independent librosa cross-check: n=%d, mean fake-vs-reference log-mel L1 = %.3f dB",
                    len(ref_distances), float(np.mean(ref_distances)))
    LOGGER.info("Wall-clock %.1fs (%d pairs this run)", time.time() - start_time, produced_this_run)


def main() -> None:
    set_seed(CFG.seed); ensure_manifest(); save_environment_info()
    if CFG.cv_dir.startswith("/path/to/"): raise RuntimeError("Edit Config.cv_dir before running.")
    (OUT / "config.json").write_text(json.dumps(dataclasses.asdict(CFG), indent=2), encoding="utf-8")
    LOGGER.info("Config: %s", json.dumps(dataclasses.asdict(CFG), indent=2))
    check_hindi_parity()
    pool = select_sources(); generate(pool)
    LOGGER.info("English control dataset ready: %s", OUT)


if __name__ == "__main__":
    main()
