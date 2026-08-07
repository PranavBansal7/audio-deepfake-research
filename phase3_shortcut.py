# %% [markdown]
# # Phase 3 — Shortcut Analysis (Replication)
#
# Silence-shortcut analysis for the wav2vec2-XLS-R + AASIST detector on ASVspoof2019 LA
# eval, following Müller, Dieckmann, Czempin, Canals, Böttinger & Williams, "Speech is
# Silver, Silence is Golden: What do ASVspoof-trained Models Really Learn?"
# (ASVspoof 2021 Workshop, arXiv:2106.12914).
#
# Three experiments, matching the Phase 3 plan:
# 1. **Silence removal**: EER before/after trimming leading/trailing silence, on the
#    full 2019 LA eval protocol.
# 2. **Silence duration distributions**: leading/trailing silence by class, plus a
#    silence-only shortcut probe (does silence duration alone predict the label?),
#    which is the specific claim in Müller et al. worth reproducing quantitatively.
# 3. **Attribution overlap with silence**: do the Phase 2 occlusion / Integrated
#    Gradients attribution maps concentrate on silent regions of the (padded) waveform?
#
# **Environment:** NVIDIA DGX A100 (40GB), Python 3.8.20, torch 1.13.1
#
# **Before running:** this notebook reuses `model.py`, `data_utils_SSL.py`, and
# `eval_metric_LA.py` from the same `SSL_Anti-spoofing` checkout used in Phases 1–2,
# and assumes the same directory layout (`database/LA/...`, `pretrained_models/...`),
# checkpoint (`best_SSL_model_LA.pth`), and the Phase 2 output directory
# (`phase2_outputs/`) for the attribution-overlap experiment.
#
# **Source verification (Aug 2026):** verified against
# [PranavBansal7/wav2vec2-xlsr-aasist](https://github.com/PranavBansal7/wav2vec2-xlsr-aasist):
#
# 1. **`pad()`** (`data_utils_SSL.py`): left-crop for long audio (`x[:max_len]`),
#    tile-repeat for short audio (`np.tile(x, (1, repeats))[:, :max_len][0]`). Used
#    verbatim here, exactly as in Phase 2, so "trimmed" and "original" inputs only
#    differ in the silence-removal step and are otherwise identical pipelines.
# 2. **`genSpoof_list()`**: the 2019-style 5-column protocol format
#    (`_, key, _, _, label = line.split()`), `d_meta[key] = 1 if label == 'bonafide'
#    else 0`. Confirmed identical to Phase 1/2 usage — same labels, same target
#    convention (1 = bonafide).
# 3. **`compute_eer(target_scores, nontarget_scores)`** (`eval_metric_LA.py`): takes
#    the **bonafide** scores as `target_scores` and **spoof** scores as
#    `nontarget_scores`, returns `(eer, threshold)`. Since our score is the bonafide
#    logit (`batch_out[:, 1]`, higher = more bonafide-like, matching
#    `produce_evaluation_file()`), this is called as
#    `compute_eer(bonafide_scores, spoof_scores)` throughout — get this backwards and
#    the EER silently comes out wrong without erroring.
# 4. **`Model.forward()`**: takes raw `(batch, 64600)` waveform, returns `(batch, 2)`
#    logits, index 1 = bonafide. Same as Phase 2.

# %%
import os
import json
import random
import logging
import argparse
import dataclasses
import time
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# --- repo-local imports (same repo/files used in Phases 1-2) ---
from model import Model
from data_utils_SSL import genSpoof_list, pad
from eval_metric_LA import compute_eer

# --- publication-quality matplotlib defaults ---
plt.rcParams.update({
    'figure.figsize': (14, 8),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'pdf.fonttype': 42,   # TrueType fonts in PDF (required by most venues)
    'ps.fonttype': 42,
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase3_shortcut")

# %% [markdown]
# ## Config

# %%
@dataclasses.dataclass
class Config:
    # --- paths (reuse Phase 1/2 layout) ---
    database_path: str = "database/LA/ASVspoof2019_LA_eval/"       # must contain flac/
    protocol_path: str = "database/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    checkpoint_path: str = "pretrained_models/best_SSL_model_LA.pth"
    phase2_out_dir: str = "phase2_outputs"   # for the attribution-overlap experiment
    # Repo checkout directory (contains model.py, data_utils_SSL.py, .git/) — used
    # to record the exact commit this run was executed against (see manifest).
    repo_path: str = "."

    out_dir: str = "phase3_outputs"

    # --- audio ---
    sample_rate: int = 16000
    fixed_len: int = 64600     # matches repo's pad() default / Dataset_ASVspoof2021_eval.cut

    # --- silence detection (3.1 & 3.2) ---
    # librosa.effects.trim's top_db: audio more than top_db quieter than the clip's
    # peak is treated as silence.
    #
    # Value source, verified Aug 2026: Müller, Dieckmann, Czempin, Canals, Böttinger
    # & Williams (arXiv:2106.12914), Section 4.3 "Silence Trimming and CQT-Features":
    #   "We use the librosa library for this and employ librosa.effects.trim with a
    #    ref db of 40 as a threshold to get the duration of the leading silence."
    # librosa.effects.trim's actual threshold parameter is named top_db, so this is
    # read as top_db=40 — used here to faithfully replicate their methodology.
    #
    # NOTE: this notebook's own project plan document specifies top_db=25 in its
    # Phase 3 code stub, with no citation given for that number. That value does
    # NOT match the Müller et al. source and its origin is unknown — it looks like
    # an unsourced placeholder rather than a deliberate methodological choice. 40
    # is used below to match the paper actually being replicated; set this back to
    # 25 explicitly (and note why) if 25 was in fact an intentional deviation.
    silence_top_db: float = 40.0
    # frame/hop for librosa.effects.trim / split — defaults are frame_length=2048,
    # hop_length=512; kept explicit here for reproducibility. Not specified in the
    # Müller et al. paper, so these are librosa's own defaults, not a paper value.
    silence_frame_length: int = 2048
    silence_hop_length: int = 512

    # --- shared ---
    # Target class = 1 (bonafide), same convention verified in Phase 2 against:
    #   - genSpoof_list(): d_meta[key] = 1 if label == 'bonafide' else 0
    #   - produce_evaluation_file(): batch_out[:, 1] used as the detection score
    target_class: int = 1

    # --- 3.1 full-eval silence-removal experiment ---
    eval_batch_size: int = 128        # A100-40GB; drop to 32-64 on a smaller GPU
    num_workers: int = 4
    # Optional: path to a Phase 1 predictions CSV (columns must include utt_id,
    # bonafide_logit). No default is guessed — set this explicitly to whatever
    # Phase 1 actually wrote, e.g. "phase1_outputs/predictions.csv". Used ONLY as a
    # post-hoc consistency check (fresh Phase 3 scores vs. cached Phase 1 scores on
    # the original, untrimmed audio) — never substituted into the scores that feed
    # EER, to avoid silently mixing values from two different inference runs into
    # one number. If None (default) or the path doesn't exist, the check is
    # skipped with a log message, no error — it is not required to run Phase 3.
    phase1_predictions_path: Optional[str] = None
    # Cap the number of eval utterances processed (None = full ~71.2k-utterance eval
    # protocol). Useful for a fast dev run before committing to the full pass.
    max_eval_utterances: Optional[int] = None
    # How often (in batches) to log a processed/total + elapsed/ETA progress line
    # during the full-eval loop, in addition to the tqdm bar — useful for long
    # unattended DGX runs where you're tailing a log file rather than watching a
    # terminal.
    log_every_n_batches: int = 20

    # --- 3.3 attribution overlap with silence ---
    # Must match Phase 2's faithfulness_topk_fracs so the "top-K attributed region"
    # definition is identical across both notebooks.
    attribution_topk_fracs: tuple = (0.05, 0.10, 0.20)
    occlusion_window_for_overlap_ms: int = 50   # which Phase 2 occlusion map to use

    seed: int = 1234
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()
os.makedirs(CFG.out_dir, exist_ok=True)
os.makedirs(os.path.join(CFG.out_dir, "figures"), exist_ok=True)


def validate_paths(cfg: Config):
    """Fail fast, with a clear message, instead of discovering a bad path 20
    minutes into a full-eval-set run. Checked before the model or any data loads."""
    flac_dir = os.path.join(cfg.database_path, "flac")
    checks = [
        (cfg.protocol_path, "protocol_path", os.path.isfile),
        (cfg.database_path, "database_path", os.path.isdir),
        (flac_dir, "database_path/flac", os.path.isdir),
        (cfg.checkpoint_path, "checkpoint_path", os.path.isfile),
    ]
    missing = [(name, path) for path, name, check_fn in checks if not check_fn(path)]
    if missing:
        lines = "\n".join(f"  - {name}: {path}" for name, path in missing)
        raise FileNotFoundError(
            f"Phase 3 cannot start — the following required path(s) do not exist "
            f"or are the wrong type:\n{lines}\n"
            f"Fix Config before re-running (paths are relative to the current "
            f"working directory unless absolute)."
        )
    logger.info("All required Phase 1/2 paths verified to exist.")


validate_paths(CFG)


# %% [markdown]
# ## Reproducibility & device

# %%
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)
logger.info(f"Device: {CFG.device}")
if CFG.device == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## Load model

# %%
def load_model(cfg: Config) -> torch.nn.Module:
    args = argparse.Namespace()  # Model() takes an args param but never reads it
    model = Model(args, cfg.device).to(cfg.device)
    state_dict = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info(f"Loaded checkpoint: {cfg.checkpoint_path}")
    return model


model = load_model(CFG)

# %% [markdown]
# ## Environment & version fingerprint
#
# For reproducibility: records the exact repo commit, checkpoint file hash, and
# library/driver versions this run was executed against. This matters because
# "we replicate X" claims are only as good as knowing precisely which code and
# weights produced the numbers — a checkpoint filename alone doesn't guarantee
# it's byte-identical to the one used in an earlier run or another notebook.

# %%
def _git_commit_hash(repo_path: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.info(f"Not able to resolve git commit hash: {result.stderr.strip()}")
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.info(f"git not available to record commit hash: {e}")
        return None


def _git_dirty(repo_path: str) -> Optional[bool]:
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _file_sha256(path: str, chunk_size: int = 1 << 20) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        logger.warning(f"Could not hash {path}: {e}")
        return None


def get_environment_fingerprint(cfg: Config) -> dict:
    fp = dict(
        git_commit_hash=_git_commit_hash(cfg.repo_path),
        git_working_tree_dirty=_git_dirty(cfg.repo_path),
        checkpoint_path=cfg.checkpoint_path,
        checkpoint_sha256=_file_sha256(cfg.checkpoint_path),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        cudnn_version=(torch.backends.cudnn.version() if torch.cuda.is_available() else None),
        gpu_name=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
    )
    logger.info(
        f"Environment fingerprint — git commit: {fp['git_commit_hash']}"
        f"{' (dirty)' if fp['git_working_tree_dirty'] else ''}, "
        f"checkpoint sha256: {fp['checkpoint_sha256'][:12] if fp['checkpoint_sha256'] else None}..., "
        f"torch: {fp['torch_version']}, cuda: {fp['cuda_version']}, cudnn: {fp['cudnn_version']}"
    )
    return fp


env_fingerprint = get_environment_fingerprint(CFG)

# %% [markdown]
# ## Load full eval protocol

# %%
def load_labels(cfg: Config):
    d_label_eval, file_eval = genSpoof_list(
        dir_meta=cfg.protocol_path, is_train=False, is_eval=False
    )
    return d_label_eval, file_eval


d_label_eval, file_eval = load_labels(CFG)
logger.info(f"Total eval utterances available: {len(file_eval)}")

if CFG.max_eval_utterances is not None:
    rng = random.Random(CFG.seed)
    file_eval_subset = file_eval.copy()
    rng.shuffle(file_eval_subset)
    file_eval_subset = file_eval_subset[: CFG.max_eval_utterances]
    logger.info(
        f"max_eval_utterances={CFG.max_eval_utterances} set — using a random subset "
        f"({len(file_eval_subset)} utterances) instead of the full eval protocol."
    )
else:
    file_eval_subset = file_eval

n_bona = sum(d_label_eval[u] == 1 for u in file_eval_subset)
n_spoof = len(file_eval_subset) - n_bona
logger.info(f"Utterances to process: {len(file_eval_subset)} ({n_bona} bonafide, {n_spoof} spoof)")

# %% [markdown]
# ## Silence detection helpers
#
# `measure_and_trim_silence` operates on the **raw, unpadded** waveform (silence
# duration is a property of the recording, not of our fixed-length crop/tile). The
# trimmed audio is then passed through the repo's own `pad()` — identical to how the
# original (untrimmed) audio is padded — so the two conditions differ *only* in
# whether silence was removed before padding, nothing else.
#
# `silence_mask` operates on the **padded 64600-sample** waveform, because that's the
# resolution attribution maps live at (Phase 2 attributions are computed on the
# padded input the model actually sees).

# %%
def measure_and_trim_silence(
    audio: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, float, float]:
    """Trim leading/trailing silence from a raw waveform.

    Returns (trimmed_audio, leading_silence_ms, trailing_silence_ms).
    Falls back to the original audio (0 ms trimmed) if the entire clip is judged
    silent by the top_db threshold — trimming an all-silence clip to zero length
    would break padding, and such clips are pathological rather than informative.
    """
    trimmed, index = librosa.effects.trim(
        audio,
        top_db=cfg.silence_top_db,
        frame_length=cfg.silence_frame_length,
        hop_length=cfg.silence_hop_length,
    )
    if trimmed.size == 0:
        return audio, 0.0, 0.0

    leading_samples = index[0]
    trailing_samples = len(audio) - index[1]
    leading_ms = leading_samples / cfg.sample_rate * 1000.0
    trailing_ms = trailing_samples / cfg.sample_rate * 1000.0
    return trimmed, leading_ms, trailing_ms


def silence_mask(padded_audio: np.ndarray, cfg: Config) -> np.ndarray:
    """Boolean mask over a (padded) waveform: True = silent sample.

    Uses librosa.effects.split to find ALL non-silent intervals (not just
    leading/trailing), since interior silence (e.g. between words) also matters for
    the attribution-overlap check in 3.3.

    Caveat: for short clips that were tile-padded by the repo's pad(), the "padding"
    region is a repeat of real speech, not silence — so this mask reflects genuine
    acoustic silence in the padded signal, not an artifact of the padding scheme.
    """
    intervals = librosa.effects.split(
        padded_audio,
        top_db=cfg.silence_top_db,
        frame_length=cfg.silence_frame_length,
        hop_length=cfg.silence_hop_length,
    )
    mask = np.ones(len(padded_audio), dtype=bool)  # start all-silent
    for start, end in intervals:
        mask[start:end] = False  # mark non-silent intervals
    return mask

# %% [markdown]
# ## 3.1 — Silence removal: EER before/after
#
# Runs the model on the full eval protocol (or `max_eval_utterances` random subset)
# twice per utterance: once on the standard padded audio, once on padded audio with
# leading/trailing silence removed first. Batched via a `Dataset`/`DataLoader` so the
# A100 can process both conditions per batch efficiently.
#
# Crash-resilient: scores are appended to `silence_removal_scores.csv` after every
# batch. On restart, utterances already present in that CSV are skipped.

# %%
class SilenceRemovalDataset(Dataset):
    """Returns (orig_padded, trimmed_padded, label, leading_ms, trailing_ms, utt_id)."""

    def __init__(self, cfg: Config, utt_ids: List[str], labels: dict):
        self.cfg = cfg
        self.utt_ids = utt_ids
        self.labels = labels

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        cfg = self.cfg
        utt_id = self.utt_ids[idx]
        path = os.path.join(cfg.database_path, "flac", f"{utt_id}.flac")
        audio, _ = librosa.load(path, sr=cfg.sample_rate)

        orig_padded = pad(audio, cfg.fixed_len)

        trimmed_audio, leading_ms, trailing_ms = measure_and_trim_silence(audio, cfg)
        trimmed_padded = pad(trimmed_audio, cfg.fixed_len)

        label = self.labels[utt_id]
        return (
            torch.from_numpy(orig_padded).float(),
            torch.from_numpy(trimmed_padded).float(),
            label,
            leading_ms,
            trailing_ms,
            utt_id,
        )


def _load_phase1_scores(cfg: Config) -> Optional[pd.DataFrame]:
    """Load cached Phase 1 original-audio scores for a POST-HOC consistency check
    only. These values are never written into silence_removal_scores.csv or used
    in the EER computation — see validate_against_phase1() below."""
    if cfg.phase1_predictions_path is None:
        return None
    if not os.path.exists(cfg.phase1_predictions_path):
        logger.info(
            f"phase1_predictions_path={cfg.phase1_predictions_path} not found — "
            "skipping the Phase 1 consistency check (this is not required to run "
            "Phase 3; it's an optional sanity check)."
        )
        return None
    df = pd.read_csv(cfg.phase1_predictions_path)
    if "utt_id" not in df.columns or "bonafide_logit" not in df.columns:
        logger.warning(
            "Phase 1 predictions CSV found but missing expected columns "
            "(utt_id, bonafide_logit) — skipping the consistency check."
        )
        return None
    logger.info(f"Loaded {len(df)} Phase 1 scores for consistency check")
    return df.set_index("utt_id")["bonafide_logit"]


def validate_against_phase1(cfg: Config, df: pd.DataFrame, tol: float = 1e-3) -> Optional[dict]:
    """Cross-phase sanity check: does Phase 3's freshly-computed original-audio
    score match what Phase 1 reported for the same utterances? A large mismatch
    would indicate a checkpoint, preprocessing, or environment difference between
    the two notebooks — worth catching before trusting the EER numbers below."""
    phase1_scores = _load_phase1_scores(cfg)
    if phase1_scores is None:
        return None

    merged = df.set_index("utt_id")[["orig_score"]].join(
        phase1_scores.rename("phase1_score"), how="inner"
    )
    if merged.empty:
        logger.warning(
            "Phase 1 CSV loaded but shares no utt_ids with this run's subset — "
            "consistency check skipped (likely a different max_eval_utterances sample)."
        )
        return None

    abs_diff = (merged["orig_score"] - merged["phase1_score"]).abs()
    n_mismatch = int((abs_diff > tol).sum())
    result = dict(
        n_compared=len(merged),
        mean_abs_diff=float(abs_diff.mean()),
        max_abs_diff=float(abs_diff.max()),
        n_mismatch_above_tol=n_mismatch,
        tol=tol,
    )
    if n_mismatch > 0:
        logger.warning(
            f"Phase1 consistency check: {n_mismatch}/{len(merged)} utterances differ "
            f"from Phase 1 scores by more than {tol} (max diff={result['max_abs_diff']:.4f}). "
            "Check that Phase 3 is using the same checkpoint/preprocessing as Phase 1."
        )
    else:
        logger.info(
            f"Phase1 consistency check passed: {len(merged)} utterances compared, "
            f"max abs diff={result['max_abs_diff']:.2e}"
        )
    return result


def run_silence_removal_experiment(
    model, cfg: Config, utt_ids: List[str], labels: dict
) -> pd.DataFrame:
    csv_path = os.path.join(cfg.out_dir, "silence_removal_scores.csv")
    columns = [
        "utt_id", "label", "orig_score", "trimmed_score",
        "orig_pred", "trimmed_pred", "leading_silence_ms", "trailing_silence_ms",
    ]

    done_ids = set()
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        done_ids = set(existing["utt_id"].tolist())
        logger.info(f"Resuming: {len(done_ids)} utterances already scored")
    else:
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False)

    remaining = [u for u in utt_ids if u not in done_ids]
    logger.info(f"Utterances remaining to score: {len(remaining)}")

    if remaining:
        dataset = SilenceRemovalDataset(cfg, remaining, labels)
        loader = DataLoader(
            dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(cfg.device == "cuda"),
        )
        n_batches = len(loader)
        n_remaining = len(remaining)
        run_start = time.time()
        n_processed = 0

        for batch_idx, (orig_batch, trimmed_batch, label_batch, lead_ms, trail_ms, utt_batch) in enumerate(
            tqdm(loader, desc="Phase 3.1 silence removal"), start=1
        ):
            orig_batch = orig_batch.to(cfg.device, non_blocking=True)
            trimmed_batch = trimmed_batch.to(cfg.device, non_blocking=True)

            with torch.no_grad():
                orig_out = model(orig_batch)
                trimmed_out = model(trimmed_batch)

            orig_scores = orig_out[:, cfg.target_class].cpu().numpy()
            trimmed_scores = trimmed_out[:, cfg.target_class].cpu().numpy()
            orig_preds = orig_out.argmax(dim=1).cpu().numpy()
            trimmed_preds = trimmed_out.argmax(dim=1).cpu().numpy()

            batch_records = []
            for i, utt_id in enumerate(utt_batch):
                batch_records.append(dict(
                    utt_id=utt_id,
                    label=int(label_batch[i]),
                    orig_score=float(orig_scores[i]),
                    trimmed_score=float(trimmed_scores[i]),
                    orig_pred=int(orig_preds[i]),
                    trimmed_pred=int(trimmed_preds[i]),
                    leading_silence_ms=float(lead_ms[i]),
                    trailing_silence_ms=float(trail_ms[i]),
                ))
            pd.DataFrame.from_records(batch_records).to_csv(
                csv_path, mode="a", header=False, index=False
            )
            n_processed += len(utt_batch)

            # Explicit timed progress log (in addition to the tqdm bar) — useful
            # for long unattended DGX runs where stdout is being tailed to a file
            # rather than watched live, since tqdm's carriage-return updates don't
            # render usefully in a plain log file.
            if batch_idx % cfg.log_every_n_batches == 0 or batch_idx == n_batches:
                elapsed = time.time() - run_start
                rate = n_processed / elapsed if elapsed > 0 else float("nan")
                remaining_utts = n_remaining - n_processed
                eta_seconds = remaining_utts / rate if rate > 0 else float("nan")
                logger.info(
                    f"[3.1] batch {batch_idx}/{n_batches}  "
                    f"utterances {n_processed}/{n_remaining}  "
                    f"elapsed={timedelta(seconds=int(elapsed))}  "
                    f"rate={rate:.1f} utt/s  "
                    f"ETA={timedelta(seconds=int(eta_seconds)) if rate > 0 else 'n/a'}"
                )

    return pd.read_csv(csv_path)


silence_removal_df = run_silence_removal_experiment(model, CFG, file_eval_subset, d_label_eval)
logger.info(f"Total scored utterances: {len(silence_removal_df)}")
silence_removal_df.head(10)

# %% [markdown]
# ### (Optional) Cross-phase consistency check against Phase 1

# %%
phase1_consistency = validate_against_phase1(CFG, silence_removal_df)
if phase1_consistency is not None:
    with open(os.path.join(CFG.out_dir, "phase1_consistency_check.json"), "w") as f:
        json.dump(phase1_consistency, f, indent=2)

# %% [markdown]
# ### EER before/after silence trimming

# %%
def compute_eer_before_after(df: pd.DataFrame) -> dict:
    bona = df[df["label"] == 1]
    spoof = df[df["label"] == 0]

    eer_orig, thresh_orig = compute_eer(
        bona["orig_score"].to_numpy(), spoof["orig_score"].to_numpy()
    )
    eer_trimmed, thresh_trimmed = compute_eer(
        bona["trimmed_score"].to_numpy(), spoof["trimmed_score"].to_numpy()
    )

    # Prediction-flip diagnostics: how often does removing silence change the
    # hard decision (argmax class), broken down by true label.
    flips = df["orig_pred"] != df["trimmed_pred"]
    flip_rate_overall = flips.mean()
    flip_rate_bona = flips[df["label"] == 1].mean() if (df["label"] == 1).any() else float("nan")
    flip_rate_spoof = flips[df["label"] == 0].mean() if (df["label"] == 0).any() else float("nan")

    result = dict(
        eer_original_pct=eer_orig * 100,
        eer_threshold_original=float(thresh_orig),
        eer_trimmed_pct=eer_trimmed * 100,
        eer_threshold_trimmed=float(thresh_trimmed),
        eer_delta_pct=(eer_trimmed - eer_orig) * 100,
        n_bonafide=len(bona),
        n_spoof=len(spoof),
        prediction_flip_rate_overall=float(flip_rate_overall),
        prediction_flip_rate_bonafide=float(flip_rate_bona),
        prediction_flip_rate_spoof=float(flip_rate_spoof),
    )
    return result


eer_results = compute_eer_before_after(silence_removal_df)
with open(os.path.join(CFG.out_dir, "eer_before_after_silence.json"), "w") as f:
    json.dump(eer_results, f, indent=2)

logger.info(
    f"EER original:  {eer_results['eer_original_pct']:.3f}%  "
    f"(n_bonafide={eer_results['n_bonafide']}, n_spoof={eer_results['n_spoof']})"
)
logger.info(f"EER after silence trim: {eer_results['eer_trimmed_pct']:.3f}%")
logger.info(f"EER delta (trimmed - original): {eer_results['eer_delta_pct']:+.3f} pp")
logger.info(
    f"Prediction flip rate: overall={eer_results['prediction_flip_rate_overall']:.4f}  "
    f"bonafide={eer_results['prediction_flip_rate_bonafide']:.4f}  "
    f"spoof={eer_results['prediction_flip_rate_spoof']:.4f}"
)
print(json.dumps(eer_results, indent=2))

# %% [markdown]
# ## 3.2 — Silence duration distributions
#
# Plots leading/trailing silence duration by class. If bonafide and spoof come from
# separated distributions, that's a **dataset-level** bias the model could exploit
# regardless of whether it actually attends to synthesis artifacts.

# %%
def plot_silence_distributions(cfg: Config, df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    df_plot = df.copy()
    df_plot["Class"] = df_plot["label"].map({1: "Bonafide", 0: "Spoof"})

    for ax, col, title in zip(
        axes,
        ["leading_silence_ms", "trailing_silence_ms"],
        ["Leading Silence Duration", "Trailing Silence Duration"],
    ):
        sns.boxplot(
            data=df_plot, x="Class", y=col, hue="Class",
            palette={"Bonafide": "steelblue", "Spoof": "coral"},
            width=0.5, legend=False, ax=ax, showfliers=False,
        )
        sns.stripplot(
            data=df_plot.sample(min(2000, len(df_plot)), random_state=cfg.seed),
            x="Class", y=col, color="black", alpha=0.15, size=2, ax=ax,
        )
        ax.set_title(title, fontsize=13)
        ax.set_ylabel("Duration (ms)")
        ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(cfg.out_dir, "figures", "silence_duration_distributions.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    logger.info(f"Saved silence duration distribution figure: {path}")


plot_silence_distributions(CFG, silence_removal_df)

# %% [markdown]
# ### Silence-only shortcut probe
#
# Reproduces the specific claim in Müller et al. (2021): that leading-silence
# duration *alone* is predictive of the bonafide/spoof label. We treat
# `leading_silence_ms` (and separately `trailing_silence_ms`) as a 1-D detection
# score and run it through the same `compute_eer` used for the full model, plus a
# distribution-separation test (Mann-Whitney U — nonparametric, doesn't assume
# normally-distributed silence durations).
#
# This does **not** tell us whether the *model* uses silence as a shortcut (that's
# 3.3) — only whether the *dataset* contains a silence-based shortcut a model could
# in principle exploit.

# %%
def silence_only_shortcut_probe(cfg: Config, df: pd.DataFrame) -> dict:
    bona = df[df["label"] == 1]
    spoof = df[df["label"] == 0]

    results = {}
    for col in ["leading_silence_ms", "trailing_silence_ms"]:
        # Higher silence duration is not a priori known to point toward either class,
        # so evaluate EER using duration as-is AND with the sign flipped, and report
        # whichever orientation is more separable (that's the meaningful "can this
        # scalar alone classify the pair" question; the model doesn't know the sign
        # convention we chose either).
        eer_pos, _ = compute_eer(bona[col].to_numpy(), spoof[col].to_numpy())
        eer_neg, _ = compute_eer((-bona[col]).to_numpy(), (-spoof[col]).to_numpy())
        eer = min(eer_pos, eer_neg)
        orientation = "longer_silence=bonafide" if eer_pos <= eer_neg else "longer_silence=spoof"

        u_stat, p_value = stats.mannwhitneyu(
            bona[col], spoof[col], alternative="two-sided"
        )

        results[col] = dict(
            eer_pct=eer * 100,
            orientation=orientation,
            mean_bonafide_ms=float(bona[col].mean()),
            mean_spoof_ms=float(spoof[col].mean()),
            median_bonafide_ms=float(bona[col].median()),
            median_spoof_ms=float(spoof[col].median()),
            mannwhitney_u=float(u_stat),
            mannwhitney_p=float(p_value),
        )
        logger.info(
            f"[{col}] silence-only EER: {eer*100:.2f}%  ({orientation})  "
            f"mean(bonafide)={results[col]['mean_bonafide_ms']:.1f}ms  "
            f"mean(spoof)={results[col]['mean_spoof_ms']:.1f}ms  "
            f"Mann-Whitney p={p_value:.2e}"
        )

    with open(os.path.join(cfg.out_dir, "silence_only_shortcut_probe.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


silence_probe_results = silence_only_shortcut_probe(CFG, silence_removal_df)
print(json.dumps(silence_probe_results, indent=2))

# %% [markdown]
# ## 3.3 — Do attributions highlight silence?
#
# Reuses the 100 samples (50 bonafide + 50 spoof) and saved attribution maps from
# Phase 2 (`phase2_outputs/attributions/<utt_id>/`). For each sample, computes a
# silence mask over the **padded** waveform (same resolution as the attribution
# arrays), then measures what fraction of the top-K attributed region falls inside
# silence, for K in `attribution_topk_fracs` — matching Phase 2's faithfulness
# definition of top-K exactly. A baseline "expected overlap under random attribution"
# (= the silent fraction of the signal itself) is reported alongside, since a sample
# that's 40% silence will show high overlap by chance alone.

# %%
def _load_phase2_selected_samples(cfg: Config) -> List[Tuple[str, int]]:
    path = os.path.join(cfg.phase2_out_dir, "selected_samples.json")
    assert os.path.isfile(path), (
        f"Phase 3.3 requires Phase 2's output — expected to find it at: {path}\n"
        f"Run the Phase 2 notebook first (it writes selected_samples.json and the "
        f"per-utterance attribution .npy files), or point phase2_out_dir at the "
        f"correct location if it's already been run."
    )
    with open(path, "r") as f:
        selected = json.load(f)
    return [(u, l) for u, l in selected]


def _load_padded_waveform(cfg: Config, utt_id: str) -> np.ndarray:
    """Reproduces Phase 2's load_waveform exactly: repo pad() on librosa-loaded audio."""
    path = os.path.join(cfg.database_path, "flac", f"{utt_id}.flac")
    audio, _ = librosa.load(path, sr=cfg.sample_rate)
    audio_padded = pad(audio, cfg.fixed_len)
    assert len(audio_padded) == cfg.fixed_len, \
        f"pad() returned length {len(audio_padded)}, expected {cfg.fixed_len}"
    return audio_padded


def attribution_silence_overlap(
    attribution: np.ndarray, sil_mask: np.ndarray, top_frac: float
) -> Tuple[float, float]:
    """Returns (overlap_fraction, expected_overlap_under_chance).

    overlap_fraction: of the top-K attributed samples (by |attribution|), what
    fraction fall in silent regions.
    expected_overlap_under_chance: the silent fraction of the whole signal — what
    overlap_fraction would be if attribution were uniformly random.
    """
    n = len(attribution)
    k = max(1, int(n * top_frac))
    top_idx = np.argsort(-np.abs(attribution))[:k]
    overlap = sil_mask[top_idx].mean()
    chance = sil_mask.mean()
    return float(overlap), float(chance)


def run_attribution_silence_overlap(cfg: Config) -> pd.DataFrame:
    selected_samples = _load_phase2_selected_samples(cfg)
    occ_key = f"occlusion_{cfg.occlusion_window_for_overlap_ms}ms.npy"

    records = []
    missing = []
    for utt_id, label in tqdm(selected_samples, desc="Phase 3.3 attribution overlap"):
        sample_dir = os.path.join(cfg.phase2_out_dir, "attributions", utt_id)
        occ_path = os.path.join(sample_dir, occ_key)
        ig_path = os.path.join(sample_dir, "integrated_gradients.npy")
        if not (os.path.exists(occ_path) and os.path.exists(ig_path)):
            missing.append(utt_id)
            continue

        padded_audio = _load_padded_waveform(cfg, utt_id)
        sil_mask = silence_mask(padded_audio, cfg)
        silence_frac = float(sil_mask.mean())

        attr_occ = np.load(occ_path)
        attr_ig = np.load(ig_path)

        for method, attr in [("occlusion", attr_occ), ("integrated_gradients", attr_ig)]:
            for top_frac in cfg.attribution_topk_fracs:
                overlap, chance = attribution_silence_overlap(attr, sil_mask, top_frac)
                records.append(dict(
                    utt_id=utt_id, label=label, method=method, topk_frac=top_frac,
                    silence_overlap_frac=overlap,
                    silence_frac_in_signal=silence_frac,
                    overlap_over_chance=(overlap / chance) if chance > 0 else np.nan,
                ))

    if missing:
        logger.warning(
            f"{len(missing)} Phase 2 samples missing attribution files, skipped: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    df = pd.DataFrame.from_records(records)
    df.to_csv(os.path.join(cfg.out_dir, "attribution_silence_overlap.csv"), index=False)
    return df


attribution_overlap_df = run_attribution_silence_overlap(CFG)
logger.info(f"Attribution overlap rows: {len(attribution_overlap_df)}")
attribution_overlap_df.head(10)

# %% [markdown]
# ### Attribution-silence overlap summary + figure

# %%
def summarize_and_plot_attribution_overlap(cfg: Config, df: pd.DataFrame):
    df_plot = df.copy()
    df_plot["Class"] = df_plot["label"].map({1: "Bonafide", 0: "Spoof"})

    summary = (
        df_plot.groupby(["method", "topk_frac", "Class"])[
            ["silence_overlap_frac", "overlap_over_chance"]
        ]
        .agg(["mean", "std"])
    )
    summary.to_csv(os.path.join(cfg.out_dir, "attribution_silence_overlap_summary.csv"))
    print("\n--- Attribution-Silence Overlap Summary ---\n")
    print(summary)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    pivot_overlap = (
        df_plot.groupby(["method", "topk_frac"])["silence_overlap_frac"].mean().unstack("topk_frac")
    )
    pivot_overlap.index = [m.replace("_", " ").title() for m in pivot_overlap.index]
    pivot_overlap.columns = [f"Top {int(c*100)}%" for c in pivot_overlap.columns]
    pivot_overlap.plot(kind="bar", ax=axes[0], rot=0, edgecolor="white", linewidth=0.8)
    axes[0].set_title("Fraction of Top-K Attribution Falling in Silence", fontsize=12)
    axes[0].set_ylabel("Overlap fraction")
    axes[0].set_xlabel("")
    axes[0].legend(title="Top-K", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    pivot_chance = (
        df_plot.groupby(["method", "topk_frac"])["overlap_over_chance"].mean().unstack("topk_frac")
    )
    pivot_chance.index = [m.replace("_", " ").title() for m in pivot_chance.index]
    pivot_chance.columns = [f"Top {int(c*100)}%" for c in pivot_chance.columns]
    pivot_chance.plot(kind="bar", ax=axes[1], rot=0, edgecolor="white", linewidth=0.8)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_title("Overlap Relative to Chance (1.0 = no silence preference)", fontsize=12)
    axes[1].set_ylabel("Overlap / silence fraction in signal")
    axes[1].set_xlabel("")
    axes[1].legend(title="Top-K", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(cfg.out_dir, "figures", "attribution_silence_overlap.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    logger.info(f"Saved attribution-silence overlap figure: {path}")

    return summary


overlap_summary = summarize_and_plot_attribution_overlap(CFG, attribution_overlap_df)

# %% [markdown]
# ## Run manifest

# %%
manifest = dict(
    phase="Phase 3 — Shortcut Analysis (Replication)",
    timestamp=datetime.now().isoformat(),
    config=dataclasses.asdict(CFG),
    environment_fingerprint=env_fingerprint,
    n_eval_utterances_scored=len(silence_removal_df),
    n_attribution_overlap_rows=len(attribution_overlap_df),
    eer_results=eer_results,
    phase1_consistency_check=phase1_consistency,
    silence_only_shortcut_probe=silence_probe_results,
    numpy_version=np.__version__,
    librosa_version=librosa.__version__,
    source_verified=(
        "Aug 2026 — pad(), genSpoof_list(), compute_eer(target=bonafide, "
        "nontarget=spoof), and Model.forward() verified against repo source. "
        "silence_top_db=40 verified against Muller et al. (arXiv:2106.12914) "
        "Section 4.3 (\"ref db of 40\"); this project's own Phase 3 plan document "
        "specified top_db=25 with no cited source for that number."
    ),
)
with open(os.path.join(CFG.out_dir, "run_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, default=str)

logger.info("Phase 3 complete.")
print(json.dumps({k: v for k, v in manifest.items() if k != "config"}, indent=2, default=str))

# %% [markdown]
# ## Output inventory

# %%
def print_output_inventory(cfg: Config):
    out = Path(cfg.out_dir)
    print(f"\n{'='*60}")
    print(f"Phase 3 Output Inventory: {out.resolve()}")
    print(f"{'='*60}")

    top_files = [f for f in out.iterdir() if f.is_file()]
    print(f"\nTop-level files ({len(top_files)}):")
    for f in sorted(top_files):
        print(f"  {f.name:45s}  {f.stat().st_size / 1024:.1f} KB")

    fig_dir = out / "figures"
    if fig_dir.exists():
        figs = list(fig_dir.iterdir())
        print(f"\nFigures ({len(figs)}):")
        for f in sorted(figs):
            print(f"  {f.name:45s}  {f.stat().st_size / 1024:.1f} KB")

    print(f"\n{'='*60}")
    print("✅ EER before/after silence trimming computed on the full eval protocol")
    print("✅ Silence-only shortcut probe (dataset-level bias) quantified")
    print("✅ Attribution-silence overlap computed for the Phase 2 100-sample subset")
    print("✅ Phase 3 outputs ready for Phase 4 (Hindi data prep) / Phase 5 comparison")


print_output_inventory(CFG)