# Phase 2 — XAI on English (Replication)

#Occlusion + Integrated Gradients attribution analysis, plus comprehensiveness/sufficiency
#faithfulness scoring, for the wav2vec2-XLS-R + AASIST detector on ASVspoof2019 LA eval.

**Environment:** NVIDIA DGX A100 (40GB), Python 3.8.20, torch 1.13.1

**Before running:** this notebook reuses `model.py`, `data_utils_SSL.py`, and
`eval_metric_LA.py` from the same `SSL_Anti-spoofing` checkout used in Phase 1, and assumes
the same directory layout (`database/LA/...`, `pretrained_models/...`) and the same
checkpoint (`best_SSL_model_LA.pth`).

**Source verification (Aug 2026):** all preprocessing, target class, and forward-pass
assumptions in this notebook have been verified against the actual repository source
([PranavBansal7/wav2vec2-xlsr-aasist](https://github.com/PranavBansal7/wav2vec2-xlsr-aasist)).
Key verifications:

1. **Preprocessing**: uses the repo's own `pad()` function from `data_utils_SSL.py`
   (left-crop for long audio, tile-repeat for short) — not a custom reimplementation.
2. **Target class**: `target_class=1` = bonafide, confirmed via `genSpoof_list()`,
   `CrossEntropyLoss(weight=[0.1, 0.9])`, and `batch_out[:, 1]` in `produce_evaluation_file()`.
3. **Forward pass**: `Model.forward()` takes raw `(batch, 64600)` waveform and returns
   `(batch, 2)` logits. No intermediate preprocessing is bypassed.
4. **IG shape**: Captum returns attributions with the same shape as input → `(64600,)` per sample.

## Imports


```python
import os
import json
import random
import logging
import argparse
import dataclasses
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from captum.attr import IntegratedGradients

# --- repo-local imports (same repo/files used in Phase 1) ---
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
logger = logging.getLogger("phase2_xai")
```

## Config


```python
@dataclasses.dataclass
class Config:
    # --- paths (reuse Phase 1 layout) ---
    database_path: str = "database/LA/ASVspoof2019_LA_eval/"       # must contain flac/
    protocol_path: str = "database/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    checkpoint_path: str = "pretrained_models/best_SSL_model_LA.pth"

    out_dir: str = "phase2_outputs"

    # --- sampling ---
    n_bonafide: int = 50
    n_spoof: int = 50
    seed: int = 1234

    # --- audio ---
    sample_rate: int = 16000
    # Matches the repo's pad() default and Dataset_ASVspoof2021_eval.cut (both 64600).
    # Verified against data_utils_SSL.py.
    fixed_len: int = 64600

    # --- occlusion ---
    occlusion_window_sizes_ms: tuple = (20, 50, 100)
    occlusion_step_frac: float = 0.5   # step = window * this fraction (50% overlap)
    occlusion_batch_size: int = 128    # A100-40GB can handle 128 comfortably;
                                       # drop to 64 if you hit OOM on a smaller GPU

    # --- integrated gradients ---
    ig_n_steps: int = 50

    # --- shared ---
    # Target class = 1 (bonafide) for ALL samples, both bonafide and spoof.
    # Verified against source:
    #   - genSpoof_list(): d_meta[key] = 1 if label == 'bonafide' else 0
    #   - CrossEntropyLoss(weight=[0.1, 0.9]): weight[1]=0.9 → class 1 = bonafide
    #   - produce_evaluation_file(): batch_out[:, 1] used as the detection score
    #   - Model.out_layer: nn.Linear(..., 2) → index 0=spoof, index 1=bonafide
    # Using the same target for all samples is critical: attributions must answer the
    # same question ("what drives the bonafide score?") for every sample, so that
    # cross-sample and cross-language comparisons in Phase 5 are valid.
    target_class: int = 1

    # --- faithfulness ---
    faithfulness_topk_fracs: tuple = (0.05, 0.10, 0.20)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()
os.makedirs(CFG.out_dir, exist_ok=True)
os.makedirs(os.path.join(CFG.out_dir, "attributions"), exist_ok=True)
os.makedirs(os.path.join(CFG.out_dir, "figures"), exist_ok=True)
```

## Reproducibility & device


```python
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
    logger.info(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
```

## Load model + verify output shape


```python
def load_model(cfg: Config) -> torch.nn.Module:
    args = argparse.Namespace()  # Model() takes an args param but never reads it (Phase 1 note)
    model = Model(args, cfg.device).to(cfg.device)
    state_dict = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    model.load_state_dict(state_dict)
    model.eval()

    # Freeze weights -- gradients w.r.t. *inputs* still flow for Integrated Gradients,
    # since we only disable requires_grad on the parameters, not on the input tensor.
    for p in model.parameters():
        p.requires_grad_(False)

    logger.info(f"Loaded checkpoint: {cfg.checkpoint_path}")
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {n_params / 1e6:.1f}M")
    return model


model = load_model(CFG)

# --- Verify model output shape ---
# Model.forward() should take (batch, 64600) and return (batch, 2).
dummy_input = torch.randn(1, CFG.fixed_len, device=CFG.device)
with torch.no_grad():
    dummy_out = model(dummy_input)
assert dummy_out.shape == (1, 2), \
    f"Model output shape mismatch: expected (1, 2), got {dummy_out.shape}"
logger.info(f"Model output shape verified: {dummy_out.shape} — [spoof_logit, bonafide_logit]")
```

## Select the ~100-sample subset (50 bonafide + 50 spoof)


```python
def load_labels(cfg: Config):
    # is_train=False, is_eval=False: the 2019 LA eval protocol ships in the standard
    # 5-column labeled format, not the bare-utt-id 2021-style trial list (Phase 1 note).
    d_label_eval, file_eval = genSpoof_list(
        dir_meta=cfg.protocol_path, is_train=False, is_eval=False
    )
    return d_label_eval, file_eval


d_label_eval, file_eval = load_labels(CFG)
logger.info(f"Total eval utterances available: {len(file_eval)}")


def select_samples(cfg: Config, d_label_eval, file_eval):
    bonafide_ids = [u for u in file_eval if d_label_eval[u] == 1]
    spoof_ids = [u for u in file_eval if d_label_eval[u] == 0]

    # Isolated RNG so that other random calls elsewhere cannot perturb sample selection.
    rng = random.Random(cfg.seed)
    rng.shuffle(bonafide_ids)
    rng.shuffle(spoof_ids)

    if len(bonafide_ids) < cfg.n_bonafide or len(spoof_ids) < cfg.n_spoof:
        raise ValueError(
            f"Not enough samples available: {len(bonafide_ids)} bonafide, "
            f"{len(spoof_ids)} spoof (need {cfg.n_bonafide}/{cfg.n_spoof})."
        )

    sel_bonafide = bonafide_ids[: cfg.n_bonafide]
    sel_spoof = spoof_ids[: cfg.n_spoof]
    selected = [(u, 1) for u in sel_bonafide] + [(u, 0) for u in sel_spoof]
    rng.shuffle(selected)
    return selected


selected_samples = select_samples(CFG, d_label_eval, file_eval)
logger.info(
    f"Selected {len(selected_samples)} samples "
    f"({CFG.n_bonafide} bonafide + {CFG.n_spoof} spoof)"
)

with open(os.path.join(CFG.out_dir, "selected_samples.json"), "w") as f:
    json.dump(selected_samples, f, indent=2)
```

## Target class sanity check

Print raw logits and softmax probabilities for one bonafide and one spoof sample.
Expected: bonafide sample has `[:,1] > [:,0]`; spoof sample has `[:,0] > [:,1]`.


```python
def sanity_check_target_class(model, cfg, selected_samples):
    """Print raw logits + softmax for one bonafide and one spoof sample."""
    sample_bonafide = next((u, l) for u, l in selected_samples if l == 1)
    sample_spoof = next((u, l) for u, l in selected_samples if l == 0)

    for utt_id, label in [sample_bonafide, sample_spoof]:
        audio = load_waveform(cfg, utt_id)
        x = torch.tensor(audio, dtype=torch.float32, device=cfg.device).unsqueeze(0)
        with torch.no_grad():
            raw_out = model(x)
            probs = torch.softmax(raw_out, dim=1)
        label_str = "BONAFIDE" if label == 1 else "SPOOF"
        logger.info(
            f"Target class check [{label_str}] {utt_id}:\n"
            f"  raw logits [spoof, bonafide] = {raw_out.cpu().numpy().ravel()}\n"
            f"  softmax    [spoof, bonafide] = {probs.cpu().numpy().ravel()}\n"
            f"  bonafide_logit ([:,1])        = {raw_out[0, 1].item():.4f}"
        )

# Defined here but called after load_waveform is defined (next cell)
# — will be called explicitly below.
```

## Audio I/O helpers

Uses the repo's own `pad()` function from `data_utils_SSL.py` to guarantee
identical preprocessing to the eval pipeline:
- Long audio (≥64600 samples): **left-crop** — `x[:max_len]`
- Short audio (<64600 samples): **tile-repeat** — `np.tile(x, (1, repeats))[:, :max_len][0]`


```python
def load_waveform(cfg: Config, utt_id: str) -> np.ndarray:
    """Load and preprocess a waveform exactly as the eval pipeline does.

    Uses the repo's own pad() from data_utils_SSL.py — NOT a custom reimplementation.
    This guarantees that attribution maps explain the same input the model saw at eval time.
    """
    path = os.path.join(cfg.database_path, "flac", f"{utt_id}.flac")
    audio, sr = librosa.load(path, sr=cfg.sample_rate)
    audio_padded = pad(audio, cfg.fixed_len)
    assert len(audio_padded) == cfg.fixed_len, \
        f"pad() returned length {len(audio_padded)}, expected {cfg.fixed_len}"
    return audio_padded


def model_score(model, cfg: Config, audio: np.ndarray) -> float:
    """Bonafide-class logit for a single waveform, no grad."""
    x = torch.tensor(audio, dtype=torch.float32, device=cfg.device).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
    return out[0, cfg.target_class].item()


# Now that load_waveform is defined, run the target class sanity check:
sanity_check_target_class(model, CFG, selected_samples)
```

## Occlusion attribution

Sliding-window occlusion over the raw waveform, at 20ms / 50ms / 100ms windows
(project spec), with 50% overlap between window positions. Batched across window
positions so the A100 does many masked forward passes per call instead of one at a
time — on a 4s clip at 20ms/10ms-step this is ~400 window positions; batching in
groups of 128 keeps this tractable across 100 samples × 3 window sizes.


```python
@torch.no_grad()
def occlusion_attribution(
    model, cfg: Config, audio: np.ndarray, window_ms: int
) -> np.ndarray:
    """Compute sample-level occlusion importance for a single waveform.

    Uses overlapping windows (50% step) and averages importance across all
    windows that cover each sample — this produces smoother, higher-resolution
    maps than non-overlapping windows.
    """
    window_size = int(cfg.sample_rate * window_ms / 1000)
    step = max(1, int(window_size * cfg.occlusion_step_frac))
    n_samples = len(audio)

    starts = list(range(0, n_samples, step))
    base_x = torch.tensor(audio, dtype=torch.float32, device=cfg.device)
    orig_score = model_score(model, cfg, audio)

    importance_sum = np.zeros(n_samples, dtype=np.float64)
    importance_count = np.zeros(n_samples, dtype=np.float64)

    for batch_start in range(0, len(starts), cfg.occlusion_batch_size):
        batch_positions = starts[batch_start : batch_start + cfg.occlusion_batch_size]
        batch_x = base_x.unsqueeze(0).repeat(len(batch_positions), 1).clone()
        for i, t in enumerate(batch_positions):
            batch_x[i, t : t + window_size] = 0.0

        batch_out = model(batch_x)
        batch_scores = batch_out[:, cfg.target_class].detach().cpu().numpy()

        for i, t in enumerate(batch_positions):
            delta = abs(orig_score - float(batch_scores[i]))
            end = min(t + window_size, n_samples)
            importance_sum[t:end] += delta
            importance_count[t:end] += 1

    importance_count[importance_count == 0] = 1.0
    return (importance_sum / importance_count).astype(np.float32)
```

## Integrated Gradients attribution

Zero-signal (silence) baseline — matches the "silence = absence of signal" baseline
implicit in the occlusion method above, so the two methods are comparing attributions
against the same counterfactual.

`internal_batch_size=ig_n_steps` batches all interpolation steps into a single forward
pass — the A100's 40GB VRAM handles this comfortably for a single-sample input.


```python
def integrated_gradients_attribution(
    model, cfg: Config, audio: np.ndarray
) -> tuple:
    """Compute sample-level Integrated Gradients attribution for a single waveform."""
    ig = IntegratedGradients(lambda x: model(x))

    x = torch.tensor(audio, dtype=torch.float32, device=cfg.device).unsqueeze(0)
    x.requires_grad_(True)
    baseline = torch.zeros_like(x)

    attributions, delta = ig.attribute(
        x,
        baselines=baseline,
        target=cfg.target_class,
        n_steps=cfg.ig_n_steps,
        internal_batch_size=cfg.ig_n_steps,  # batch all interpolation steps at once (A100)
        return_convergence_delta=True,
    )
    attr_np = attributions.squeeze(0).detach().cpu().numpy()
    delta_value = float(delta.item())

    # Verify IG returns attributions at waveform resolution, not some internal
    # representation (e.g. wav2vec2's 1024-dim frame embeddings).
    assert attr_np.shape == (cfg.fixed_len,), \
        f"IG attribution shape mismatch: expected ({cfg.fixed_len},), got {attr_np.shape}"

    return attr_np, delta_value
```

## Faithfulness: comprehensiveness & sufficiency

- **Comprehensiveness**: mask the top-K most-attributed samples — how much does the
  bonafide score drop? Higher = more faithful (the attribution correctly found the
  regions the model actually relies on).
- **Sufficiency**: keep *only* the top-K most-attributed samples (zero everything
  else) — does the score survive? Higher (closer to the original score) = more
  faithful (the top-K region alone is sufficient to reproduce the decision).

K is defined here as a contiguous top-K% of the sample-level attribution magnitude,
which applies cleanly to both occlusion (after window-averaging to sample resolution)
and IG (already sample-resolution).


```python
def comprehensiveness_sufficiency(
    model, cfg: Config, audio: np.ndarray, attribution: np.ndarray, top_frac: float
):
    """Compute comprehensiveness (score drop on masking top-K) and sufficiency
    (raw score when keeping only top-K).

    Returns:
        comprehensiveness: orig_score - masked_score (higher = more faithful)
        sufficiency: score from top-K-only input (higher = more faithful)
    """
    orig_score = model_score(model, cfg, audio)
    n = len(audio)
    k = max(1, int(n * top_frac))
    top_idx = np.argsort(-np.abs(attribution))[:k]

    # Comprehensiveness: remove top-K, score should drop
    masked_comp = audio.copy()
    masked_comp[top_idx] = 0.0
    comp_score = model_score(model, cfg, masked_comp)
    comprehensiveness = orig_score - comp_score

    # Sufficiency: keep ONLY top-K, score should survive
    masked_suff = np.zeros_like(audio)
    masked_suff[top_idx] = audio[top_idx]
    suff_score = model_score(model, cfg, masked_suff)
    sufficiency = suff_score

    return comprehensiveness, sufficiency
```

## Main pipeline

Runs occlusion (3 window sizes) + Integrated Gradients on every selected sample,
saves each attribution map to disk (`.npy`, one file per sample per method), and
scores comprehensiveness/sufficiency at each top-K fraction.

Crash-resilient: attribution `.npy` files are saved per-sample inside the loop,
and faithfulness rows are appended to the CSV after each sample completes.
On restart, samples whose attribution directories already contain all expected
files are skipped automatically — no GPU work is repeated.

Also saves per-sample predictions (logits, probabilities, predicted class) to
`predictions.csv` for downstream analysis.


```python
def _sample_is_complete(cfg: Config, utt_id: str) -> bool:
    """Check whether all expected attribution files already exist for this sample."""
    sample_dir = os.path.join(cfg.out_dir, "attributions", utt_id)
    if not os.path.isdir(sample_dir):
        return False
    expected = [f"occlusion_{w}ms.npy" for w in cfg.occlusion_window_sizes_ms]
    expected.append("integrated_gradients.npy")
    return all(os.path.exists(os.path.join(sample_dir, f)) for f in expected)


def run_phase2(model, cfg: Config, selected_samples):
    csv_path = os.path.join(cfg.out_dir, "faithfulness_scores.csv")
    pred_path = os.path.join(cfg.out_dir, "predictions.csv")
    all_records = []
    failed = []
    skipped = 0

    # If resuming, load existing CSV rows so the final DataFrame is complete
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        all_records = existing_df.to_dict('records')
        logger.info(f"Resuming: loaded {len(all_records)} existing faithfulness rows")
    else:
        # Write CSV header
        pd.DataFrame(columns=[
            'utt_id', 'label', 'method', 'window_ms', 'topk_frac',
            'comprehensiveness', 'sufficiency'
        ]).to_csv(csv_path, index=False)

    # Predictions CSV header
    if not os.path.exists(pred_path):
        pd.DataFrame(columns=[
            'utt_id', 'label', 'bonafide_logit', 'spoof_logit',
            'bonafide_prob', 'spoof_prob', 'predicted_class', 'ig_convergence_delta'
        ]).to_csv(pred_path, index=False)

    for utt_id, label in tqdm(selected_samples, desc="Phase 2 XAI"):
        # --- Resume check: skip samples already fully processed ---
        if _sample_is_complete(cfg, utt_id):
            skipped += 1
            continue

        try:
            audio = load_waveform(cfg, utt_id)
        except Exception as e:
            logger.warning(f"Skipping {utt_id}: failed to load audio ({e})")
            failed.append(utt_id)
            continue

        sample_dir = os.path.join(cfg.out_dir, "attributions", utt_id)
        os.makedirs(sample_dir, exist_ok=True)
        sample_records = []

        # --- Save model prediction for this sample ---
        x_t = torch.tensor(audio, dtype=torch.float32, device=cfg.device).unsqueeze(0)
        with torch.no_grad():
            raw_out = model(x_t)
            probs = torch.softmax(raw_out, dim=1)
        pred_record = dict(
            utt_id=utt_id,
            label=label,
            bonafide_logit=raw_out[0, 1].item(),
            spoof_logit=raw_out[0, 0].item(),
            bonafide_prob=probs[0, 1].item(),
            spoof_prob=probs[0, 0].item(),
            predicted_class=int(raw_out.argmax(dim=1).item()),
        )

        # --- Occlusion, multiple window sizes ---
        for w_ms in cfg.occlusion_window_sizes_ms:
            attr = occlusion_attribution(model, cfg, audio, w_ms)
            np.save(os.path.join(sample_dir, f"occlusion_{w_ms}ms.npy"), attr)

            for frac in cfg.faithfulness_topk_fracs:
                comp, suff = comprehensiveness_sufficiency(model, cfg, audio, attr, frac)
                sample_records.append(dict(
                    utt_id=utt_id, label=label, method="occlusion",
                    window_ms=w_ms, topk_frac=frac,
                    comprehensiveness=comp, sufficiency=suff,
                ))

        # --- Integrated Gradients ---
        with torch.enable_grad():
            attr_ig, ig_delta = integrated_gradients_attribution(
                model,
                cfg,
                audio,
            )
        np.save(os.path.join(sample_dir, "integrated_gradients.npy"), attr_ig)
        pred_record["ig_convergence_delta"] = ig_delta

        pd.DataFrame([pred_record]).to_csv(
            pred_path,
            mode="a",
            header=False,
            index=False,
        )

        for frac in cfg.faithfulness_topk_fracs:
            comp, suff = comprehensiveness_sufficiency(model, cfg, audio, attr_ig, frac)
            sample_records.append(dict(
                utt_id=utt_id, label=label, method="integrated_gradients",
                window_ms=None, topk_frac=frac,
                comprehensiveness=comp, sufficiency=suff,
            ))

        # --- Append this sample's faithfulness rows to CSV immediately ---
        pd.DataFrame.from_records(sample_records).to_csv(
            csv_path, mode='a', header=False, index=False
        )
        all_records.extend(sample_records)

    if skipped:
        logger.info(f"Skipped {skipped} already-complete samples (resume mode)")
    if failed:
        logger.warning(f"{len(failed)} utterances failed to load and were skipped: {failed}")

    return pd.DataFrame.from_records(all_records)


faithfulness_df = run_phase2(model, CFG, selected_samples)
logger.info(f"Total faithfulness rows: {len(faithfulness_df)}")
faithfulness_df.head(10)
```

## Aggregate faithfulness results


```python
summary = (
    faithfulness_df
    .groupby(["method", "window_ms", "topk_frac"], dropna=False)[["comprehensiveness", "sufficiency"]]
    .agg(["mean", "std"])
)
summary.to_csv(os.path.join(CFG.out_dir, "faithfulness_summary.csv"))
print("\n--- Faithfulness Summary (ASVspoof 2019 LA Eval, 100 samples) ---\n")
display(summary) if hasattr(__builtins__, '__IPYTHON__') else print(summary)
```

## Faithfulness bar chart

Publication-ready grouped bar chart: comprehensiveness and sufficiency across methods
and top-K fractions.


```python
def plot_faithfulness_bars(cfg: Config, df: pd.DataFrame):
    """Grouped bar chart of comprehensiveness/sufficiency by method and top-K."""
    # Use IG + occlusion-50ms only — keeps the chart clean and avoids
    # silently averaging across all three occlusion window sizes.
    ig_rows = df[df["method"] == "integrated_gradients"]
    occ50_rows = df[(df["method"] == "occlusion") & (df["window_ms"] == 50)]
    plot_df = pd.concat([ig_rows, occ50_rows])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in zip(
        axes,
        ["comprehensiveness", "sufficiency"],
        ["Comprehensiveness (\u2191 = more faithful)", "Sufficiency (\u2191 = more faithful)"],
    ):
        pivot = (
            plot_df.groupby(["method", "topk_frac"])[metric]
            .mean()
            .unstack("topk_frac")
        )
        pivot.index = [m.replace("_", " ").title() for m in pivot.index]
        pivot.columns = [f"Top {int(c*100)}%" for c in pivot.columns]

        pivot.plot(kind="bar", ax=ax, rot=0, edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.legend(title="Top-K", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(cfg.out_dir, "figures", "faithfulness_bars.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    logger.info(f"Saved faithfulness bar chart: {path}")


plot_faithfulness_bars(CFG, faithfulness_df)
```

## Example attribution plots

Publication-ready attribution figures: waveform, occlusion (50ms), and |Integrated
Gradients| stacked on a shared time axis. Uses `fill_between` for visual clarity.

Generates for the first 3 bonafide and first 3 spoof samples in the selection.


```python
def plot_attribution_example(cfg: Config, utt_id: str, label: int, audio: np.ndarray):
    """Three-panel attribution figure: waveform, occlusion, |IG|."""
    sample_dir = os.path.join(cfg.out_dir, "attributions", utt_id)
    label_str = "Bonafide" if label == 1 else "Spoofed"

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Attribution Analysis  |  {utt_id}  |  {label_str}",
        fontsize=15, fontweight="bold", y=0.98,
    )

    t = np.arange(len(audio)) / cfg.sample_rate

    # --- Panel 1: waveform ---
    axes[0].plot(t, audio, color="dimgray", linewidth=0.4, alpha=0.85)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Raw Waveform", fontsize=12)

    # --- Panel 2: occlusion (50ms) ---
    attr_occ = np.load(os.path.join(sample_dir, "occlusion_50ms.npy"))
    attr_occ_norm = attr_occ / (np.max(np.abs(attr_occ)) + 1e-10)
    axes[1].fill_between(t, attr_occ_norm, color="tab:orange", alpha=0.7)
    axes[1].set_ylabel("Importance (norm.)")
    axes[1].set_title("Occlusion Attribution (50ms window)", fontsize=12)

    # --- Panel 3: |Integrated Gradients| ---
    attr_ig = np.load(os.path.join(sample_dir, "integrated_gradients.npy"))
    attr_ig_abs = np.abs(attr_ig)
    attr_ig_norm = attr_ig_abs / (np.max(attr_ig_abs) + 1e-10)
    axes[2].fill_between(t, attr_ig_norm, color="royalblue", alpha=0.7)
    axes[2].set_ylabel("Importance (norm.)")
    axes[2].set_title("|Integrated Gradients| Attribution", fontsize=12)
    axes[2].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save both PDF (publication) and PNG (quick preview)
    pdf_path = os.path.join(cfg.out_dir, "figures", f"{utt_id}_attribution.pdf")
    png_path = os.path.join(cfg.out_dir, "figures", f"{utt_id}_attribution.png")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


# Plot first 3 bonafide + first 3 spoof from the selection
n_examples = 3
bonafide_plotted, spoof_plotted = 0, 0
for utt_id, label in selected_samples:
    if label == 1 and bonafide_plotted < n_examples:
        audio = load_waveform(CFG, utt_id)
        plot_attribution_example(CFG, utt_id, label, audio)
        bonafide_plotted += 1
    elif label == 0 and spoof_plotted < n_examples:
        audio = load_waveform(CFG, utt_id)
        plot_attribution_example(CFG, utt_id, label, audio)
        spoof_plotted += 1
    if bonafide_plotted >= n_examples and spoof_plotted >= n_examples:
        break
```

## Bonafide vs Spoof attribution comparison

Aggregate mean attribution magnitude by class — a quick sanity check that
attributions are actually class-discriminative.


```python
def plot_class_attribution_comparison(cfg: Config, selected_samples):
    """Compare mean |attribution| across bonafide vs spoof samples for IG and occlusion."""
    bonafide_ig, spoof_ig = [], []
    bonafide_occ, spoof_occ = [], []

    for utt_id, label in selected_samples:
        sample_dir = os.path.join(cfg.out_dir, "attributions", utt_id)
        ig_path = os.path.join(sample_dir, "integrated_gradients.npy")
        occ_path = os.path.join(sample_dir, "occlusion_50ms.npy")

        if not os.path.exists(ig_path) or not os.path.exists(occ_path):
            continue

        ig_attr = np.abs(np.load(ig_path))
        occ_attr = np.load(occ_path)

        if label == 1:
            bonafide_ig.append(ig_attr.mean())
            bonafide_occ.append(occ_attr.mean())
        else:
            spoof_ig.append(ig_attr.mean())
            spoof_occ.append(occ_attr.mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, bonafide_vals, spoof_vals, method_name in zip(
        axes,
        [bonafide_ig, bonafide_occ],
        [spoof_ig, spoof_occ],
        ["|Integrated Gradients|", "Occlusion (50ms)"],
    ):
        data = pd.DataFrame({
            "Mean Attribution": bonafide_vals + spoof_vals,
            "Class": ["Bonafide"] * len(bonafide_vals) + ["Spoof"] * len(spoof_vals),
        })
        sns.boxplot(data=data, x="Class", y="Mean Attribution", hue="Class",
                    palette={"Bonafide": "steelblue", "Spoof": "coral"},
                    width=0.5, legend=False, ax=ax)
        ax.set_title(f"{method_name}  \u2014  Mean Attribution by Class", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(cfg.out_dir, "figures", "class_attribution_comparison.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    logger.info(f"Saved class attribution comparison: {path}")


plot_class_attribution_comparison(CFG, selected_samples)
```

## Run manifest

Record the exact config used, for reproducibility and for comparing against Phase 5 (Hindi) later.


```python
manifest = dict(
    phase="Phase 2 \u2014 XAI on English (Replication)",
    timestamp=datetime.now().isoformat(),
    config=dataclasses.asdict(CFG),
    n_samples=len(selected_samples),
    n_faithfulness_rows=len(faithfulness_df),
    torch_version=torch.__version__,
    numpy_version=np.__version__,
    librosa_version=librosa.__version__,
    device_name=torch.cuda.get_device_name(0) if CFG.device == "cuda" else "cpu",
    device_vram_gb=(
        round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        if CFG.device == "cuda" else None
    ),
    source_verified="Aug 2026 — pad(), target_class, Model.forward() verified against repo source",
)
with open(os.path.join(CFG.out_dir, "run_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, default=str)

logger.info("Phase 2 complete.")
print(json.dumps(manifest, indent=2, default=str))
```

## Output inventory

Quick summary of everything this notebook produced — verify all expected files exist
before proceeding to Phase 3.


```python
def print_output_inventory(cfg: Config):
    out = Path(cfg.out_dir)
    print(f"\n{'='*60}")
    print(f"Phase 2 Output Inventory: {out.resolve()}")
    print(f"{'='*60}")

    # Top-level files
    top_files = [f for f in out.iterdir() if f.is_file()]
    print(f"\nTop-level files ({len(top_files)}):")
    for f in sorted(top_files):
        print(f"  {f.name:40s}  {f.stat().st_size / 1024:.1f} KB")

    # Attribution directories
    attr_dir = out / "attributions"
    if attr_dir.exists():
        sample_dirs = [d for d in attr_dir.iterdir() if d.is_dir()]
        n_npy = sum(1 for d in sample_dirs for f in d.glob("*.npy"))
        print(f"\nAttribution maps: {n_npy} .npy files across {len(sample_dirs)} samples")
        # Spot-check one sample
        if sample_dirs:
            example = sorted(sample_dirs)[0]
            print(f"  Example ({example.name}/):")
            for f in sorted(example.glob("*.npy")):
                arr = np.load(f)
                print(f"    {f.name:35s}  shape={arr.shape}  dtype={arr.dtype}")

    # Figures
    fig_dir = out / "figures"
    if fig_dir.exists():
        figs = list(fig_dir.iterdir())
        print(f"\nFigures ({len(figs)}):")
        for f in sorted(figs):
            print(f"  {f.name:40s}  {f.stat().st_size / 1024:.1f} KB")

    print(f"\n{'='*60}")
    print("\u2705 Phase 2 outputs ready for Phase 3 (shortcut analysis)")
    print("\u2705 Attribution .npy files ready for Phase 5 (Hindi comparison)")
    print("\u2705 predictions.csv ready for downstream analysis")


print_output_inventory(CFG)
```
d(figs):
            print(f"  {f.name:40s}  {f.stat().st_size / 1024:.1f} KB")

    print(f"\n{'='*60}")
    print("\u2705 Phase 2 outputs ready for Phase 3 (shortcut analysis)")
    print("\u2705 Attribution .npy files ready for Phase 5 (Hindi comparison)")
    print("\u2705 predictions.csv ready for downstream analysis")


print_output_inventory(CFG)
```
