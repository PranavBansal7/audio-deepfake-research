#!/usr/bin/env python3
"""Final Phase 5: controlled English/Hindi Griffin-Lim evaluation + XAI.

Primary experiment
------------------
CommonVoice English Griffin-Lim vs CommonVoice Hindi Griffin-Lim.

Secondary references
--------------------
* Phase-1 ASVspoof 2019 LA EER (from phase1_outputs/eer_summary.json if present,
  otherwise recomputed from the Phase-1 score file eval_CM_scores_2019_LA_eval.txt).
* Sampled ASVspoof A11 Griffin-Lim XAI reference.

The finalized Hindi Phase-4 dataset is NOT regenerated here.

Run from the repository root (the same directory used for Phases 1-4) so that
model.py / data_utils_SSL.py / eval_metric_LA.py and the xlsr2_300m.pt path used
by model.SSLModel resolve exactly as in the earlier phases.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, Sequence, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import torch
from captum.attr import IntegratedGradients
from scipy import stats
from tqdm.auto import tqdm

from model import Model
from data_utils_SSL import pad
from eval_metric_LA import compute_eer


@dataclasses.dataclass(frozen=True)
class Config:
    checkpoint_path: str = "pretrained_models/best_SSL_model_LA.pth"
    hindi_dir: str = "hindi_griffinlim_eval_final"
    english_cv_dir: str = "english_griffinlim_eval_final"
    phase1_out_dir: str = "phase1_outputs"
    # Phase-1 wrote utterance scores here (see run_eval_2019LA.py); used as the
    # fallback source for the ASVspoof 2019 LA reference EER when
    # phase1_outputs/eer_summary.json does not exist.
    phase1_scores_path: str = "eval_CM_scores_2019_LA_eval.txt"
    # Same layout convention as phase2_xai_english_clean.py / phase3_shortcut.py.
    en_protocol_path: str = "database/ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    en_database_path: str = "database/LA/ASVspoof2019_LA_eval"
    out_dir: str = "phase5_outputs"

    sample_rate: int = 16000
    fixed_len: int = 64600

    eval_batch_size: int = 64
    num_workers: int = 4
    n_eer_bootstrap: int = 2000
    n_bootstrap: int = 2000
    ci: float = 0.95

    n_xai_pairs_per_language: int = 100
    n_english_a11: int = 100
    n_english_bonafide_matched: int = 100
    seed: int = 1234

    occlusion_windows_ms: Tuple[int, ...] = (20, 50, 100)
    occlusion_step_frac: float = 0.5
    occlusion_batch_size: int = 128

    ig_n_steps: int = 50
    ig_internal_batch_size: int = 50
    target_class: int = 1
    faithfulness_fracs: Tuple[float, ...] = (0.05, 0.10, 0.20)

    stft_n_fft: int = 512
    stft_hop: int = 160
    freq_bands: Tuple[Tuple[str, Tuple[int, int]], ...] = (
        ("low", (0, 1000)),
        ("mid", (1000, 4000)),
        ("high", (4000, 8000)),
    )

    stability_frac: float = 0.10
    silence_top_db: float = 40.0
    silence_frame_length: int = 2048
    silence_hop_length: int = 512
    dpi: int = 300
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()
OUT = Path(CFG.out_dir)


def setup_logging() -> logging.Logger:
    OUT.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase5")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(OUT / "phase5_eval.log"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


LOGGER = setup_logging()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def cfg_hash() -> str:
    raw = json.dumps(dataclasses.asdict(CFG), sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def ensure_manifest() -> None:
    path = OUT / "run_manifest.json"
    payload = {"config": dataclasses.asdict(CFG), "config_sha256": cfg_hash(), "script": "phase5_hindi_eval.py"}
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("config_sha256") != payload["config_sha256"]:
            raise RuntimeError("phase5_outputs contains a different configuration; use a fresh output directory.")
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model() -> torch.nn.Module:
    args = type("Args", (), {})()  # Model() takes an args param but never reads it (see run_eval_2019LA.py)
    model = Model(args, CFG.device).to(CFG.device)
    state = torch.load(CFG.checkpoint_path, map_location=CFG.device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    dummy = torch.zeros(1, CFG.fixed_len, device=CFG.device)
    with torch.no_grad(): out = model(dummy)
    if tuple(out.shape) != (1, 2):
        raise RuntimeError(f"Unexpected model output shape: {tuple(out.shape)}")
    LOGGER.info("Model output verified as [spoof_logit, bonafide_logit] (index 1 = bonafide, per Phase 1/2 verification).")
    return model


def raw_audio(path: str) -> np.ndarray:
    x, _ = librosa.load(path, sr=CFG.sample_rate, mono=True)
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all(): raise ValueError(f"Non-finite audio: {path}")
    return x


def model_input(path: str) -> np.ndarray:
    x = raw_audio(path)
    y = np.asarray(pad(x, CFG.fixed_len), dtype=np.float32)
    if len(y) != CFG.fixed_len:
        raise RuntimeError(f"pad() returned {len(y)} for {path}")
    return y


def score(model: torch.nn.Module, audio: np.ndarray) -> float:
    x = torch.tensor(audio, dtype=torch.float32, device=CFG.device).unsqueeze(0)
    with torch.no_grad(): out = model(x)
    return float(out[0, CFG.target_class].item())


def content_fraction(duration_s: float) -> float:
    return float(min(1.0, duration_s / (CFG.fixed_len / CFG.sample_rate)))


def load_meta(directory: str, name: str) -> pd.DataFrame:
    path = Path(directory) / "metadata.csv"
    if not path.exists(): raise FileNotFoundError(f"{name} metadata not found: {path}")
    df = pd.read_csv(path)
    required = {"pair_id", "real_path", "fake_path", "duration_s"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"{name} metadata missing {sorted(missing)}")
    if df.pair_id.duplicated().any(): raise ValueError(f"{name}: duplicate pair_id")
    if "client_id" not in df.columns: df["client_id"] = np.nan
    df["pair_id"] = df["pair_id"].astype(str)
    valid_client = df["client_id"].notna() & (df["client_id"].astype(str).str.strip() != "") & (df["client_id"].astype(str).str.lower() != "nan")
    df["analysis_cluster"] = np.where(valid_client, df["client_id"].astype(str), "pair:" + df["pair_id"])
    if "inferred_normalization" not in df.columns: df["inferred_normalization"] = "unknown"
    return df


def audit_metadata(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Verify every file on disk and record per-file amplitude/duration QC.

    Uses a single soundfile decode per file (the datasets are 16 kHz mono WAV),
    which is substantially faster than the sf.info + librosa.load double decode.
    """
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Auditing {name}"):
        for role, p in (("real", str(r.real_path)), ("fake", str(r.fake_path))):
            if not Path(p).exists(): raise FileNotFoundError(f"{name}: missing {p}")
            x, sr = sf.read(p, dtype="float32")
            if sr != CFG.sample_rate: raise ValueError(f"{name}: sample rate mismatch {p} ({sr} != {CFG.sample_rate})")
            if x.ndim != 1: raise ValueError(f"{name}: expected mono audio, got shape {x.shape} for {p}")
            if not np.isfinite(x).all(): raise ValueError(f"{name}: non-finite samples in {p}")
            rows.append({
                "pair_id": str(r.pair_id), "cluster": str(r.analysis_cluster), "role": role,
                "path": p, "duration_s": len(x) / CFG.sample_rate,
                "rms": float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12)),
                "peak": float(np.max(np.abs(x))) if len(x) else 0.0,
                "content_fraction": content_fraction(float(r.duration_s)),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"{name}_audio_audit.csv", index=False)
    LOGGER.info("%s: %d pairs, %d independent clusters, mean content fraction %.3f", name, len(df), df.analysis_cluster.nunique(), out.content_fraction.mean())
    return out


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, meta: pd.DataFrame):
        self.rows = []
        for _, r in meta.iterrows():
            base = {"pair_id": str(r.pair_id), "cluster": str(r.analysis_cluster), "duration_s": float(r.duration_s)}
            self.rows += [{**base, "path": str(r.real_path), "label": 1}, {**base, "path": str(r.fake_path), "label": 0}]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        return (torch.tensor(model_input(r["path"])), int(r["label"]), r["pair_id"], r["cluster"], r["duration_s"], r["path"])


def score_dataset(model: torch.nn.Module, meta: pd.DataFrame, tag: str) -> pd.DataFrame:
    path = OUT / f"{tag}_eer_scores.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(old.path.astype(str)) if not old.empty and "path" in old.columns else set()
    ds = PairDataset(meta)
    keep = [i for i, r in enumerate(ds.rows) if r["path"] not in done]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, keep), batch_size=CFG.eval_batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=(CFG.device == "cuda"))
    rows = old.to_dict("records") if not old.empty else []
    for audio, labels, pairs, clusters, durations, paths in tqdm(loader, desc=f"Scoring {tag}"):
        audio = audio.to(CFG.device, non_blocking=(CFG.device == "cuda"))
        with torch.no_grad(): logits = model(audio)
        for i in range(len(paths)):
            rows.append({
                "pair_id": str(pairs[i]), "analysis_cluster": str(clusters[i]),
                "path": str(paths[i]), "label": int(labels[i]), "duration_s": float(durations[i]),
                "bonafide_logit": float(logits[i, 1]), "spoof_logit": float(logits[i, 0]),
            })
    out = pd.DataFrame(rows).drop_duplicates("path", keep="last").reset_index(drop=True)
    out.to_csv(path, index=False)
    if len(out) != 2 * len(meta): raise RuntimeError(f"{tag}: expected {2 * len(meta)} scores, got {len(out)}")
    return out


def _concat_class(groups: dict, draw: Sequence[str], label: int) -> np.ndarray:
    parts = [groups[s][label] for s in draw if len(groups[s][label])]
    return np.concatenate(parts) if parts else np.empty(0, dtype=float)


def bootstrap_eer(df: pd.DataFrame, offset: int = 0) -> dict:
    groups = {str(s): {1: g[g.label == 1].bonafide_logit.to_numpy(), 0: g[g.label == 0].bonafide_logit.to_numpy()} for s, g in df.groupby("analysis_cluster")}
    clusters = list(groups)
    b0 = df[df.label == 1].bonafide_logit.to_numpy(); s0 = df[df.label == 0].bonafide_logit.to_numpy()
    observed, threshold = compute_eer(b0, s0)
    rng = np.random.default_rng(CFG.seed + offset); vals = []
    for _ in range(CFG.n_eer_bootstrap):
        draw = rng.choice(clusters, len(clusters), replace=True)
        b = _concat_class(groups, draw, 1); sp = _concat_class(groups, draw, 0)
        if b.size == 0 or sp.size == 0: continue
        vals.append(compute_eer(b, sp)[0])
    if not vals: raise RuntimeError("EER bootstrap produced no valid draws (check class balance per cluster).")
    a = (1 - CFG.ci) / 2
    unit = "speaker" if all(not str(c).startswith("pair:") for c in clusters) else "pair_fallback"
    return {"eer_pct": observed * 100, "threshold": float(threshold), "bootstrap_mean_eer_pct": float(np.mean(vals) * 100), "ci_lo_pct": float(np.quantile(vals, a) * 100), "ci_hi_pct": float(np.quantile(vals, 1 - a) * 100), "n_utterances": int(len(df)), "n_clusters": int(len(clusters)), "bootstrap_unit": unit}


def bootstrap_eer_diff(en: pd.DataFrame, hi: pd.DataFrame) -> dict:
    def groups(df):
        return {str(s): {1: g[g.label == 1].bonafide_logit.to_numpy(), 0: g[g.label == 0].bonafide_logit.to_numpy()} for s, g in df.groupby("analysis_cluster")}
    eg, hg = groups(en), groups(hi); es, hs = list(eg), list(hg); rng = np.random.default_rng(CFG.seed + 500); diffs = []
    en_obs = compute_eer(en[en.label == 1].bonafide_logit.to_numpy(), en[en.label == 0].bonafide_logit.to_numpy())[0]
    hi_obs = compute_eer(hi[hi.label == 1].bonafide_logit.to_numpy(), hi[hi.label == 0].bonafide_logit.to_numpy())[0]
    for _ in range(CFG.n_eer_bootstrap):
        ed = rng.choice(es, len(es), replace=True); hd = rng.choice(hs, len(hs), replace=True)
        eb = _concat_class(eg, ed, 1); ef = _concat_class(eg, ed, 0)
        hb = _concat_class(hg, hd, 1); hf = _concat_class(hg, hd, 0)
        if min(eb.size, ef.size, hb.size, hf.size) == 0: continue
        diffs.append(compute_eer(hb, hf)[0] - compute_eer(eb, ef)[0])
    if not diffs: raise RuntimeError("EER-difference bootstrap produced no valid draws.")
    a = (1 - CFG.ci) / 2
    return {"observed_hindi_minus_english_pct_points": float((hi_obs - en_obs) * 100), "ci_lo_pct_points": float(np.quantile(diffs, a) * 100), "ci_hi_pct_points": float(np.quantile(diffs, 1 - a) * 100)}


def parse_asvspoof_protocol(path: str) -> pd.DataFrame:
    """Parse the standard 5-column ASVspoof LA protocol format:

        SPEAKER_ID  UTT_ID  -  SYSTEM_ID  LABEL

    (bonafide lines carry "-" in the SYSTEM_ID column). This is the same format
    that genSpoof_list(..., is_train=False, is_eval=False) parses in
    data_utils_SSL.py and that Phase 2/3 used.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"ASVspoof LA eval protocol not found: {p}. Expected the layout used by "
            f"Phase 2/3 (database/ASVspoof_LA_cm_protocols/...). Edit Config.en_protocol_path "
            f"if your local layout differs.")
    rows = []
    with p.open("r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"{p}:{ln}: expected 5 space-delimited columns, got {len(parts)}: {line.strip()!r}")
            spk, utt, _dash, system, label = parts
            rows.append({"speaker_id": spk, "utt_id": utt, "system_id": system, "label_name": label})
    df = pd.DataFrame(rows)
    bad = set(df.label_name.unique()) - {"bonafide", "spoof"}
    if bad:
        raise ValueError(f"Unexpected labels in {p}: {sorted(bad)}")
    return df


def phase1_reference_eer() -> dict:
    """Phase-1 ASVspoof 2019 LA EER as a secondary benchmark reference.

    run_eval_2019LA.py prints the EER and writes per-utterance scores to
    eval_CM_scores_2019_LA_eval.txt; it does not write phase1_outputs/
    eer_summary.json. Prefer the summary JSON if the user created one;
    otherwise recompute the EER from the score file + the eval protocol.
    """
    summary = Path(CFG.phase1_out_dir) / "eer_summary.json"
    if summary.exists():
        return {"eer_pct": json.loads(summary.read_text(encoding="utf-8")).get("eer_pct"), "source": str(summary)}
    scores_path = Path(CFG.phase1_scores_path)
    if not scores_path.exists():
        LOGGER.warning("Phase-1 reference unavailable: neither %s nor %s exists; skipping the ASVspoof 2019 LA reference row.", summary, scores_path)
        return {}
    proto = parse_asvspoof_protocol(CFG.en_protocol_path)
    label_by_utt = dict(zip(proto.utt_id, proto.label_name))
    bona, spoof = [], []
    with scores_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            utt, s = parts
            lab = label_by_utt.get(utt)
            if lab == "bonafide": bona.append(float(s))
            elif lab == "spoof": spoof.append(float(s))
    if not bona or not spoof:
        LOGGER.warning("Phase-1 score file %s matched no protocol utterances; skipping the ASVspoof reference.", scores_path)
        return {}
    eer = compute_eer(np.asarray(bona), np.asarray(spoof))[0]
    return {"eer_pct": float(eer * 100), "source": str(scores_path), "note": "recomputed from the Phase-1 score file", "n_bonafide": len(bona), "n_spoof": len(spoof)}


# ------------------------------ XAI ----------------------------------------

def occlusion(model, audio: np.ndarray, window_ms: int) -> np.ndarray:
    window = int(CFG.sample_rate * window_ms / 1000); step = max(1, int(window * CFG.occlusion_step_frac))
    starts = list(range(0, len(audio), step)); x = torch.tensor(audio, dtype=torch.float32, device=CFG.device).unsqueeze(0)
    with torch.no_grad(): base = model(x)[0, CFG.target_class].item()
    s = np.zeros(len(audio), dtype=np.float64); c = np.zeros(len(audio), dtype=np.float64)
    for j in range(0, len(starts), CFG.occlusion_batch_size):
        pos = starts[j:j + CFG.occlusion_batch_size]; batch = x.repeat(len(pos), 1)
        for i, st in enumerate(pos): batch[i, st:min(st + window, len(audio))] = 0.0
        with torch.no_grad(): scores = model(batch)[:, CFG.target_class].cpu().numpy()
        for i, st in enumerate(pos):
            en = min(st + window, len(audio)); s[st:en] += abs(base - float(scores[i])); c[st:en] += 1
    out = np.zeros_like(s, dtype=np.float32); mask = c > 0; out[mask] = (s[mask] / c[mask]).astype(np.float32); return out


def integrated_gradients(model, audio: np.ndarray) -> Tuple[np.ndarray, float]:
    ig = IntegratedGradients(lambda x: model(x))
    x = torch.tensor(audio, dtype=torch.float32, device=CFG.device).unsqueeze(0); x.requires_grad_(True); baseline = torch.zeros_like(x)
    attr, delta = ig.attribute(x, baselines=baseline, target=CFG.target_class, n_steps=CFG.ig_n_steps, internal_batch_size=CFG.ig_internal_batch_size, return_convergence_delta=True)
    arr = attr.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if arr.shape != (CFG.fixed_len,): raise RuntimeError(f"IG shape {arr.shape} != {(CFG.fixed_len,)}")
    return arr, float(delta.item())


def faithfulness(model, audio: np.ndarray, attr: np.ndarray, frac: float) -> Tuple[float, float]:
    base = score(model, audio); k = max(1, int(len(audio) * frac)); idx = np.argsort(-np.abs(attr))[:k]
    comp = audio.copy(); comp[idx] = 0.0; comp_score = score(model, comp)
    kept = np.zeros_like(audio); kept[idx] = audio[idx]; suff_score = score(model, kept)
    return float(base - comp_score), float(suff_score)


def select_xai_pairs(scores: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    pairs = scores.pair_id.drop_duplicates().sample(n=min(n, scores.pair_id.nunique()), random_state=seed)
    out = scores[scores.pair_id.isin(pairs)].copy()
    if set(out[out.label == 1].pair_id) != set(out[out.label == 0].pair_id): raise RuntimeError("Real/fake XAI selection lost pairing")
    return out.sort_values(["pair_id", "label"]).reset_index(drop=True)


def run_xai(model, samples: pd.DataFrame, tag: str) -> pd.DataFrame:
    adir = OUT / f"attributions_{tag}"; adir.mkdir(parents=True, exist_ok=True); csv = OUT / f"faithfulness_{tag}.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []; done = {str(r["sample_key"]) for r in rows}
    for _, r in tqdm(samples.iterrows(), total=len(samples), desc=f"XAI {tag}"):
        key = f"{r.pair_id}_{Path(str(r.path)).stem}"; d = adir / key; d.mkdir(parents=True, exist_ok=True)
        required = [*(f"occlusion_{w}ms.npy" for w in CFG.occlusion_windows_ms), "integrated_gradients.npy", "xai_qc.json"]
        if key in done and all((d / f).exists() for f in required): continue
        audio = model_input(str(r.path)); attr, delta = integrated_gradients(model, audio); np.save(d / "integrated_gradients.npy", attr)
        for w in CFG.occlusion_windows_ms: np.save(d / f"occlusion_{w}ms.npy", occlusion(model, audio, w))
        rec = {"sample_key": key, "pair_id": str(r.pair_id), "analysis_cluster": str(r.get("analysis_cluster", r.get("client_id", r.pair_id))), "label": int(r.label), "path": str(r.path), "ig_delta": delta}
        for frac in CFG.faithfulness_fracs:
            c, s = faithfulness(model, audio, attr, frac); pct = int(frac * 100); rec[f"ig_comp_{pct}"] = c; rec[f"ig_suff_{pct}"] = s
        qc = {"ig_delta": delta, "ig_abs_sum": float(np.abs(attr).sum()), "ig_l2": float(np.linalg.norm(attr))}; (d / "xai_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
        rows = [x for x in rows if str(x.get("sample_key")) != key]; rows.append(rec); pd.DataFrame(rows).to_csv(csv, index=False)
    return pd.read_csv(csv)


def freq_proxy(audio: np.ndarray, attr: np.ndarray) -> dict:
    S = np.abs(librosa.stft(audio, n_fft=CFG.stft_n_fft, hop_length=CFG.stft_hop, center=False)); freqs = librosa.fft_frequencies(sr=CFG.sample_rate, n_fft=CFG.stft_n_fft); n = S.shape[1]
    fa = np.zeros(n)
    for t in range(n):
        st = t * CFG.stft_hop; en = min(st + CFG.stft_n_fft, len(attr)); fa[t] = np.mean(np.abs(attr[st:en])) if st < len(attr) else 0.0
    totalE = np.sum(S * S, axis=0) + 1e-12; vals = {}; total = 0.0
    for name, (lo, hi) in CFG.freq_bands:
        mask = (freqs >= lo) & (freqs < hi); v = float(np.sum(fa * (np.sum(S[mask] * S[mask], axis=0) / totalE))); vals[name] = v; total += v
    if total <= 0: return {"low_frac": np.nan, "mid_frac": np.nan, "high_frac": np.nan}
    return {f"{k}_frac": v / total for k, v in vals.items()}


def speech_silence(raw: np.ndarray, attr: np.ndarray) -> dict:
    intervals = librosa.effects.split(raw, top_db=CFG.silence_top_db, frame_length=CFG.silence_frame_length, hop_length=CFG.silence_hop_length)
    mask = np.zeros(len(raw), dtype=bool)
    for st, en in intervals: mask[st:en] = True
    if len(attr) > len(mask): mask = np.tile(mask, int(math.ceil(len(attr) / len(mask))))[:len(attr)]
    else: mask = mask[:len(attr)]
    a = np.abs(attr); total = a.sum()
    if total <= 1e-12: return {"speech_frac": np.nan, "silence_frac": np.nan, "silence_duration_frac": float(np.mean(~mask))}
    return {"speech_frac": float(a[mask].sum() / total), "silence_frac": float(a[~mask].sum() / total), "silence_duration_frac": float(np.mean(~mask))}


def temporal(attr: np.ndarray) -> dict:
    a = np.abs(attr).astype(float); total = a.sum()
    if total <= 1e-12: return {"gini": np.nan, "entropy": np.nan, "topk_1pct": np.nan, "topk_5pct": np.nan, "topk_10pct": np.nan}
    s = np.sort(a); n = len(s); idx = np.arange(1, n + 1); g = (2 * np.sum(idx * s) / (n * total)) - (n + 1) / n; p = a[a > 0] / total; ent = float(-np.sum(p * np.log2(p)))
    return {"gini": float(g), "entropy": ent, "topk_1pct": float(np.sum(s[-max(1, int(.01 * n)):]) / total), "topk_5pct": float(np.sum(s[-max(1, int(.05 * n)):]) / total), "topk_10pct": float(np.sum(s[-max(1, int(.10 * n)):]) / total)}


def extract_features(faith: pd.DataFrame, tag: str) -> pd.DataFrame:
    adir = OUT / f"attributions_{tag}"; rows = []
    for _, r in tqdm(faith.iterrows(), total=len(faith), desc=f"Features {tag}"):
        attr = np.load(adir / str(r.sample_key) / "integrated_gradients.npy"); raw = raw_audio(str(r.path)); padded = model_input(str(r.path))
        rec = {"sample_key": str(r.sample_key), "pair_id": str(r.pair_id), "analysis_cluster": str(r.analysis_cluster), "label": int(r.label), "path": str(r.path), "duration_s": len(raw) / CFG.sample_rate, "content_fraction": content_fraction(len(raw) / CFG.sample_rate), "ig_delta": float(r.ig_delta)}
        rec.update(freq_proxy(padded, attr)); rec.update(speech_silence(raw, attr)); rec.update(temporal(attr)); rows.append(rec)
    out = pd.DataFrame(rows); out.to_csv(OUT / f"explanation_features_{tag}.csv", index=False); return out


def bh(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float); out = np.full(len(p), np.nan); m = np.isfinite(p); vals = p[m]
    if len(vals) == 0: return out
    order = np.argsort(vals); ranked = vals[order]; adj = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]; q = np.empty_like(adj); q[order] = np.clip(adj, 0, 1); out[m] = q; return out


def cluster_bootstrap_mean_diff(a: pd.DataFrame, b: pd.DataFrame, feature: str, offset: int = 0) -> dict:
    ga = {str(s): g[feature].dropna().to_numpy() for s, g in a.groupby("analysis_cluster") if g[feature].notna().any()}; gb = {str(s): g[feature].dropna().to_numpy() for s, g in b.groupby("analysis_cluster") if g[feature].notna().any()}
    sa, sb = list(ga), list(gb); observed = float(b[feature].mean() - a[feature].mean()); rng = np.random.default_rng(CFG.seed + offset); vals = []
    for _ in range(CFG.n_bootstrap):
        da = rng.choice(sa, len(sa), replace=True); db = rng.choice(sb, len(sb), replace=True)
        va = [ga[s] for s in da if len(ga[s])]; vb = [gb[s] for s in db if len(gb[s])]
        if not va or not vb: continue
        vals.append(float(np.concatenate(vb).mean() - np.concatenate(va).mean()))
    if not vals: raise RuntimeError(f"Cluster bootstrap produced no valid draws for feature {feature}.")
    alpha = (1 - CFG.ci) / 2; unit = "speaker" if all(not s.startswith("pair:") for s in sa + sb) else "pair_fallback"
    return {"observed": observed, "ci_lo": float(np.quantile(vals, alpha)), "ci_hi": float(np.quantile(vals, 1 - alpha)), "n_clusters_a": len(sa), "n_clusters_b": len(sb), "bootstrap_unit": unit}


def paired_bootstrap(feature_df: pd.DataFrame, feature: str, offset: int = 0) -> dict:
    wide = feature_df.pivot_table(index="pair_id", columns="label", values=feature, aggfunc="mean").dropna()
    diff = wide[0].to_numpy() - wide[1].to_numpy(); observed = float(diff.mean()); rng = np.random.default_rng(CFG.seed + offset); vals = []
    for _ in range(CFG.n_bootstrap):
        idx = rng.choice(len(diff), len(diff), replace=True); vals.append(float(diff[idx].mean()))
    alpha = (1 - CFG.ci) / 2
    return {"observed_fake_minus_real": observed, "ci_lo": float(np.quantile(vals, alpha)), "ci_hi": float(np.quantile(vals, 1 - alpha)), "n_pairs": len(diff)}


def compare_features(features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    independent = [("cross_language_fake", "cv_en_fake", "cv_hi_fake"), ("cross_language_real", "cv_en_real", "cv_hi_real")]
    paired = [("english_fake_minus_real", "cv_en_real", "cv_en_fake"), ("hindi_fake_minus_real", "cv_hi_real", "cv_hi_fake")]
    # All explanation features promised in README_PHASE5.md sec.10 are statistically
    # compared here, including the top-k temporal concentration metrics.
    names = ["low_frac", "mid_frac", "high_frac", "speech_frac", "silence_frac", "gini", "entropy", "topk_1pct", "topk_5pct", "topk_10pct"]
    rows = []
    for comp, akey, bkey in independent:
        a, b = features[akey], features[bkey]
        for feat in names:
            av, bv = a[feat].dropna(), b[feat].dropna()
            if len(av) < 2 or len(bv) < 2: continue
            boot = cluster_bootstrap_mean_diff(a, b, feat, len(rows) + 100); _, p = stats.mannwhitneyu(av, bv, alternative="two-sided"); pooled = math.sqrt((np.var(av, ddof=1) + np.var(bv, ddof=1)) / 2); d = boot["observed"] / pooled if pooled > 0 else np.nan
            rows.append({"comparison": comp, "feature": feat, "mean_a": float(av.mean()), "mean_b": float(bv.mean()), "mean_difference_b_minus_a": boot["observed"], "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"], "p_value": float(p), "effect_size": float(d), "test": "Mann-Whitney U + cluster bootstrap", "bootstrap_unit": boot["bootstrap_unit"]})
    for comp, real_key, fake_key in paired:
        merged = pd.concat([features[real_key], features[fake_key]], ignore_index=True)
        for feat in names:
            wide = merged.pivot_table(index="pair_id", columns="label", values=feat, aggfunc="mean").dropna()
            if len(wide) < 3: continue
            diff = wide[0].to_numpy() - wide[1].to_numpy();
            try: _, p = stats.wilcoxon(diff, alternative="two-sided")
            except ValueError: p = 1.0
            boot = paired_bootstrap(merged, feat, len(rows) + 1000); d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else np.nan
            rows.append({"comparison": comp, "feature": feat, "mean_a": float(wide[1].mean()), "mean_b": float(wide[0].mean()), "mean_difference_b_minus_a": boot["observed_fake_minus_real"], "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"], "p_value": float(p), "effect_size": float(d), "test": "Wilcoxon signed-rank + pair bootstrap", "bootstrap_unit": "pair"})
    out = pd.DataFrame(rows); out["q_value_bh"] = np.nan
    # Family-wise FDR is applied separately to each predefined comparison family.
    for comp in out.comparison.unique():
        m = out.comparison == comp; out.loc[m, "q_value_bh"] = bh(out.loc[m, "p_value"].to_numpy())
    out.to_csv(OUT / "explanation_drift_statistics.csv", index=False); return out


# --------------------- supplementary stability -----------------------------

def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b); return np.nan if d <= 1e-12 else float(np.dot(a, b) / d)


def spearman(a, b):
    r, _ = stats.spearmanr(np.abs(a), np.abs(b)); return float(r) if np.isfinite(r) else np.nan


def overlap(a, b):
    k = max(1, int(len(a) * CFG.stability_frac)); A = set(np.argsort(np.abs(a))[-k:]); B = set(np.argsort(np.abs(b))[-k:]); return float(len(A & B) / k)


def stability(a_dir: Path, b_dir: Path, label: str) -> dict:
    A = sorted([p.name for p in a_dir.iterdir() if p.is_dir()]); B = sorted([p.name for p in b_dir.iterdir() if p.is_dir()]); Battr = {k: np.load(b_dir / k / "integrated_gradients.npy") for k in B}; mats = [np.full((len(A), len(B)), np.nan, dtype=np.float32) for _ in range(3)]
    for i, k in enumerate(tqdm(A, desc=f"Stability {label}")):
        aa = np.load(a_dir / k / "integrated_gradients.npy")
        for j, q in enumerate(B): mats[0][i, j] = cosine(aa, Battr[q]); mats[1][i, j] = spearman(aa, Battr[q]); mats[2][i, j] = overlap(aa, Battr[q])
    res = {}; rng = np.random.default_rng(CFG.seed + 900)
    for name, m in zip(("cosine", "spearman", "topk_overlap"), mats):
        vals = []; nr, nc = m.shape
        for _ in range(CFG.n_bootstrap):
            ia = rng.choice(nr, nr, replace=True); ib = rng.choice(nc, nc, replace=True); vals.append(np.nanmean(m[np.ix_(ia, ib)]))
        alpha = (1 - CFG.ci) / 2; res[name] = {"mean": float(np.nanmean(m)), "ci_lo": float(np.quantile(vals, alpha)), "ci_hi": float(np.quantile(vals, 1 - alpha)), "n_rows": nr, "n_cols": nc}; np.save(OUT / f"stability_{name}_{label}.npy", m)
    return res


# ------------------------------ silence ------------------------------------

def silence_dataset(model, meta: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Leading/trailing-silence measurements + original/trimmed scores.

    Crash-resilient like the rest of this script: results are appended to CSV
    periodically and already-scored files are skipped on resume.
    """
    path = OUT / f"{tag}_silence_scores.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(old.path.astype(str)) if not old.empty and "path" in old.columns else set()
    rows = old.to_dict("records") if not old.empty else []
    pending = 0
    for _, r in tqdm(meta.iterrows(), total=len(meta), desc=f"Silence {tag}"):
        for label, p in ((1, str(r.real_path)), (0, str(r.fake_path))):
            if p in done:
                continue
            raw = raw_audio(p)
            trimmed, idx = librosa.effects.trim(raw, top_db=CFG.silence_top_db, frame_length=CFG.silence_frame_length, hop_length=CFG.silence_hop_length)
            if trimmed.size == 0:
                # Same fallback as phase3_shortcut.py: a clip that is entirely
                # silence under the top_db threshold cannot be trimmed (and an
                # empty waveform would crash the repo's pad()); keep the original.
                trimmed, idx = raw, np.array([0, len(raw)])
            lead = idx[0] / CFG.sample_rate * 1000; trail = (len(raw) - idx[1]) / CFG.sample_rate * 1000
            a = np.asarray(pad(raw, CFG.fixed_len), dtype=np.float32); t = np.asarray(pad(trimmed, CFG.fixed_len), dtype=np.float32)
            s0 = score(model, a); s1 = score(model, t)
            rows.append({"pair_id": str(r.pair_id), "analysis_cluster": str(r.analysis_cluster), "label": label, "path": p, "duration_s": len(raw) / CFG.sample_rate, "leading_silence_ms": lead, "trailing_silence_ms": trail, "score_original": s0, "score_trimmed": s1, "delta_score": s1 - s0})
            pending += 1
            if pending >= 200:
                pd.DataFrame(rows).to_csv(path, index=False); pending = 0
    out = pd.DataFrame(rows).drop_duplicates("path", keep="last").reset_index(drop=True)
    out.to_csv(path, index=False); return out


def silence_summary(df: pd.DataFrame) -> dict:
    res = {"config": {"top_db": CFG.silence_top_db, "frame_length": CFG.silence_frame_length, "hop_length": CFG.silence_hop_length}}
    for col in ("leading_silence_ms", "trailing_silence_ms"):
        r = df[df.label == 1].set_index("pair_id"); f = df[df.label == 0].set_index("pair_id"); common = r.index.intersection(f.index); rv = r.loc[common, col].to_numpy(); fv = f.loc[common, col].to_numpy(); e1 = compute_eer(rv, fv)[0]; e2 = compute_eer(-rv, -fv)[0]
        try: _, p = stats.wilcoxon(rv, fv)
        except ValueError: p = 1.0
        res[col] = {"silence_only_eer_pct": float(min(e1, e2) * 100), "mean_real_ms": float(rv.mean()), "mean_fake_ms": float(fv.mean()), "mean_fake_minus_real_ms": float((fv - rv).mean()), "wilcoxon_p": float(p), "n_pairs": len(common)}
    return res


# ------------------------------ figures ------------------------------------

def savefig(fig, name):
    d = OUT / "figures"; d.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(d / f"{name}.pdf", format="pdf", bbox_inches="tight"); fig.savefig(d / f"{name}.png", dpi=CFG.dpi, bbox_inches="tight"); plt.close(fig)


def make_plots(eer, features, se, sh):
    labels = ["English CV (GL)", "Hindi CV (GL)"]; vals = [eer["english_cv_gl"]["bootstrap_mean_eer_pct"], eer["hindi_gl"]["bootstrap_mean_eer_pct"]]; lo = [vals[0] - eer["english_cv_gl"]["ci_lo_pct"], vals[1] - eer["hindi_gl"]["ci_lo_pct"]]; hi = [eer["english_cv_gl"]["ci_hi_pct"] - vals[0], eer["hindi_gl"]["ci_hi_pct"] - vals[1]]
    if "asvspoof_2019_la" in eer: labels.append("ASVspoof 2019 LA"); vals.append(float(eer["asvspoof_2019_la"]["eer_pct"])); lo.append(0); hi.append(0)
    fig, ax = plt.subplots(figsize=(7.5, 5)); x = np.arange(len(vals)); ax.bar(x, vals, yerr=[lo, hi], capsize=4); ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("EER (%)"); ax.set_title("Controlled Griffin-Lim detection performance"); ax.grid(axis="y", alpha=.3); savefig(fig, "eer_comparison")
    bands = ["low_frac", "mid_frac", "high_frac"]; x = np.arange(3); w = .36; fig, ax = plt.subplots(figsize=(7.5, 5)); ax.bar(x - w / 2, [features["cv_en_fake"][b].mean() * 100 for b in bands], w, label="English"); ax.bar(x + w / 2, [features["cv_hi_fake"][b].mean() * 100 for b in bands], w, label="Hindi"); ax.set_xticks(x); ax.set_xticklabels(["0–1 kHz", "1–4 kHz", "4–8 kHz"]); ax.set_ylabel("Attribution proxy (%)"); ax.legend(); ax.set_title("Frequency-band attribution proxy"); savefig(fig, "frequency_band_attribution_proxy")
    cats = ["speech_frac", "silence_frac"]; x = np.arange(2); fig, ax = plt.subplots(figsize=(7, 5)); ax.bar(x - w / 2, [features["cv_en_fake"][c].mean() * 100 for c in cats], w, label="English"); ax.bar(x + w / 2, [features["cv_hi_fake"][c].mean() * 100 for c in cats], w, label="Hindi"); ax.set_xticks(x); ax.set_xticklabels(["Speech", "Silence"]); ax.set_ylabel("Attribution (%)"); ax.legend(); ax.set_title("Speech vs. silence attribution"); savefig(fig, "speech_silence_attribution")
    fig, ax = plt.subplots(figsize=(6.5, 5)); ax.violinplot([features["cv_en_fake"].gini.dropna(), features["cv_hi_fake"].gini.dropna()], showmeans=True); ax.set_xticks([1, 2]); ax.set_xticklabels(["English", "Hindi"]); ax.set_ylabel("Gini coefficient"); ax.set_title("Temporal concentration"); savefig(fig, "temporal_concentration")
    d = pd.concat([se.assign(Language="English"), sh.assign(Language="Hindi")]); d["Condition"] = d.label.map({1: "Real", 0: "Fake"}); fig, ax = plt.subplots(1, 2, figsize=(11, 4.5)); sns.boxplot(data=d, x="Language", y="leading_silence_ms", hue="Condition", ax=ax[0]); sns.boxplot(data=d, x="Language", y="trailing_silence_ms", hue="Condition", ax=ax[1]); ax[0].set_title("Leading silence"); ax[1].set_title("Trailing silence"); savefig(fig, "silence_distributions")


def main():
    seed_all(CFG.seed); ensure_manifest(); LOGGER.info("Starting Phase 5 on %s", CFG.device)
    model = load_model()
    h = load_meta(CFG.hindi_dir, "Hindi"); e = load_meta(CFG.english_cv_dir, "English CV")
    audit_metadata(h, "hindi"); audit_metadata(e, "english_cv")

    # Final Hindi QC: all 1,125 pairs are primary. Keep the 44 peak-limited pairs
    # and expose the 1,081 RMS-matched subset as a robustness condition.
    norm_counts = h["inferred_normalization"].fillna("unknown").value_counts().to_dict()
    rms_rel = np.abs(h["fake_rms"] - h["real_rms"]) / np.maximum(h["real_rms"], 1e-12) if {"fake_rms", "real_rms"}.issubset(h.columns) else pd.Series(dtype=float)
    qc_summary = {
        "n_pairs": int(len(h)),
        "normalization_counts": {str(k): int(v) for k, v in norm_counts.items()},
        "primary_includes_all_pairs": True,
        "rms_matched_pairs": int((h["inferred_normalization"] == "rms_matched").sum()),
        "peak_limited_pairs": int((h["inferred_normalization"] == "peak_normalized").sum()),
    }
    if not rms_rel.empty:
        qc_summary.update({
            "mean_relative_rms_error": float(rms_rel.mean()),
            "median_relative_rms_error": float(rms_rel.median()),
            "max_relative_rms_error": float(rms_rel.max()),
        })
    (OUT / "hindi_amplitude_qc_summary.json").write_text(json.dumps(qc_summary, indent=2), encoding="utf-8")

    hs = score_dataset(model, h, "hindi"); es = score_dataset(model, e, "english_cv")
    norm_map = h.set_index("pair_id")["inferred_normalization"].to_dict(); hs["inferred_normalization"] = hs.pair_id.map(norm_map).fillna("unknown"); hs["peak_limited"] = hs.inferred_normalization.eq("peak_normalized"); hs.to_csv(OUT / "hindi_eer_scores.csv", index=False)

    eer = {
        "english_cv_gl": bootstrap_eer(es, 1),
        "hindi_gl": bootstrap_eer(hs, 2),
        "language_difference": bootstrap_eer_diff(es, hs),
        # Pair-level counts from the Hindi metadata (NOT from the 2-rows-per-pair
        # score table, which would double-count).
        "hindi_amplitude_qc": {
            "rms_matched_pairs": int((h.inferred_normalization == "rms_matched").sum()),
            "peak_limited_pairs": int((h.inferred_normalization == "peak_normalized").sum()),
        },
    }
    clean_ids = set(h.loc[h.inferred_normalization == "rms_matched", "pair_id"].astype(str)); hs_clean = hs[hs.pair_id.isin(clean_ids)].copy()
    if len(clean_ids) >= 50: eer["hindi_rms_matched_sensitivity"] = bootstrap_eer(hs_clean, 3)
    p1 = phase1_reference_eer()
    if p1: eer["asvspoof_2019_la"] = p1
    (OUT / "eer_comparison.json").write_text(json.dumps(eer, indent=2), encoding="utf-8")

    # Pair-preserving XAI: 100 common pairs per language, i.e. 100 real + 100 fake.
    # Peak-limited Hindi pairs are retained in the primary XAI sample (the same
    # inclusion policy as the primary EER); how many were selected is recorded
    # below for the sensitivity discussion.
    ex = select_xai_pairs(es, CFG.n_xai_pairs_per_language, CFG.seed); hi = select_xai_pairs(hs, CFG.n_xai_pairs_per_language, CFG.seed)
    ex.to_csv(OUT / "xai_selected_english_cv.csv", index=False); hi.to_csv(OUT / "xai_selected_hindi_cv.csv", index=False)
    groups = {"cv_en_real": ex[ex.label == 1], "cv_en_fake": ex[ex.label == 0], "cv_hi_real": hi[hi.label == 1], "cv_hi_fake": hi[hi.label == 0]}
    faith = {g: run_xai(model, d, g) for g, d in groups.items()}
    features = {g: extract_features(f, g) for g, f in faith.items()}

    # XAI amplitude bookkeeping: count how many selected Hindi pairs come from the
    # peak-limited subset (pair-level count; the selection retains them by design).
    peak_limited_ids = set(h.loc[h.inferred_normalization == "peak_normalized", "pair_id"].astype(str))
    selected_peak_limited = sorted(peak_limited_ids & set(hi.pair_id.astype(str)))
    (OUT / "xai_hindi_peak_limited_selected.json").write_text(json.dumps({
        "selected_peak_limited_pair_count": len(selected_peak_limited),
        "selected_peak_limited_pair_ids": selected_peak_limited,
        "note": "Peak-limited pairs are retained in the primary XAI sample, matching the primary-EER inclusion policy; this file records the count for the sensitivity analysis.",
    }, indent=2), encoding="utf-8")

    # Secondary ASVspoof A11 XAI reference and sampled EER.
    proto = parse_asvspoof_protocol(CFG.en_protocol_path); a11 = proto[proto.system_id == "A11"].utt_id.tolist(); bona = proto[proto.label_name == "bonafide"].utt_id.tolist()
    if not a11: raise RuntimeError("No A11 samples found in the ASVspoof LA evaluation protocol.")
    rng = random.Random(CFG.seed + 42); rng.shuffle(a11); rng.shuffle(bona); a11 = a11[:CFG.n_english_a11]; bona = bona[:CFG.n_english_bonafide_matched]
    a11_rows = [{"pair_id": f"a11_{i:03d}", "analysis_cluster": "asvspoof_a11", "label": 0, "path": str(Path(CFG.en_database_path) / "flac" / f"{u}.flac")} for i, u in enumerate(a11)] + [{"pair_id": f"a11_bona_{i:03d}", "analysis_cluster": "asvspoof_bonafide", "label": 1, "path": str(Path(CFG.en_database_path) / "flac" / f"{u}.flac")} for i, u in enumerate(bona)]
    a11_df = pd.DataFrame(a11_rows); a11_df.to_csv(OUT / "xai_selected_asvspoof_a11.csv", index=False)
    for pth in a11_df.path:
        if not Path(pth).exists(): raise FileNotFoundError(f"ASVspoof audio not found: {pth}")
    faith["asvspoof_a11"] = run_xai(model, a11_df, "asvspoof_a11"); features["asvspoof_a11"] = extract_features(faith["asvspoof_a11"], "asvspoof_a11")
    a11_scores = []
    for _, r in tqdm(a11_df.iterrows(), total=len(a11_df), desc="Scoring sampled ASVspoof A11"):
        a11_scores.append({"label": int(r.label), "score": score(model, model_input(str(r.path)))})
    a11_scores = pd.DataFrame(a11_scores); a11_eer = compute_eer(a11_scores[a11_scores.label == 1].score.to_numpy(), a11_scores[a11_scores.label == 0].score.to_numpy())[0]
    (OUT / "asvspoof_a11_reference.json").write_text(json.dumps({"eer_pct": float(a11_eer * 100), "n_bonafide": len(bona), "n_a11": len(a11), "interpretation": "sampled secondary reference, not full ASVspoof-A11 EER"}, indent=2), encoding="utf-8")

    stats_table = compare_features(features); (OUT / "explanation_drift_summary.json").write_text(json.dumps(stats_table.to_dict(orient="records"), indent=2), encoding="utf-8")
    stab = {"fake_cross_language": stability(OUT / "attributions_cv_en_fake", OUT / "attributions_cv_hi_fake", "fake_cross_language"), "real_cross_language": stability(OUT / "attributions_cv_en_real", OUT / "attributions_cv_hi_real", "real_cross_language")}; (OUT / "stability_supplementary.json").write_text(json.dumps(stab, indent=2), encoding="utf-8")

    se = silence_dataset(model, e, "english_cv"); sh = silence_dataset(model, h, "hindi_cv"); silence = {"english_cv": silence_summary(se), "hindi_cv": silence_summary(sh)}; (OUT / "silence_comparison.json").write_text(json.dumps(silence, indent=2), encoding="utf-8")

    make_plots(eer, features, se, sh)
    inventory = {"english_pairs": len(e), "hindi_pairs": len(h), "english_clusters": e.analysis_cluster.nunique(), "hindi_clusters": h.analysis_cluster.nunique(), "hindi_client_id_available": bool((h.client_id.notna() & (h.client_id.astype(str).str.strip() != "")).any()), "hindi_rms_matched": int((h.inferred_normalization == "rms_matched").sum()), "hindi_peak_limited": int((h.inferred_normalization == "peak_normalized").sum()), "xai_pairs_per_language": CFG.n_xai_pairs_per_language, "a11_xai_samples": len(a11_df), "config_sha256": cfg_hash()}
    (OUT / "output_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    LOGGER.info("Phase 5 completed successfully: %s", OUT)


if __name__ == "__main__":
    main()
