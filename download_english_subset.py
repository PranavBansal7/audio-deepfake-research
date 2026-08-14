#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_english_subset.py — Phase 5 English source acquisition (streaming).

Common Voice moved exclusively to Mozilla Data Collective (MDC) in Oct 2025;
the Hugging Face mirrors are empty shells. The English bundle is ~88 GB as a
single .tar.gz, which cannot be partially fetched (gzip is not seekable).
This script therefore STREAMS the archive over HTTP and keeps only what the
pipeline needs — disk usage stays at ~2-4 GB regardless of the 88 GB transfer:

  PASS 1 (small):  stream the tarball only until validated.tsv (and
                   clip_durations.tsv, present since cv-corpus-13) have been
                   read; they sit at the archive root, before clips/.
  SELECT:          seed-42 shuffle, vote margin >= 2, <= 5 clips/speaker, and
                   — when clip_durations.tsv exists — the 2-8 s duration window
                   applied UP FRONT, so only ~pool-size clips are fetched
                   (slack 1.10). Without durations, slack 3.0 is used and
                   prepare_english_griffinlim.py probes durations locally.
  PASS 2 (stream): stream the tarball again, writing only the selected clip
                   members to clips/. Resumable: existing non-empty files are
                   skipped, and the pass stops early once everything is found.

Output is a Common-Voice-shaped folder:

  <out>/en/validated.tsv        <- ONLY the selected/downloaded rows
  <out>/en/clip_durations.tsv   <- subset, when the source bundle has one
  <out>/en/clips/*.mp3
  <out>/download_manifest.json  <- provenance: dataset id, params, failures
  <out>/downloaded_sources.csv  <- per-clip provenance

Point Config.cv_dir of prepare_english_griffinlim.py at <out>/en and run it
unchanged; it re-applies the seed-42 shuffle + vote/speaker/duration filters
over exactly the files present on disk, so selection is deterministic
end-to-end.

Prerequisites (one-time, in the browser):
  1. Create a free account at https://mozilladatacollective.com
  2. Open "Common Voice Scripted Speech 26.0 - English", ACCEPT THE TERMS,
     click "Connect API", and copy the Dataset ID.
  3. Generate an API key from your account dashboard.

Usage:
  export MDC_API_KEY=...        # or pass --api-key
  python download_english_subset.py --dataset-id <DATASET_ID> --out ./cv_english_subset

Tested target environment: Python 3.8, requests only.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
LOGGER = logging.getLogger("cv_en_subset")

# Canonical MDC API host (verified against the official datacollective-python
# SDK, api_utils.DEFAULT_API_URL). The old datacollective.mozillafoundation.org
# host redirects and can turn the POST into a GET -> HTTP 405.
MDC_API = os.environ.get("MDC_API_URL",
                         "https://mozilladatacollective.com/api").rstrip("/")


@dataclass
class Config:
    out: str = "./cv_english_subset"
    dataset_id: str = ""         # from the dataset page "Connect API" button
    api_key: str = ""            # or env MDC_API_KEY
    seed: int = 42               # == Hindi pipeline
    target_pairs: int = 1125     # == Hindi pipeline
    oversample_factor: float = 1.05
    min_vote_margin: int = 2
    max_clips_per_speaker: int = 5
    min_duration_s: float = 2.0
    max_duration_s: float = 8.0
    slack_with_durations: float = 1.10   # durations known pre-download
    slack_without_durations: float = 3.0  # durations probed post-download
    timeout_s: int = 120


CFG = Config()


# --------------------------------------------------------------------------- #
# MDC API
# --------------------------------------------------------------------------- #

def resolve_download_url(session: requests.Session) -> Tuple[str, dict]:
    """MDC download handshake."""
    headers = {
        "Authorization": f"Bearer {CFG.api_key}",
        "Content-Length": "0",
    }

    r = session.post(
        f"{MDC_API}/datasets/{CFG.dataset_id}/download",
        headers=headers,
        timeout=CFG.timeout_s,
        allow_redirects=False,
    )

    if r.is_redirect or r.is_permanent_redirect:
        raise RuntimeError(
            f"MDC redirected the download request "
            f"(HTTP {r.status_code} -> {r.headers.get('Location')})."
        )

    if r.status_code == 401:
        raise RuntimeError("MDC authentication failed (401).")

    if r.status_code == 403:
        raise RuntimeError(
            "MDC access denied (403): accept the dataset Terms & Conditions first."
        )

    if r.status_code == 404:
        raise RuntimeError(
            "Dataset not found (404): check the Dataset ID."
        )

    if r.status_code == 429:
        raise RuntimeError("MDC rate limit (429).")

    r.raise_for_status()

    payload = r.json()

    url = (
        payload.get("downloadUrl")
        or payload.get("download_url")
        or payload.get("url")
    )

    if not url:
        raise RuntimeError(
            f"Unexpected MDC download response: {payload!r}"
        )

    raw_size = payload.get("sizeBytes")

    try:
        size_bytes = int(raw_size) if raw_size is not None else None
    except (TypeError, ValueError):
        size_bytes = None

    meta = {
        "sizeBytes": size_bytes,
        "checksum": payload.get("checksum"),
    }

    if size_bytes is not None:
        LOGGER.info(
            "Archive size: %.1f GB (streamed, never stored whole).",
            size_bytes / 1e9,
        )

    return url, meta


def open_archive_stream(session: requests.Session, url: str) -> Tuple[requests.Response, tarfile.TarFile]:
    # The download URL is pre-signed: sending the API key again can make S3
    # reject the request ("only one auth mechanism allowed"). No auth here.
    headers = {"Accept-Encoding": "identity"}
    r = session.get(url, headers=headers, stream=True, timeout=CFG.timeout_s)
    r.raise_for_status()
    r.raw.decode_content = False  # tarfile does the gzip decoding itself
    return r, tarfile.open(fileobj=r.raw, mode="r|gz")


# --------------------------------------------------------------------------- #
# Pass 1: metadata only
# --------------------------------------------------------------------------- #

def fetch_metadata(session: requests.Session, url: str) -> Tuple[List[dict], Optional[Dict[str, float]]]:
    """Stream the archive until validated.tsv / clip_durations.tsv are read."""
    want = {"validated.tsv": None, "clip_durations.tsv": None}
    LOGGER.info("Pass 1: streaming archive header for metadata TSVs ...")
    r, tar = open_archive_stream(session, url)
    try:
        for member in tar:
            base = os.path.basename(member.name)
            if base in want and member.isfile():
                want[base] = tar.extractfile(member).read().decode("utf-8", "replace")
                LOGGER.info("  captured %s (%d bytes)", base, member.size)
                if all(v is not None for v in want.values()):
                    break
            # validated.tsv is at the archive root; if we are deep into clips/
            # without it, something is wrong — keep scanning anyway (cheap).
    finally:
        r.close()
    if want["validated.tsv"] is None:
        raise RuntimeError("validated.tsv not found in the archive stream; "
                           "is this a Common Voice language bundle?")

    rows = []
    reader = csv.DictReader(want["validated.tsv"].splitlines(), delimiter="\t")
    for row in reader:
        rows.append(row)
    durations = None
    if want["clip_durations.tsv"] is not None:
        durations = {}
        for line in want["clip_durations.tsv"].splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    durations[parts[0]] = float(parts[1]) / 1000.0  # ms -> s
                except ValueError:
                    continue
    LOGGER.info("Pass 1 done: %d validated rows; clip_durations.tsv %s.",
                len(rows), "found" if durations else "NOT found (will use slack %.2f)"
                % CFG.slack_without_durations)
    return rows, durations


# --------------------------------------------------------------------------- #
# Selection (mirrors the Hindi pipeline order: shuffle once, filter while
# scanning, stop at the pool target)
# --------------------------------------------------------------------------- #

def select_sources(rows: List[dict], durations: Optional[Dict[str, float]]) -> List[dict]:
    pool = int(-(-CFG.target_pairs * CFG.oversample_factor // 1))  # ceil
    slack = CFG.slack_with_durations if durations else CFG.slack_without_durations
    target = int(-(-pool * slack // 1))

    order = list(range(len(rows)))
    random.Random(CFG.seed).shuffle(order)

    speaker_counts: Dict[str, int] = {}
    chosen: List[dict] = []
    for i in order:
        if len(chosen) >= target:
            break
        row = rows[i]
        try:
            if int(row.get("up_votes", 0)) - int(row.get("down_votes", 0)) < CFG.min_vote_margin:
                continue
        except (TypeError, ValueError):
            continue
        spk = str(row.get("client_id", ""))
        if not spk or speaker_counts.get(spk, 0) >= CFG.max_clips_per_speaker:
            continue
        path = str(row.get("path", ""))
        if not path:
            continue
        if durations is not None:
            dur = durations.get(path) or durations.get(os.path.basename(path))
            if dur is None or not (CFG.min_duration_s <= dur <= CFG.max_duration_s):
                continue
        speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
        chosen.append(row)

    LOGGER.info("Selected %d clips from %d speakers (target %d, pool %d).",
                len(chosen), len(speaker_counts), target, pool)
    if len(chosen) < pool:
        raise RuntimeError(
            f"Selection ({len(chosen)}) is below the pipeline pool size ({pool}). "
            f"The validated split may be smaller than expected; loosen filters.")
    return chosen


# --------------------------------------------------------------------------- #
# Pass 2: stream-extract selected clips
# --------------------------------------------------------------------------- #

def stream_extract(session: requests.Session, url: str, chosen: List[dict],
                   clips_dir: Path) -> List[str]:
    """Stream the archive, keeping only selected clips. Returns missing paths."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    wanted = {os.path.basename(str(r["path"])) for r in chosen}
    # resume: never re-write a clip that is already on disk
    wanted -= {p.name for p in clips_dir.glob("*.mp3") if p.stat().st_size > 0}
    LOGGER.info("Pass 2: streaming archive, extracting %d clips ...", len(wanted))
    found = set()
    r, tar = open_archive_stream(session, url)
    t0 = time.time()
    try:
        for member in tar:
            base = os.path.basename(member.name)
            if base in wanted and member.isfile():
                src = tar.extractfile(member)
                dest = clips_dir / base
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                tmp.replace(dest)
                found.add(base)
                if len(found) % 500 == 0:
                    LOGGER.info("  %d/%d clips extracted (%.0f min elapsed).",
                                len(found), len(wanted), (time.time() - t0) / 60)
                if found >= wanted:
                    LOGGER.info("All clips found; stopping the stream early.")
                    break
    finally:
        r.close()
    missing = sorted(wanted - found)
    if missing:
        LOGGER.warning("%d selected clips were NOT found in the archive.", len(missing))
        (Path(CFG.out) / "missing_clips.txt").write_text("\n".join(missing),
                                                         encoding="utf-8")
    return missing


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_outputs(chosen: List[dict], durations: Optional[Dict[str, float]],
                  missing: List[str], dl_meta: Optional[dict] = None) -> None:
    out = Path(CFG.out)
    en = out / "en"
    en.mkdir(parents=True, exist_ok=True)
    ok = [r for r in chosen if os.path.basename(str(r["path"])) not in set(missing)]

    fieldnames = list(chosen[0].keys()) if chosen else ["client_id", "path"]
    with open(en / "validated.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in sorted(ok, key=lambda r: str(r["path"])):
            w.writerow(r)

    if durations is not None:
        with open(en / "clip_durations.tsv", "w", encoding="utf-8") as f:
            f.write("clip\tduration[ms]\n")
            for r in sorted(ok, key=lambda r: str(r["path"])):
                base = os.path.basename(str(r["path"]))
                d = durations.get(str(r["path"])) or durations.get(base)
                if d is not None:
                    f.write(f"{base}\t{int(round(d * 1000))}\n")

    with open(out / "downloaded_sources.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["client_id", "path", "up_votes",
                                          "down_votes"], extrasaction="ignore")
        w.writeheader()
        for r in sorted(ok, key=lambda r: str(r["path"])):
            w.writerow(r)

    manifest = {
        "source": "Mozilla Data Collective", "dataset_id": CFG.dataset_id,
        "archive": dl_meta or {},
        "params": {k: v for k, v in asdict(CFG).items()
                   if k not in ("api_key", "dataset_id")},
        "n_selected": len(chosen), "n_written": len(ok), "n_missing": len(missing),
        "n_speakers": len({str(r.get("client_id", "")) for r in ok}),
        "note": "validated.tsv lists only downloaded clips; point "
                "prepare_english_griffinlim.py Config.cv_dir at <out>/en.",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "download_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                encoding="utf-8")
    LOGGER.info("Wrote %d rows to %s", len(ok), en / "validated.tsv")
    LOGGER.info("Next: set Config.cv_dir = %r in prepare_english_griffinlim.py",
                str(en))


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-id", default=CFG.dataset_id,
                   help="MDC dataset id or slug (dataset page -> Connect API)")
    p.add_argument("--api-key", default=os.environ.get("MDC_API_KEY", ""))
    p.add_argument("--out", default=CFG.out)
    p.add_argument("--seed", type=int, default=CFG.seed)
    args = p.parse_args(argv)
    CFG.out, CFG.dataset_id, CFG.seed = args.out, args.dataset_id, args.seed
    CFG.api_key = args.api_key
    if not CFG.dataset_id:
        p.error("--dataset-id is required (see docstring prerequisites)")
    if not CFG.api_key:
        p.error("MDC API key required: --api-key or env MDC_API_KEY")

    session = requests.Session()
    url, dl_meta = resolve_download_url(session)
    rows, durations = fetch_metadata(session, url)
    chosen = select_sources(rows, durations)
    missing = stream_extract(session, url, chosen, Path(CFG.out) / "en" / "clips")
    write_outputs(chosen, durations, missing, dl_meta)


if __name__ == "__main__":
    sys.exit(main())
