#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resumable Common Voice English subset downloader for Phase 5.

Why this exists:
  The current MDC English release is a single large .tar.gz archive.  A normal
  tar.gz stream cannot resume decompression from an arbitrary compressed byte.
  This version uses indexed_gzip + HTTP Range requests so the gzip stream can
  seek to previously checkpointed *uncompressed* positions without storing the
  94+ GB archive locally.

Properties:
  - preserves the Phase-5 selection: target_pairs=1125, oversample=1.05,
    therefore extraction target remains 1301; do NOT change this unless the
    experiment is intentionally changed.
  - reuses already downloaded MP3s.
  - persists selected_sources_full.csv so selection does not change across runs.
  - checkpoints tar progress and an indexed-gzip index periodically.
  - retries interrupted HTTP range reads instead of aborting the whole pass.
  - does not download/store the entire archive.

One-time dependency:
    pip install "indexed-gzip>=1.9,<1.11"

Usage:
    python download_english_subset_resumable.py \
        --dataset-id cmqim2hn800ssnr07gvmpcnwu \
        --out ./cv_english_subset

The script expects MDC_API_KEY in the environment or --api-key.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

try:
    import indexed_gzip as igzip
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'indexed-gzip'. Install it with:\n"
        '    pip install "indexed-gzip>=1.9,<1.11"\n'
    ) from exc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("cv_en_resumable")

MDC_API = os.environ.get(
    "MDC_API_URL", "https://mozilladatacollective.com/api"
).rstrip("/")

TAR_BLOCK = 512


@dataclass
class Config:
    out: str = "./cv_english_subset"
    dataset_id: str = ""
    api_key: str = ""

    seed: int = 42
    target_pairs: int = 1125
    oversample_factor: float = 1.05

    # Keep these identical to the established Phase-5 English/Hindi protocol.
    min_vote_margin: int = 2
    max_clips_per_speaker: int = 5
    min_duration_s: float = 2.0
    max_duration_s: float = 8.0

    # Keep the user's current 1301 extraction target exactly as-is.
    extraction_target: int = 1301

    # HTTP / gzip tuning.
    request_timeout_s: int = 180
    http_retries: int = 5
    range_read_size: int = 16 * 1024 * 1024
    max_url_refreshes: int = 6

    # Checkpoint every ~512 MiB of uncompressed TAR traversal.
    checkpoint_bytes: int = 512 * 1024 * 1024
    gzip_index_spacing: int = 16 * 1024 * 1024


CFG = Config()
OUT = Path(CFG.out)
CLIPS_DIR = OUT / "en" / "clips"
STATE_PATH = OUT / "resumable_state.json"
INDEX_PATH = OUT / "archive.gzidx"
SELECTED_PATH = OUT / "selected_sources_full.csv"


class HTTPRangeFile:
    """A seekable read-only file backed by HTTP Range requests.

    The object presents the compressed archive bytes to indexed_gzip.  It keeps
    only a local read buffer and never stores the complete archive.
    """

    def __init__(
        self,
        session: requests.Session,
        url: str,
        size_bytes: int,
        refresh_url_cb,
        read_size: int,
        retries: int,
    ) -> None:
        self.session = session
        self.url = url
        self.size_bytes = int(size_bytes)
        self.refresh_url_cb = refresh_url_cb
        self.read_size = int(read_size)
        self.retries = int(retries)
        self.pos = 0
        self.buf_start = 0
        self.buf = b""
        self.url_refreshes = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            new_pos = offset
        elif whence == 1:
            new_pos = self.pos + offset
        elif whence == 2:
            new_pos = self.size_bytes + offset
        else:
            raise ValueError(f"unsupported whence={whence}")
        if new_pos < 0:
            raise ValueError("negative seek position")
        self.pos = min(new_pos, self.size_bytes)
        return self.pos

    def _refresh_url(self) -> None:
        if self.url_refreshes >= CFG.max_url_refreshes:
            raise RuntimeError(
                "Exceeded maximum MDC presigned-URL refreshes."
            )
        self.url = self.refresh_url_cb()
        self.url_refreshes += 1
        self.buf = b""
        self.buf_start = self.pos
        LOGGER.warning(
            "Refreshed MDC presigned URL (%d/%d).",
            self.url_refreshes,
            CFG.max_url_refreshes,
        )

    def _fetch(self, start: int, length: int) -> bytes:
        if start >= self.size_bytes or length <= 0:
            return b""
        end = min(self.size_bytes - 1, start + length - 1)
        headers = {
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
        }

        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.get(
                    self.url,
                    headers=headers,
                    stream=False,
                    timeout=CFG.request_timeout_s,
                    allow_redirects=True,
                )
                # A presigned URL may expire or otherwise become invalid.
                if r.status_code in (401, 403, 410):
                    r.close()
                    self._refresh_url()
                    continue
                r.raise_for_status()

                data = r.content
                r.close()

                if start == 0 and r.status_code == 200:
                    # A server may ignore Range for a zero-offset request.  In
                    # that case the content can still be useful, but it must
                    # cover the requested region.
                    if len(data) < min(length, self.size_bytes):
                        raise IOError(
                            f"HTTP 200 returned only {len(data)} bytes for a "
                            f"{length}-byte request"
                        )
                elif r.status_code != 206:
                    raise IOError(
                        f"Expected HTTP 206 for Range {start}-{end}, got "
                        f"{r.status_code}"
                    )

                expected = end - start + 1
                if len(data) < expected:
                    raise IOError(
                        f"short range read: got {len(data)} bytes, expected "
                        f"{expected}"
                    )
                return data[:expected]

            except Exception as exc:
                last_exc = exc
                LOGGER.warning(
                    "Range read failed at byte %d (attempt %d/%d): %s",
                    start,
                    attempt,
                    self.retries,
                    exc,
                )
                time.sleep(min(2 ** (attempt - 1), 15))
                # Refresh only after a repeated failure, not on every retry.
                if attempt >= 2:
                    try:
                        self._refresh_url()
                    except Exception:
                        pass

        raise IOError(
            f"Unable to read compressed archive bytes {start}-{end}"
        ) from last_exc

    def read(self, size: int = -1) -> bytes:
        if self.pos >= self.size_bytes:
            return b""
        if size is None or size < 0:
            size = self.size_bytes - self.pos
        else:
            size = min(size, self.size_bytes - self.pos)

        chunks: List[bytes] = []
        remaining = size
        while remaining > 0:
            buf_end = self.buf_start + len(self.buf)
            if self.buf and self.buf_start <= self.pos < buf_end:
                take = min(remaining, buf_end - self.pos)
                start_i = self.pos - self.buf_start
                chunks.append(self.buf[start_i:start_i + take])
                self.pos += take
                remaining -= take
                continue

            fetch_len = max(self.read_size, remaining)
            fetch_len = min(fetch_len, self.size_bytes - self.pos)
            data = self._fetch(self.pos, fetch_len)
            if not data:
                break
            self.buf_start = self.pos
            self.buf = data

        return b"".join(chunks)

    def close(self) -> None:
        self.buf = b""


class ResumableExtractor:
    def __init__(self, selected: List[dict], session: requests.Session) -> None:
        self.selected = selected
        self.session = session
        self.selected_by_basename = {
            os.path.basename(str(r["path"])): r for r in selected
        }
        self.remaining = set(self.selected_by_basename)
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        for p in CLIPS_DIR.glob("*.mp3"):
            if p.is_file() and p.stat().st_size > 0:
                self.remaining.discard(p.name)

    def save_state(self, next_offset: int, archive_size: int) -> None:
        payload = {
            "dataset_id": CFG.dataset_id,
            "next_uncompressed_tar_offset": int(next_offset),
            "archive_size_bytes": int(archive_size),
            "extraction_target": CFG.extraction_target,
            "selected_count": len(self.selected),
            "remaining_count": len(self.remaining),
            "downloaded_existing_count": len(self.selected) - len(self.remaining),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(STATE_PATH)

    def load_state(self) -> int:
        if not STATE_PATH.exists():
            return 0
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if state.get("dataset_id") != CFG.dataset_id:
                LOGGER.warning("State dataset ID differs; starting from tar offset 0.")
                return 0
            if int(state.get("extraction_target", -1)) != CFG.extraction_target:
                LOGGER.warning("State extraction target differs; starting from offset 0.")
                return 0
            return max(0, int(state.get("next_uncompressed_tar_offset", 0)))
        except Exception as exc:
            LOGGER.warning("Ignoring unreadable resumable state: %s", exc)
            return 0

    def _save_index(self, gz) -> None:
        tmp = INDEX_PATH.with_suffix(".tmp")
        gz.export_index(str(tmp))
        tmp.replace(INDEX_PATH)

    def _read_exact(self, gz, n: int) -> bytes:
        chunks: List[bytes] = []
        remaining = n
        while remaining:
            part = gz.read(remaining)
            if not part:
                break
            chunks.append(part)
            remaining -= len(part)
        data = b"".join(chunks)
        if len(data) != n:
            raise EOFError(f"expected {n} bytes from tar stream, got {len(data)}")
        return data

    @staticmethod
    def _tar_name(header: bytes) -> str:
        name = header[0:100].rstrip(b"\0").decode("utf-8", "replace")
        prefix = header[345:500].rstrip(b"\0").decode("utf-8", "replace")
        return f"{prefix}/{name}" if prefix and name else (prefix or name)

    @staticmethod
    def _tar_size(header: bytes) -> int:
        raw = header[124:136].rstrip(b"\0 ")
        if not raw:
            return 0
        try:
            return int(raw, 8)
        except ValueError:
            # GNU base-256 is not expected here, but handle it defensively.
            if raw and (raw[0] & 0x80):
                data = bytearray(raw)
                data[0] &= 0x7F
                return int.from_bytes(data, "big", signed=False)
            raise

    @staticmethod
    def _is_zero_block(header: bytes) -> bool:
        return not header.strip(b"\0")

    def extract(self, get_url_cb, archive_url: str, archive_size: int) -> None:
        if not self.remaining:
            LOGGER.info("All %d selected clips are already present.", len(self.selected))
            return

        resume_offset = self.load_state()
        LOGGER.info(
            "Resumable extraction: %d/%d clips still missing; resume tar offset=%d.",
            len(self.remaining), len(self.selected), resume_offset,
        )

        def refresh_url() -> str:
            return get_url_cb()

        remote = HTTPRangeFile(
            self.session,
            archive_url,
            archive_size,
            refresh_url,
            CFG.range_read_size,
            CFG.http_retries,
        )

        # indexed_gzip maps uncompressed gzip offsets to compressed offsets and
        # persists the seek index between runs. This is what makes checkpoints
        # useful after a network failure.
        try:
            gz = igzip.IndexedGzipFile(
                fileobj=remote,
                index_file=str(INDEX_PATH) if INDEX_PATH.exists() else None,
                spacing=CFG.gzip_index_spacing,
                readbuf_size=CFG.range_read_size,
                auto_build=True,
            )
        except TypeError:
            # Compatibility with older indexed_gzip releases which may not
            # accept all keyword arguments together.
            gz = igzip.IndexedGzipFile(
                fileobj=remote,
                index_file=str(INDEX_PATH) if INDEX_PATH.exists() else None,
                spacing=CFG.gzip_index_spacing,
            )

        if resume_offset:
            try:
                gz.seek(resume_offset)
            except Exception as exc:
                LOGGER.warning(
                    "Could not seek to saved offset %d (%s); restarting from 0.",
                    resume_offset,
                    exc,
                )
                resume_offset = 0
                gz.seek(0)
        else:
            gz.seek(0)

        next_checkpoint = max(
            CFG.checkpoint_bytes,
            ((resume_offset // CFG.checkpoint_bytes) + 1) * CFG.checkpoint_bytes,
        )
        last_report = time.time()
        processed_headers = 0
        start_time = time.time()

        try:
            while True:
                header_offset = gz.tell()
                header = self._read_exact(gz, TAR_BLOCK)
                if self._is_zero_block(header):
                    # TAR normally ends with two zero blocks. One is enough to
                    # signal end for our purposes.
                    LOGGER.info("Reached end of TAR stream.")
                    break

                name = self._tar_name(header)
                size = self._tar_size(header)
                typeflag = header[156:157]
                data_start = gz.tell()
                padded_size = ((size + TAR_BLOCK - 1) // TAR_BLOCK) * TAR_BLOCK
                data_end = data_start + padded_size
                processed_headers += 1

                base = os.path.basename(name)
                selected = base in self.remaining and typeflag in (b"0", b"", b"\x00")

                if selected:
                    dest = CLIPS_DIR / base
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    LOGGER.info(
                        "Extracting %s (%d bytes); %d remaining after this file.",
                        base,
                        size,
                        len(self.remaining) - 1,
                    )
                    with open(tmp, "wb") as fh:
                        remaining = size
                        while remaining:
                            chunk = gz.read(min(CFG.range_read_size, remaining))
                            if not chunk:
                                raise EOFError(f"unexpected EOF reading {base}")
                            fh.write(chunk)
                            remaining -= len(chunk)
                        fh.flush()
                        os.fsync(fh.fileno())
                    tmp.replace(dest)
                    self.remaining.discard(base)

                    # Skip TAR padding without materializing it.
                    if padded_size > size:
                        gz.seek(data_start + padded_size)
                else:
                    # Do not decompress material that is not needed; indexed_gzip
                    # can seek to the next TAR header using its gzip index.
                    gz.seek(data_end)

                current = gz.tell()
                if current >= next_checkpoint:
                    self.save_state(current, archive_size)
                    try:
                        self._save_index(gz)
                    except Exception as exc:
                        LOGGER.warning("Could not save gzip index: %s", exc)
                    next_checkpoint = (
                        (current // CFG.checkpoint_bytes) + 1
                    ) * CFG.checkpoint_bytes

                if not self.remaining:
                    self.save_state(current, archive_size)
                    try:
                        self._save_index(gz)
                    except Exception as exc:
                        LOGGER.warning("Could not save final gzip index: %s", exc)
                    LOGGER.info("All selected clips extracted.")
                    break

                if time.time() - last_report > 60:
                    elapsed = max(time.time() - start_time, 1.0)
                    LOGGER.info(
                        "Progress: %d/%d clips present; %d TAR headers; "
                        "uncompressed scan %.1f GiB; elapsed %.1f min.",
                        len(self.selected) - len(self.remaining),
                        len(self.selected),
                        processed_headers,
                        current / (1024 ** 3),
                        elapsed / 60,
                    )
                    last_report = time.time()

        except Exception:
            current = gz.tell()
            self.save_state(current, archive_size)
            try:
                self._save_index(gz)
            except Exception as exc:
                LOGGER.warning("Could not save gzip index after failure: %s", exc)
            LOGGER.exception(
                "Extraction interrupted at uncompressed TAR offset %d. "
                "Rerun this script; it will resume from the checkpoint.",
                current,
            )
            raise
        finally:
            try:
                gz.close()
            except Exception:
                pass
            remote.close()

        # Clean state only after the extraction itself is complete.
        if not self.remaining:
            if STATE_PATH.exists():
                STATE_PATH.unlink()
            LOGGER.info("Extraction complete: %d clips present.", len(self.selected))
        else:
            LOGGER.warning(
                "Extraction ended with %d selected clips still missing.",
                len(self.remaining),
            )


def resolve_download_url(session: requests.Session) -> Tuple[str, dict]:
    headers = {
        "Authorization": f"Bearer {CFG.api_key}",
        "Content-Length": "0",
    }
    r = session.post(
        f"{MDC_API}/datasets/{CFG.dataset_id}/download",
        headers=headers,
        timeout=CFG.request_timeout_s,
        allow_redirects=False,
    )
    if r.is_redirect or r.is_permanent_redirect:
        raise RuntimeError(
            f"MDC redirected the download request (HTTP {r.status_code}). "
            "Use the canonical MDC API host."
        )
    if r.status_code == 401:
        raise RuntimeError("MDC authentication failed (401).")
    if r.status_code == 403:
        raise RuntimeError("MDC access denied (403): accept dataset terms first.")
    if r.status_code == 404:
        raise RuntimeError("Dataset not found (404).")
    if r.status_code == 429:
        raise RuntimeError("MDC rate limit (429).")
    r.raise_for_status()
    payload = r.json()
    url = payload.get("downloadUrl") or payload.get("download_url") or payload.get("url")
    if not url:
        raise RuntimeError(f"Unexpected MDC response: {payload!r}")
    raw_size = payload.get("sizeBytes")
    try:
        size = int(raw_size) if raw_size is not None else None
    except (TypeError, ValueError):
        size = None
    if size is None:
        raise RuntimeError("MDC response did not contain a usable sizeBytes value.")
    LOGGER.info("Archive size: %.1f GB (never stored as a whole).", size / 1e9)
    return url, {"sizeBytes": size, "checksum": payload.get("checksum")}


def fetch_metadata(session: requests.Session, url: str) -> Tuple[List[dict], Dict[str, float]]:
    """Small metadata pass; this is intentionally kept because it is cheap."""
    # Use the same sequential stream as before for the small root metadata files.
    r = session.get(
        url,
        headers={"Accept-Encoding": "identity"},
        stream=True,
        timeout=CFG.request_timeout_s,
    )
    r.raise_for_status()
    r.raw.decode_content = False
    import tarfile

    tar = tarfile.open(fileobj=r.raw, mode="r|gz")
    want: Dict[str, Optional[str]] = {"validated.tsv": None, "clip_durations.tsv": None}
    try:
        for member in tar:
            base = os.path.basename(member.name)
            if base in want and member.isfile():
                want[base] = tar.extractfile(member).read().decode("utf-8", "replace")
                LOGGER.info("  captured %s (%d bytes)", base, member.size)
                if all(v is not None for v in want.values()):
                    break
    finally:
        r.close()

    if want["validated.tsv"] is None:
        raise RuntimeError("validated.tsv not found in MDC archive.")

    rows = list(csv.DictReader(want["validated.tsv"].splitlines(), delimiter="\t"))
    durations: Dict[str, float] = {}
    if want["clip_durations.tsv"] is not None:
        for line in want["clip_durations.tsv"].splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    durations[parts[0]] = float(parts[1]) / 1000.0
                except ValueError:
                    pass
    LOGGER.info(
        "Pass 1 done: %d validated rows; clip_durations.tsv %s.",
        len(rows),
        "found" if durations else "NOT found",
    )
    return rows, durations


def select_sources(rows: List[dict], durations: Dict[str, float]) -> List[dict]:
    """Exactly preserves the current 1301-clip selection target."""
    if SELECTED_PATH.exists():
        LOGGER.info("Loading frozen selection from %s", SELECTED_PATH)
        with SELECTED_PATH.open("r", newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    # Keep the user's current 1301 number. Do not change this.
    pool = int(math.ceil(CFG.target_pairs * CFG.oversample_factor))  # 1182
    target = CFG.extraction_target  # 1301 by explicit experiment choice

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
        dur = durations.get(path) or durations.get(os.path.basename(path))
        if dur is None or not (CFG.min_duration_s <= dur <= CFG.max_duration_s):
            continue
        speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
        chosen.append(row)

    if len(chosen) < target:
        raise RuntimeError(
            f"Could only select {len(chosen)} clips; need {target}."
        )

    # Freeze the exact current selection for all future resumes.
    fields = list(chosen[0].keys())
    SELECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SELECTED_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(chosen)

    LOGGER.info(
        "Selected/frozen %d clips from %d speakers (pool=%d, extraction target=%d).",
        len(chosen),
        len(speaker_counts),
        pool,
        target,
    )
    return chosen


def write_outputs(chosen: List[dict]) -> None:
    """Create the CommonVoice-shaped metadata expected by the generator."""
    en = OUT / "en"
    en.mkdir(parents=True, exist_ok=True)
    present = {p.name for p in CLIPS_DIR.glob("*.mp3") if p.stat().st_size > 0}
    ok = [r for r in chosen if os.path.basename(str(r["path"])) in present]

    if ok:
        fields = list(ok[0].keys())
        with (en / "validated.tsv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(sorted(ok, key=lambda r: str(r.get("path", ""))))

    # The English generator only requires validated.tsv + clips/ and will probe
    # durations itself. Still write a provenance manifest for reproducibility.
    manifest = {
        "dataset_id": CFG.dataset_id,
        "selection_file": str(SELECTED_PATH),
        "extraction_target": CFG.extraction_target,
        "n_selected": len(chosen),
        "n_downloaded": len(ok),
        "n_remaining": len(chosen) - len(ok),
        "seed": CFG.seed,
        "min_vote_margin": CFG.min_vote_margin,
        "max_clips_per_speaker": CFG.max_clips_per_speaker,
        "duration_window_s": [CFG.min_duration_s, CFG.max_duration_s],
    }
    (OUT / "resumable_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--api-key", default=os.environ.get("MDC_API_KEY", ""))
    parser.add_argument("--out", default=CFG.out)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    args = parser.parse_args(argv)

    CFG.out = args.out
    CFG.dataset_id = args.dataset_id
    CFG.api_key = args.api_key
    CFG.seed = args.seed

    if not CFG.api_key:
        parser.error("MDC API key required: --api-key or env MDC_API_KEY")

    global OUT, CLIPS_DIR, STATE_PATH, INDEX_PATH, SELECTED_PATH
    OUT = Path(CFG.out)
    CLIPS_DIR = OUT / "en" / "clips"
    STATE_PATH = OUT / "resumable_state.json"
    INDEX_PATH = OUT / "archive.gzidx"
    SELECTED_PATH = OUT / "selected_sources_full.csv"
    OUT.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    url, meta = resolve_download_url(session)

    # If a frozen selection exists, Pass 1 is unnecessary. Otherwise build it.
    if SELECTED_PATH.exists():
        with SELECTED_PATH.open("r", newline="", encoding="utf-8") as fh:
            selected = list(csv.DictReader(fh))
        LOGGER.info("Reusing frozen 1301-clip selection; skipping metadata scan.")
    else:
        rows, durations = fetch_metadata(session, url)
        selected = select_sources(rows, durations)

    extractor = ResumableExtractor(selected, session)

    # Refresh callback for expired/broken presigned URLs. The API documentation
    # says these URLs are valid for a limited period and can be renewed.
    def refresh_url() -> str:
        new_url, _ = resolve_download_url(session)
        return new_url

    try:
        extractor.extract(refresh_url, url, int(meta["sizeBytes"]))
    finally:
        write_outputs(selected)

    if extractor.remaining:
        LOGGER.warning("%d selected clips remain; rerun to continue.", len(extractor.remaining))
        return 2

    LOGGER.info("All 1301 selected English clips are available under %s", CLIPS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
