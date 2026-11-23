"""Reproducible acquisition engine for NASA/JPL AI4MARS v0.6 and NASA raw samples.

Downloads verified Mars rover imagery, semantic segmentation masks, rover masks,
30-meter range masks, and official documentation directly from the official
Zenodo AI4MARS v0.6 archive (DOI: 10.5281/zenodo.15995036) using HTTP Range
requests with CRC32 verification and Zip64 Central Directory parsing.

Also acquires a small qualitative sample of external Mars imagery (Perseverance
M2020 and recent Curiosity MSL) from official NASA Mars Raw Image APIs, kept
strictly separate under `datasets/samples/external_nasa/`.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import logging
import re
import shutil
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from tqdm import tqdm

ZENODO_V06_URL = (
    "https://zenodo.org/api/records/15995036/files/"
    "ai4mars-dataset-merged-0.6.zip/content"
)
M2020_RAW_API = (
    "https://mars.nasa.gov/rss/api/"
    "?feed=raw_images&category=mars2020,ingenuity&feedtype=json&num=20"
)
MSL_RAW_API = (
    "https://mars.nasa.gov/api/v1/raw_image_items/"
    "?order=sol+desc&per_page=25&page=0&condition_1=msl:mission"
)

LOGGER = logging.getLogger("mangal_drishti.acquire")


@dataclass(frozen=True)
class ZipEntry:
    """Represents a single entry parsed from a Zip64 Central Directory."""

    filename: str
    comp_method: int
    crc32: int
    comp_size: int
    uncomp_size: int
    offset: int
    end_offset: int


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "MangalDrishti-DatasetAcquirer/1.0 (Research/PlanetaryAI)",
        }
    )
    return session


def fetch_range(
    session: requests.Session,
    url: str,
    start: int,
    end: int,
    max_retries: int = 10,
    timeout: int = 90,
) -> bytes:
    """Fetch an inclusive byte range [start, end] with retries and rate-limit handling."""
    headers = {"Range": f"bytes={start}-{end}"}
    expected_len = end - start + 1
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, headers=headers, timeout=timeout)
            remaining = resp.headers.get("X-RateLimit-Remaining")
            reset_hdr = resp.headers.get("X-RateLimit-Reset")
            if remaining is not None and int(remaining) <= 3 and reset_hdr is not None:
                delay = max(1.0, min(float(reset_hdr) - time.time() + 1.5, 65.0))
                LOGGER.info(
                    "Approaching Zenodo rate limit (remaining=%s); pausing %.1fs",
                    remaining,
                    delay,
                )
                time.sleep(delay)
            if resp.status_code in (200, 206):
                content = resp.content
                if resp.status_code == 206 and len(content) == expected_len:
                    return content
                if resp.status_code == 200 and len(content) >= expected_len:
                    return content[:expected_len]
                LOGGER.warning(
                    "Short read on range %d-%d: got %d bytes (expected %d)",
                    start,
                    end,
                    len(content),
                    expected_len,
                )
            elif resp.status_code in (429, 500, 502, 503, 504):
                retry_after = float(resp.headers.get("Retry-After", 2**attempt))
                wait_s = min(retry_after + 1.5, 65.0)
                LOGGER.warning(
                    "HTTP %d for %d-%d (attempt %d/%d); sleeping %.1fs",
                    resp.status_code,
                    start,
                    end,
                    attempt,
                    max_retries,
                    wait_s,
                )
                time.sleep(wait_s)
                continue
            else:
                resp.raise_for_status()
        except requests.RequestException as exc:
            wait_s = min(2.0**attempt, 20.0)
            LOGGER.warning(
                "Request error on range %d-%d (%s) attempt %d/%d; sleeping %.1fs",
                start,
                end,
                exc,
                attempt,
                max_retries,
                wait_s,
            )
            time.sleep(wait_s)
    raise RuntimeError(f"Failed to fetch byte range {start}-{end} after {max_retries} attempts")


def ensure_central_directory(
    cd_path: Path, url: str = ZENODO_V06_URL
) -> bytes:
    """Load cached Zip64 Central Directory bytes or download from Zenodo."""
    if cd_path.exists() and cd_path.stat().st_size > 1_000_000:
        LOGGER.info("Using cached Central Directory: %s", cd_path)
        return cd_path.read_bytes()

    LOGGER.info("Downloading Zip64 Central Directory metadata from Zenodo...")
    session = _create_session()
    head = session.head(url, allow_redirects=True, timeout=30)
    head.raise_for_status()
    total_size = int(head.headers["Content-Length"])

    tail_size = 65536
    tail = fetch_range(session, url, total_size - tail_size, total_size - 1)
    z64_idx = tail.rfind(b"PK\x06\x06")
    if z64_idx == -1:
        raise RuntimeError("Zip64 EOCD signature not found in archive tail.")
    _, _, _, _, _, _, _, _, cd_size, cd_offset = struct.unpack(
        "<4sQHHIIQQQQ", tail[z64_idx : z64_idx + 56]
    )
    LOGGER.info(
        "Archive size: %.2f GB | CD offset: %d | CD size: %.2f MB",
        total_size / 1e9,
        cd_offset,
        cd_size / 1e6,
    )
    cd_bytes = fetch_range(session, url, cd_offset, cd_offset + cd_size - 1)
    cd_path.parent.mkdir(parents=True, exist_ok=True)
    cd_path.write_bytes(cd_bytes)
    return cd_bytes


def parse_central_directory(cd_bytes: bytes, archive_total_size: int) -> List[ZipEntry]:
    """Parse all Zip64 Central Directory records and compute upper byte bounds."""
    raw_entries: List[Tuple[str, int, int, int, int, int]] = []
    pos = 0
    n = len(cd_bytes)

    while pos < n:
        if cd_bytes[pos : pos + 4] != b"PK\x01\x02":
            break
        (
            _,
            _,
            _,
            _flags,
            comp,
            _,
            _,
            crc,
            csize,
            usize,
            fn_len,
            ex_len,
            cm_len,
            _,
            _,
            _,
            offset,
        ) = struct.unpack("<4sHHHHHHIIIHHHHHII", cd_bytes[pos : pos + 46])
        fn = cd_bytes[pos + 46 : pos + 46 + fn_len].decode("utf-8", errors="replace")
        ex = cd_bytes[pos + 46 + fn_len : pos + 46 + fn_len + ex_len]

        if offset == 0xFFFFFFFF or csize == 0xFFFFFFFF or usize == 0xFFFFFFFF:
            ep = 0
            while ep < len(ex):
                eh, es = struct.unpack("<HH", ex[ep : ep + 4])
                if eh == 1:
                    vals = struct.unpack("<" + "Q" * (es // 8), ex[ep + 4 : ep + 4 + es])
                    vi = 0
                    if usize == 0xFFFFFFFF:
                        usize = vals[vi]
                        vi += 1
                    if csize == 0xFFFFFFFF:
                        csize = vals[vi]
                        vi += 1
                    if offset == 0xFFFFFFFF:
                        offset = vals[vi]
                        vi += 1
                ep += 4 + es

        if not fn.endswith("/"):
            raw_entries.append((fn, comp, crc, csize, usize, offset))
        pos += 46 + fn_len + ex_len + cm_len

    raw_entries.sort(key=lambda item: item[5])
    entries: List[ZipEntry] = []
    for idx, (fn, comp, crc, csize, usize, offset) in enumerate(raw_entries):
        if idx + 1 < len(raw_entries):
            next_off = raw_entries[idx + 1][5]
            # Bound end_offset safely while avoiding huge gaps
            end_offset = min(next_off - 1, offset + 256 + len(fn.encode("utf-8")) + csize)
        else:
            end_offset = min(
                archive_total_size - 1, offset + 256 + len(fn.encode("utf-8")) + csize
            )
        entries.append(
            ZipEntry(
                filename=fn,
                comp_method=comp,
                crc32=crc,
                comp_size=csize,
                uncomp_size=usize,
                offset=offset,
                end_offset=end_offset,
            )
        )
    return entries


def save_archive_manifest_summary(entries: List[ZipEntry], output_path: Path) -> None:
    """Save an accurate inventory of the full official AI4MARS v0.6 archive."""
    dir_stats: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: {"file_count": 0, "compressed_bytes": 0, "uncompressed_bytes": 0}
    )
    ext_counts: collections.Counter[str] = collections.Counter()
    total_c = 0
    total_u = 0

    for entry in entries:
        parent = entry.filename.rsplit("/", 1)[0] if "/" in entry.filename else ""
        dir_stats[parent]["file_count"] += 1
        dir_stats[parent]["compressed_bytes"] += entry.comp_size
        dir_stats[parent]["uncompressed_bytes"] += entry.uncomp_size
        ext = entry.filename.rsplit(".", 1)[-1] if "." in entry.filename else ""
        ext_counts[ext] += 1
        total_c += entry.comp_size
        total_u += entry.uncomp_size

    summary = {
        "source": "NASA/JPL AI4MARS v0.6 (Zenodo Record 15995036)",
        "doi": "10.5281/zenodo.15995036",
        "archive_name": "ai4mars-dataset-merged-0.6.zip",
        "total_file_entries": len(entries),
        "total_compressed_bytes": total_c,
        "total_uncompressed_bytes": total_u,
        "extensions": dict(ext_counts),
        "directories": dict(sorted(dir_stats.items())),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _build_download_batches(
    selected: List[ZipEntry],
    max_gap_bytes: int = 2_000_000,
    max_batch_bytes: int = 20_000_000,
) -> List[Tuple[int, int, List[ZipEntry]]]:
    """Merge nearby archive entries into contiguous multi-entry HTTP Range batches."""
    if not selected:
        return []
    ordered = sorted(selected, key=lambda e: e.offset)
    batches: List[Tuple[int, int, List[ZipEntry]]] = []
    cur_start = ordered[0].offset
    cur_end = ordered[0].end_offset
    cur_items = [ordered[0]]

    for entry in ordered[1:]:
        gap = entry.offset - cur_end - 1
        new_span = entry.end_offset - cur_start + 1
        if gap <= max_gap_bytes and new_span <= max_batch_bytes:
            cur_end = max(cur_end, entry.end_offset)
            cur_items.append(entry)
        else:
            batches.append((cur_start, cur_end, cur_items))
            cur_start = entry.offset
            cur_end = entry.end_offset
            cur_items = [entry]
    batches.append((cur_start, cur_end, cur_items))
    return batches


def _verify_existing_file(path: Path, expected_crc: int) -> bool:
    if not path.exists():
        return False
    return (zlib.crc32(path.read_bytes()) & 0xFFFFFFFF) == expected_crc


def _extract_entry_from_buffer(
    buf: bytes, buf_start: int, entry: ZipEntry, target_root: Path
) -> Tuple[bool, int]:
    """Extract and CRC-verify a single ZipEntry from an in-memory range buffer."""
    rel = entry.filename.removeprefix("ai4mars-dataset-merged-0.6/")
    out_path = target_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rel_off = entry.offset - buf_start
    header = buf[rel_off : rel_off + 30]
    if header[:4] != b"PK\x03\x04":
        raise ValueError(f"Bad local file header signature for {entry.filename}")
    _, _, _, comp_method, _, _, _, _, _, fn_len, ex_len = struct.unpack(
        "<4sHHHHHIIIHH", header
    )
    data_start = rel_off + 30 + fn_len + ex_len
    payload = buf[data_start : data_start + entry.comp_size]

    if comp_method == 0:
        raw_bytes = payload
    elif comp_method == 8:
        raw_bytes = zlib.decompress(payload, -15)
    else:
        raise ValueError(f"Unsupported compression method {comp_method} for {entry.filename}")

    actual_crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
    if actual_crc != entry.crc32:
        raise ValueError(
            f"CRC32 mismatch for {entry.filename}: got {actual_crc:#x}, expected {entry.crc32:#x}"
        )

    out_path.write_bytes(raw_bytes)
    return True, len(raw_bytes)


def download_entries(
    entries: Iterable[ZipEntry],
    target_root: Path,
    url: str = ZENODO_V06_URL,
    max_workers: int = 6,
    desc: str = "Downloading entries",
) -> Tuple[int, int]:
    """Download, decompress, CRC-verify, and write selected ZIP entries."""
    pending: List[ZipEntry] = []
    skipped = 0
    for entry in entries:
        rel = entry.filename.removeprefix("ai4mars-dataset-merged-0.6/")
        out_path = target_root / rel
        if _verify_existing_file(out_path, entry.crc32):
            skipped += 1
        else:
            pending.append(entry)

    if not pending:
        LOGGER.info("%s: all %d files already present and CRC-verified.", desc, skipped)
        return skipped, 0

    batches = _build_download_batches(pending)
    LOGGER.info(
        "%s: %d files to fetch (%d already verified) across %d range batches",
        desc,
        len(pending),
        skipped,
        len(batches),
    )

    thread_local = threading.local()

    def get_session() -> requests.Session:
        if not hasattr(thread_local, "session"):
            thread_local.session = _create_session()
        return thread_local.session

    def process_batch(batch: Tuple[int, int, List[ZipEntry]]) -> Tuple[int, int]:
        b_start, b_end, b_entries = batch
        buf = fetch_range(get_session(), url, b_start, b_end)
        written_count = 0
        written_bytes = 0
        for entry in b_entries:
            _, nbytes = _extract_entry_from_buffer(buf, b_start, entry, target_root)
            written_count += 1
            written_bytes += nbytes
        return written_count, written_bytes

    total_written = 0
    total_bytes = 0
    with tqdm(total=len(pending), desc=desc, unit="file") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            for fut in concurrent.futures.as_completed(futures):
                cnt, nbytes = fut.result()
                total_written += cnt
                total_bytes += nbytes
                pbar.update(cnt)

    return skipped + total_written, total_bytes


def extract_sclk(stem: str) -> int:
    """Extract integer Spacecraft Clock (SCLK) timestamp from MSL/MER filename."""
    # Example MSL: NLA_397586934EDR_F0010008AUT_04096M1 -> 397586934
    match = re.search(r"^[A-Za-z]{3}_(\d{9,10})", stem)
    if match:
        return int(match.group(1))
    match_mer = re.search(r"^[12][a-z](\d{9})", stem)
    if match_mer:
        return int(match_mer.group(1))
    return 0


def acquire_external_nasa_samples(sample_root: Path) -> Dict[str, object]:
    """Fetch small qualitative test samples from official NASA Raw Image APIs.

    Strictly stored outside `datasets/raw/ai4mars/` for qualitative testing only.
    """
    sample_root.mkdir(parents=True, exist_ok=True)
    session = _create_session()
    records = []

    # 1. Perseverance (M2020) NavCam / Mastcam-Z raw images
    try:
        resp = session.get(M2020_RAW_API, timeout=30)
        resp.raise_for_status()
        m2020_items = resp.json().get("images", [])
        # Prefer NAVCAM or MASTCAM-Z surface terrain imagery
        preferred = [
            it
            for it in m2020_items
            if any(
                cam in str(it.get("camera", {}).get("instrument", "")).upper()
                for cam in ("NAVCAM", "MCZ", "HAZCAM")
            )
        ] or m2020_items

        saved_m2020 = 0
        for item in preferred:
            if saved_m2020 >= 4:
                break
            img_files = item.get("image_files", {})
            img_url = img_files.get("large") or img_files.get("medium") or img_files.get("full_res")
            if not img_url:
                continue
            ext = Path(img_url.split("?")[0]).suffix or ".jpg"
            fname = f"m2020_sol{item.get('sol', 'unk')}_{item.get('imageid', saved_m2020)}{ext}"
            out_path = sample_root / fname
            if not out_path.exists():
                r_img = session.get(img_url, timeout=45)
                r_img.raise_for_status()
                out_path.write_bytes(r_img.content)
            records.append(
                {
                    "mission": "Mars 2020 (Perseverance)",
                    "filename": fname,
                    "sol": item.get("sol"),
                    "instrument": item.get("camera", {}).get("instrument"),
                    "imageid": item.get("imageid"),
                    "source_url": img_url,
                }
            )
            saved_m2020 += 1
    except Exception as exc:
        LOGGER.warning("Could not fetch M2020 external samples: %s", exc)

    # 2. Recent Curiosity (MSL) NavCam raw images
    try:
        resp = session.get(MSL_RAW_API, timeout=30)
        resp.raise_for_status()
        msl_items = resp.json().get("items", [])
        nav_items = [
            it
            for it in msl_items
            if "NAV" in str(it.get("instrument", "")).upper()
            or "MAST" in str(it.get("instrument", "")).upper()
        ] or msl_items

        saved_msl = 0
        for item in nav_items:
            if saved_msl >= 4:
                break
            img_url = item.get("url")
            if not img_url:
                continue
            orig_name = Path(img_url.split("?")[0]).name
            fname = f"msl_sol{item.get('sol', 'unk')}_{orig_name}"
            out_path = sample_root / fname
            if not out_path.exists():
                r_img = session.get(img_url, timeout=45)
                r_img.raise_for_status()
                out_path.write_bytes(r_img.content)
            records.append(
                {
                    "mission": "MSL (Curiosity - Recent Raw API)",
                    "filename": fname,
                    "sol": item.get("sol"),
                    "instrument": item.get("instrument"),
                    "imageid": item.get("id"),
                    "source_url": img_url,
                }
            )
            saved_msl += 1
    except Exception as exc:
        LOGGER.warning("Could not fetch MSL external samples: %s", exc)

    meta = {
        "purpose": "Qualitative external generalization test samples (UNLABELED; DO NOT MIX INTO TRAINING)",
        "count": len(records),
        "samples": records,
    }
    (sample_root / "external_samples_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def populate_quick_samples(ai4mars_root: Path, samples_root: Path, split_manifest: dict) -> None:
    """Copy a small curated set of AI4MARS image-mask pairs into datasets/samples/."""
    samples_root.mkdir(parents=True, exist_ok=True)
    # Pick 4 train samples and 4 test samples (2 MSL, 2 MER)
    chosen = (
        split_manifest["splits"]["train"][:2]
        + split_manifest["splits"]["val"][:2]
        + split_manifest["splits"]["test_msl"][:2]
        + split_manifest["splits"]["test_mer"][:2]
    )
    for item in chosen:
        img_src = ai4mars_root / item["image_relpath"]
        mask_src = ai4mars_root / item["mask_relpath"]
        if img_src.exists() and mask_src.exists():
            shutil.copy2(img_src, samples_root / img_src.name)
            shutil.copy2(mask_src, samples_root / mask_src.name)


def run_acquisition(
    project_root: Path,
    cd_cache_path: Path,
    train_val_limit: int = 1800,
    train_count: int = 1500,
    max_workers: int = 6,
) -> Dict[str, object]:
    """Execute end-to-end official dataset acquisition and verification."""
    raw_root = project_root / "datasets" / "raw" / "ai4mars"
    raw_root.mkdir(parents=True, exist_ok=True)

    session = _create_session()
    head = session.head(ZENODO_V06_URL, allow_redirects=True, timeout=30)
    head.raise_for_status()
    archive_size = int(head.headers["Content-Length"])

    cd_bytes = ensure_central_directory(cd_cache_path, ZENODO_V06_URL)
    entries = parse_central_directory(cd_bytes, archive_size)
    LOGGER.info("Parsed %d file entries from official v0.6 archive.", len(entries))

    save_archive_manifest_summary(
        entries, raw_root / "official_v0.6_archive_manifest_summary.json"
    )

    by_dir: Dict[str, Dict[str, ZipEntry]] = collections.defaultdict(dict)
    doc_entries: List[ZipEntry] = []

    for entry in entries:
        rel = entry.filename.removeprefix("ai4mars-dataset-merged-0.6/")
        parent = rel.rsplit("/", 1)[0] if "/" in rel else ""
        fname = rel.rsplit("/", 1)[-1]
        by_dir[parent][fname] = entry
        if fname.endswith((".md", ".txt", ".json", ".csv")):
            doc_entries.append(entry)

    # 1. Documentation & Metadata files
    download_entries(doc_entries, raw_root, max_workers=4, desc="Docs & Metadata")

    # 2. Official Test Sets (MSL Gold + MER Gold)
    msl_test_min3 = by_dir["msl/ncam/labels/test/masked-gold-min3-100agree"]
    msl_test_min2 = by_dir["msl/ncam/labels/test/masked-gold-min2-100agree"]
    msl_test_min1 = by_dir["msl/ncam/labels/test/masked-gold-min1-100agree"]
    msl_edr = by_dir["msl/ncam/images/edr"]
    msl_mxy = by_dir["msl/ncam/images/mxy"]
    msl_rng = by_dir["msl/ncam/images/rng-30m"]
    msl_train_lbls = by_dir["msl/ncam/labels/train"]

    mer_test_imgs = by_dir["mer/images/test"]
    mer_test_min3 = by_dir["mer/labels/test/masked-gold-min3-100agree"]
    mer_test_min2 = by_dir["mer/labels/test/masked-gold-min2-100agree"]
    mer_test_min1 = by_dir["mer/labels/test/masked-gold-min1-100agree"]

    # Map stem -> entry
    edr_by_stem = {k.rsplit(".", 1)[0]: v for k, v in msl_edr.items()}
    mxy_by_stem = {
        k.replace("MXY_", "EDR_").rsplit(".", 1)[0]: v for k, v in msl_mxy.items()
    }
    rng_by_stem = {
        k.replace("RNG_", "EDR_").rsplit(".", 1)[0]: v for k, v in msl_rng.items()
    }
    train_by_stem = {k.rsplit(".", 1)[0]: v for k, v in msl_train_lbls.items()}
    msl_test_stems = {
        k.replace("_merged.png", "") for k in msl_test_min3.keys() if k.endswith(".png")
    }

    # Gather MSL Test assets (322 images + all min1/min2/min3 masks + mxy + rng)
    msl_test_entries: List[ZipEntry] = []
    msl_test_entries.extend(msl_test_min1.values())
    msl_test_entries.extend(msl_test_min2.values())
    msl_test_entries.extend(msl_test_min3.values())
    msl_test_pairs: List[Dict[str, str]] = []
    for stem in sorted(msl_test_stems):
        if stem in edr_by_stem:
            msl_test_entries.append(edr_by_stem[stem])
            if stem in mxy_by_stem:
                msl_test_entries.append(mxy_by_stem[stem])
            if stem in rng_by_stem:
                msl_test_entries.append(rng_by_stem[stem])
            msl_test_pairs.append(
                {
                    "id": stem,
                    "mission": "MSL (Curiosity)",
                    "sclk": extract_sclk(stem),
                    "image_relpath": f"msl/ncam/images/edr/{edr_by_stem[stem].filename.rsplit('/', 1)[-1]}",
                    "mask_relpath": f"msl/ncam/labels/test/masked-gold-min3-100agree/{stem}_merged.png",
                    "mask_min2_relpath": f"msl/ncam/labels/test/masked-gold-min2-100agree/{stem}_merged.png",
                    "mask_min1_relpath": f"msl/ncam/labels/test/masked-gold-min1-100agree/{stem}_merged.png",
                    "mxy_relpath": f"msl/ncam/images/mxy/{mxy_by_stem[stem].filename.rsplit('/', 1)[-1]}" if stem in mxy_by_stem else "",
                    "rng_relpath": f"msl/ncam/images/rng-30m/{rng_by_stem[stem].filename.rsplit('/', 1)[-1]}" if stem in rng_by_stem else "",
                }
            )

    # Gather MER Test assets (204 images + min1/min2/min3 masks)
    mer_test_entries: List[ZipEntry] = []
    mer_test_entries.extend(mer_test_imgs.values())
    mer_test_entries.extend(mer_test_min1.values())
    mer_test_entries.extend(mer_test_min2.values())
    mer_test_entries.extend(mer_test_min3.values())

    mer_img_by_stem = {k.rsplit(".", 1)[0]: v for k, v in mer_test_imgs.items()}
    mer_min3_by_stem = {
        k.split("_")[0]: v for k, v in mer_test_min3.items() if k.endswith(".png")
    }
    mer_min2_by_stem = {
        k.split("_")[0]: v for k, v in mer_test_min2.items() if k.endswith(".png")
    }
    mer_min1_by_stem = {
        k.split("_")[0]: v for k, v in mer_test_min1.items() if k.endswith(".png")
    }

    mer_test_pairs: List[Dict[str, str]] = []
    for stem in sorted(mer_img_by_stem.keys()):
        if stem in mer_min3_by_stem:
            rover = "MER-1 (Opportunity)" if stem.startswith("1") else "MER-2 (Spirit)"
            mer_test_pairs.append(
                {
                    "id": stem,
                    "mission": rover,
                    "sclk": extract_sclk(stem),
                    "image_relpath": f"mer/images/test/{mer_img_by_stem[stem].filename.rsplit('/', 1)[-1]}",
                    "mask_relpath": f"mer/labels/test/masked-gold-min3-100agree/{mer_min3_by_stem[stem].filename.rsplit('/', 1)[-1]}",
                    "mask_min2_relpath": f"mer/labels/test/masked-gold-min2-100agree/{mer_min2_by_stem[stem].filename.rsplit('/', 1)[-1]}" if stem in mer_min2_by_stem else "",
                    "mask_min1_relpath": f"mer/labels/test/masked-gold-min1-100agree/{mer_min1_by_stem[stem].filename.rsplit('/', 1)[-1]}" if stem in mer_min1_by_stem else "",
                }
            )

    # 3. Gather 1,800 contiguous MSL NavCam Train/Val pairs (strictly disjoint from MSL Test)
    edr_sorted_by_offset = sorted(msl_edr.values(), key=lambda e: e.offset)
    selected_train_stems: List[str] = []
    train_val_entries: List[ZipEntry] = []
    for edr_entry in edr_sorted_by_offset:
        stem = edr_entry.filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if (
            stem in train_by_stem
            and stem not in msl_test_stems
            and stem in mxy_by_stem
            and stem in rng_by_stem
        ):
            selected_train_stems.append(stem)
            train_val_entries.append(edr_entry)
            train_val_entries.append(train_by_stem[stem])
            train_val_entries.append(mxy_by_stem[stem])
            train_val_entries.append(rng_by_stem[stem])
            if len(selected_train_stems) >= train_val_limit:
                break

    # Chronological SCLK sort to prevent temporal leakage between train and val
    train_val_records = [
        {
            "id": stem,
            "mission": "MSL (Curiosity)",
            "sclk": extract_sclk(stem),
            "image_relpath": f"msl/ncam/images/edr/{edr_by_stem[stem].filename.rsplit('/', 1)[-1]}",
            "mask_relpath": f"msl/ncam/labels/train/{train_by_stem[stem].filename.rsplit('/', 1)[-1]}",
            "mxy_relpath": f"msl/ncam/images/mxy/{mxy_by_stem[stem].filename.rsplit('/', 1)[-1]}",
            "rng_relpath": f"msl/ncam/images/rng-30m/{rng_by_stem[stem].filename.rsplit('/', 1)[-1]}",
        }
        for stem in selected_train_stems
    ]
    train_val_records.sort(key=lambda r: (r["sclk"], r["id"]))
    train_records = train_val_records[:train_count]
    val_records = train_val_records[train_count:]

    # Deduplicate entries by filename before downloading
    all_target_map: Dict[str, ZipEntry] = {}
    for e in msl_test_entries + mer_test_entries + train_val_entries:
        all_target_map[e.filename] = e

    # Download grouped by type/contiguity for optimal speed
    msl_test_edr_only = [e for e in msl_test_entries if "/edr/" in e.filename]
    msl_test_masks_only = [e for e in msl_test_entries if "/edr/" not in e.filename]
    tv_edr_only = [e for e in train_val_entries if "/edr/" in e.filename]
    tv_masks_only = [e for e in train_val_entries if "/edr/" not in e.filename]

    all_edr_entries = list(
        {e.filename: e for e in (tv_edr_only + msl_test_edr_only)}.values()
    )
    all_msl_mask_entries = list(
        {e.filename: e for e in (tv_masks_only + msl_test_masks_only)}.values()
    )

    download_entries(
        all_msl_mask_entries,
        raw_root,
        max_workers=max_workers,
        desc="MSL Labels & Aux Masks",
    )
    download_entries(
        all_edr_entries,
        raw_root,
        max_workers=max_workers,
        desc="MSL NavCam EDR Images",
    )
    download_entries(
        mer_test_entries,
        raw_root,
        max_workers=max_workers,
        desc="MER Gold Test Images & Masks",
    )

    split_manifest = {
        "dataset": "NASA/JPL AI4MARS v0.6",
        "doi": "10.5281/zenodo.15995036",
        "split_strategy": {
            "train_val": "Chronological Spacecraft Clock (SCLK) split of 1,800 MSL NavCam labeled pairs (first 1,500 earliest SCLK -> train; latest 300 SCLK -> val) to prevent temporal scene leakage.",
            "test": "Official JPL Expert Gold-Standard Test Sets (322 MSL NavCam + 204 MER NavCam, 100% consensus labels min1/min2/min3), strictly held out.",
        },
        "counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test_msl": len(msl_test_pairs),
            "test_mer": len(mer_test_pairs),
            "total_pairs": len(train_records)
            + len(val_records)
            + len(msl_test_pairs)
            + len(mer_test_pairs),
        },
        "splits": {
            "train": train_records,
            "val": val_records,
            "test_msl": msl_test_pairs,
            "test_mer": mer_test_pairs,
        },
    }
    manifest_path = raw_root / "split_manifest.json"
    manifest_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    # Populate quick reference samples and external NASA raw API samples
    populate_quick_samples(raw_root, project_root / "datasets" / "samples", split_manifest)
    ext_meta = acquire_external_nasa_samples(
        project_root / "datasets" / "samples" / "external_nasa"
    )

    return {
        "split_counts": split_manifest["counts"],
        "external_samples": ext_meta["count"],
        "manifest_path": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire NASA/JPL AI4MARS v0.6 dataset subset and official test sets."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Root directory of MANGAL DRISHTI repository.",
    )
    parser.add_argument(
        "--cd-cache",
        type=Path,
        default=Path(
            "C:/Users/ASUS/.gemini/antigravity/brain/bdfc099c-444d-47b7-af01-8ddfc5b09bd8/scratch_cd.bin"
        ),
        help="Optional local cache path for Zip64 Central Directory.",
    )
    parser.add_argument("--train-val-limit", type=int, default=1800)
    parser.add_argument("--train-count", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    cd_path = (
        args.cd_cache
        if args.cd_cache.exists()
        else args.project_root / "datasets" / "raw" / "ai4mars" / ".cd_cache.bin"
    )
    res = run_acquisition(
        project_root=args.project_root,
        cd_cache_path=cd_path,
        train_val_limit=args.train_val_limit,
        train_count=args.train_count,
        max_workers=args.workers,
    )
    LOGGER.info("Acquisition complete: %s", json.dumps(res))


if __name__ == "__main__":
    main()
