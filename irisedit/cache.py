"""Stream Hugging Face edit datasets into a pixel cache (see irisedit/data.py).

A data config lists sources with column names, a per-source count and optional
filters (score, type whitelist, per-category cap) and `interleave: N` for datasets stored
sorted by category. Filters run on plain columns before any image is decoded, so rejected rows
cost only their download. Resumable: a per-source cursor records how far into the
stream we got, and items already in the index are not rewritten.
"""
from __future__ import annotations

import io
import json
import time
from collections import Counter
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .data import fit_pair, read_index, write_item



def first_str(v) -> str:
    if isinstance(v, (list, tuple)):
        return next((e.strip() for e in v if isinstance(e, str) and e.strip()), "")
    return str(v).strip() if v is not None else ""


def decode(value):
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])
    return None


# A streaming worker that holds more than this is stopped (cursor saved) and resumed by the
# next run; seen in practice on a resumed interleaved OmniEdit stream (79GB before the OOM kill).
RSS_LIMIT_GB = 20


class MemoryGuard(Exception):
    pass


def _rss_gb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30


def _cursor_path(out_dir: Path, key: str) -> Path:
    return out_dir / f".cursor_{key.replace('/', '_').replace('|', '_')}.json"


def _cursor(out_dir: Path, key: str) -> dict:
    try:
        return json.loads(_cursor_path(out_dir, key).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cursor(out_dir: Path, key: str, streamed: int, kept: int):
    path = _cursor_path(out_dir, key)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"streamed": streamed, "kept": kept}))
    tmp.replace(path)


def _process(out_dir: Path, name: str, src, tgt, instruction: str, size: int) -> bool:
    try:
        src_img, tgt_img = decode(src), decode(tgt)
        if src_img is None or tgt_img is None:
            return False
        tgt_img, src_img = fit_pair(tgt_img, src_img, size)
        write_item(out_dir, name, src_img, tgt_img, instruction)
        return True
    except Exception:
        return False


def cache_source(source: dict, out_dir: Path, size: int, pool: ThreadPoolExecutor, fresh_cursor: bool):
    from datasets import Image as HFImage
    from datasets import load_dataset

    repo, split = source["repo"], source.get("split", "train")
    src_k, tgt_k, ins_k = source["src"], source["tgt"], source["instr"]
    count = int(source["count"])
    prefix = source.get("prefix") or repo.split("/")[-1].lower()
    types = {str(t).strip().lower() for t in source.get("types") or []}
    type_field = source.get("type_field")
    score_field, min_score = source.get("score_field"), source.get("min_score")
    balance_field, max_per_type = source.get("balance_field"), source.get("max_per_type")
    key = f"{repo}|{split}|{prefix}"

    indexed = read_index(out_dir)
    cursor = {} if fresh_cursor else _cursor(out_dir, key)
    streamed, idx = int(cursor.get("streamed", 0)), int(cursor.get("kept", 0))
    if idx >= count:
        print(f"[cache] {repo}: {idx}/{count} already cached", flush=True)
        return 0

    try:
        ds = load_dataset(repo, split=split, streaming=True)
        if source.get("interleave"):
            # datasets stored sorted by category (OmniEdit: one task per shard range) would
            # give only the first category; draw round-robin from evenly spaced shards instead
            from datasets import interleave_datasets

            n, k = ds.num_shards, min(int(source["interleave"]), ds.num_shards)
            picks = sorted({round(i * (n - 1) / max(k - 1, 1)) for i in range(k)})
            ds = interleave_datasets([ds.shard(num_shards=n, index=i) for i in picks],
                                     stopping_strategy="all_exhausted")
        for col in (src_k, tgt_k):
            try:
                ds = ds.cast_column(col, HFImage(decode=False))
            except Exception:
                pass
        if streamed:
            ds = ds.skip(streamed)
    except Exception as e:
        print(f"[cache] SKIP {repo}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return 0

    print(f"[cache] {repo}: from row {streamed:,}, {idx}/{count} kept", flush=True)
    seen_per_type: Counter = Counter()
    pending, kept, t0 = [], 0, time.monotonic()
    for row in ds:
        if idx >= count:
            break
        streamed += 1
        if types and str(row.get(type_field, "")).strip().lower() not in types:
            continue
        if score_field and min_score is not None:
            score = row.get(score_field)
            if score is None or float(score) < float(min_score):
                continue
        bucket = None
        if balance_field and max_per_type:
            bucket = str(row.get(balance_field, ""))
            if seen_per_type[bucket] >= int(max_per_type):
                continue
        instruction = first_str(row.get(ins_k))
        if len(instruction) < 3:
            continue
        name = f"{prefix}_{idx:07d}"
        idx += 1
        if bucket is not None:
            seen_per_type[bucket] += 1
        if name in indexed:
            continue
        pending.append(pool.submit(_process, out_dir, name, row.get(src_k), row.get(tgt_k), instruction, size))
        if len(pending) >= 64:
            kept += sum(f.result() for f in pending)
            pending.clear()
            _save_cursor(out_dir, key, streamed, idx)
        if streamed % 200 == 0 and _rss_gb() > RSS_LIMIT_GB:
            kept += sum(f.result() for f in pending)
            pending.clear()
            _save_cursor(out_dir, key, streamed, idx)
            print(f"[cache] {repo}: memory {_rss_gb():.0f}GB > {RSS_LIMIT_GB}GB, stopping at row {streamed:,} "
                  "(rerun resumes)", flush=True)
            raise MemoryGuard
        if streamed % 2000 == 0:
            print(f"[cache] {repo}: {idx}/{count} ({streamed:,} streamed, {time.monotonic() - t0:.0f}s)", flush=True)
    kept += sum(f.result() for f in pending)
    _save_cursor(out_dir, key, streamed, idx)
    print(f"[cache] {repo}: wrote {kept} pairs in {time.monotonic() - t0:.0f}s ({streamed:,} rows streamed)", flush=True)
    return kept


def _cache_one(source: dict, out_dir: Path, size: int, workers: int, fresh_cursor: bool):
    """Child-process entry: cache one source, then exit without interpreter teardown.

    `datasets` streaming can crash while its background threads are torn down
    (PyGILState_Release at finalization). Progress is already on disk (index + cursor), so
    skipping teardown with os._exit loses nothing and keeps one source's crash from taking
    the others with it."""
    code = 0
    try:
        with ThreadPoolExecutor(workers) as pool:
            cache_source(source, out_dir, size, pool, fresh_cursor)
    except MemoryGuard:
        code = 3
    except Exception as e:
        print(f"[cache] {source['repo']}: failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
        code = 1
    os._exit(code)


def run(cfg: dict, out_dir: Path, workers: int = 16, fresh_cursor: bool = False):
    """One independent process per source (HF streaming is network-bound per stream and not
    safe to share across threads), each with its own image-worker threads. A source that dies
    is reported and the rest carry on; rerunning resumes everything from the cursors."""
    out_dir.mkdir(parents=True, exist_ok=True)
    size = int(cfg.get("size", 512))
    sources = cfg["sources"]
    per = max(2, workers // len(sources))
    before = len(read_index(out_dir))
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_cache_one, args=(src, out_dir, size, per, fresh_cursor)) for src in sources]
    for p in procs:
        p.start()
    for src, p in zip(sources, procs):
        p.join()
        if p.exitcode:
            print(f"[cache] {src['repo']}: worker exited with code {p.exitcode} (rerun to resume)", flush=True)
    after = len(read_index(out_dir))
    print(f"[cache] done: {after - before} new pairs, {after} in {out_dir}", flush=True)
