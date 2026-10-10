"""Edit-pair cache: geometry, on-disk format, dataset and sampler.

Iris works on pixels, so a cache item is just the two images, already sized:
`<name>.src.jpg` and `<name>.tgt.jpg` (quality 95, no chroma subsampling) plus one row
in `index.jsonl` with the instruction and size. The target keeps its aspect ratio at
an area of ~size^2 on the 16-pixel patch grid; the reference is center-cropped onto
the same grid so its tokens line up with the target's.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

INDEX = "index.jsonl"
JPEG = {"quality": 95, "subsampling": 0}


def ar_dims(w: int, h: int, side: int, multiple: int = 16) -> tuple[int, int]:
    """(W, H) with the aspect ratio of w/h, area ~ side^2, both a multiple of `multiple`."""
    th = (side * side * h / w) ** 0.5
    tw = th * w / h
    return max(multiple, round(tw / multiple) * multiple), max(multiple, round(th / multiple) * multiple)


def center_crop_resize(img: Image.Image, tw: int, th: int) -> Image.Image:
    img = img.convert("RGB")
    scale = max(tw / img.width, th / img.height)
    cw, ch = round(tw / scale), round(th / scale)
    left, top = (img.width - cw) // 2, (img.height - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((tw, th), Image.LANCZOS)


def fit_pair(target: Image.Image, source: Image.Image, side: int) -> tuple[Image.Image, Image.Image]:
    tw, th = ar_dims(target.width, target.height, side)
    return center_crop_resize(target, tw, th), center_crop_resize(source, tw, th)


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> [3, H, W] float in [-1, 1]."""
    x = torch.from_numpy(np.asarray(img.convert("RGB")).copy()).permute(2, 0, 1).float()
    return x / 127.5 - 1.0


def to_pil(x: torch.Tensor) -> Image.Image:
    """[3, H, W] in [-1, 1] -> PIL."""
    x = ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def read_index(cache_dir: str | Path) -> dict[str, dict]:
    path = Path(cache_dir) / INDEX
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["n"]] = row
    return rows


_INDEX_LOCK = threading.Lock()


def write_item(cache_dir: Path, name: str, source: Image.Image, target: Image.Image, instruction: str):
    source.save(cache_dir / f"{name}.src.jpg", **JPEG)
    target.save(cache_dir / f"{name}.tgt.jpg", **JPEG)
    line = json.dumps({"n": name, "i": instruction, "s": [target.width, target.height]}) + "\n"
    with _INDEX_LOCK:  # one O_APPEND write per line, so concurrent processes don't interleave
        fd = os.open(cache_dir / INDEX, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)


class EditPairs(Dataset):
    def __init__(self, cache_dir: str | Path, min_alignment: float | None = None):
        self.dir = Path(cache_dir)
        rows = read_index(self.dir)
        self.rows = [r for r in rows.values()
                     if (self.dir / f"{r['n']}.tgt.jpg").exists() and (self.dir / f"{r['n']}.src.jpg").exists()]
        if not self.rows:
            raise SystemExit(f"[data] no cached pairs in {self.dir}; run `irisedit cache <data-config>` first")
        self.dropped = 0
        if min_alignment is not None:
            from .align import score_cache

            scores = score_cache(self.dir, [r["n"] for r in self.rows])
            kept = [r for r in self.rows if scores[r["n"]] >= min_alignment]
            self.dropped = len(self.rows) - len(kept)
            self.rows = kept

    def __len__(self):
        return len(self.rows)

    def shapes(self):
        return [tuple(r["s"]) for r in self.rows]

    def instructions(self):
        return [r["i"] for r in self.rows]

    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "source": to_tensor(Image.open(self.dir / f"{r['n']}.src.jpg")),
            "target": to_tensor(Image.open(self.dir / f"{r['n']}.tgt.jpg")),
            "instruction": r["i"],
        }


def collate(batch):
    return {
        "source": torch.stack([b["source"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "instruction": [b["instruction"] for b in batch],
    }


class ShapeBucketSampler(Sampler):
    """Infinite shuffled batches whose items share a size. `start_index` (= resumed
    step * batch size) continues through the data instead of replaying it."""

    def __init__(self, shapes, batch_size: int, seed: int = 0, start_index: int = 0):
        self.shapes = list(shapes)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.start_index = int(start_index)

    def __iter__(self):
        n = len(self.shapes)
        index = self.start_index
        while True:
            epoch = index // n
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
            pools: dict[tuple, list[int]] = {}
            for offset in range(index - epoch * n, n):
                i = perm[offset]
                pool = pools.setdefault(self.shapes[i], [])
                pool.append(i)
                if len(pool) == self.batch_size:
                    yield list(pool)
                    pool.clear()
            index = (epoch + 1) * n

    def __len__(self):
        return max(1, len(self.shapes) // self.batch_size)
