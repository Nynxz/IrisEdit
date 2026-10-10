"""Structural alignment of cached pairs.

An instruction edit should keep the source's composition, even when colours or style
change completely. Alignment is the correlation of edge maps of 32x32 grayscale
thumbnails: recolours, weather and style edits keep it high, a redrawn scene drops it to
~0. HQ-Edit, whose before/after images were generated separately, scores ~0 throughout.
Scores are written once to `<cache>/alignment.json`; training drops pairs below
`min_alignment`.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

FILE = "alignment.json"


def _edges(path: Path) -> np.ndarray:
    x = np.asarray(Image.open(path).convert("L").resize((32, 32), Image.BOX), dtype=np.float32)
    e = np.concatenate([np.diff(x, axis=0).ravel(), np.diff(x, axis=1).ravel()])
    return e - e.mean()


def _score(args) -> tuple[str, float]:
    cache_dir, name = args
    a, b = _edges(cache_dir / f"{name}.src.jpg"), _edges(cache_dir / f"{name}.tgt.jpg")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return name, float(a @ b / denom) if denom > 0 else 0.0


def load(cache_dir: Path) -> dict[str, float]:
    path = Path(cache_dir) / FILE
    return json.loads(path.read_text()) if path.exists() else {}


def score_cache(cache_dir: Path, names: list[str], workers: int = 16) -> dict[str, float]:
    """Score `names` not yet in alignment.json, append, and return all scores."""
    cache_dir = Path(cache_dir)
    scores = load(cache_dir)
    todo = [n for n in names if n not in scores]
    if todo:
        with ProcessPoolExecutor(workers) as pool:
            scores.update(pool.map(_score, [(cache_dir, n) for n in todo], chunksize=256))
        tmp = cache_dir / (FILE + ".tmp")
        tmp.write_text(json.dumps(scores))
        tmp.replace(cache_dir / FILE)
    return scores
