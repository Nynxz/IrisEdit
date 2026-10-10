"""How well an adapter predicts the clean target from the reference, per noise level.

For a few cached pairs, noise the target to each sigma, run one forward pass and report
the PSNR of the implied clean image (x_t - sigma * v) against the true target. At sigma 1
x_t holds nothing of the target, so that column measures how much of the edit comes from
the reference alone; it should at least reach the "copy the source" baseline. A model that
scores well at 0.9 but collapses at 1.0 reads the layout from x_t and redraws the scene when
sampling from noise.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import torch
from PIL import Image

from . import lora
from . import model as iris
from .align import load as load_alignment
from .config import load as load_config
from .data import read_index, to_tensor
from .paths import resolve_weights
from .text import TextEncoder

SIGMAS = (1.0, 0.99, 0.97, 0.9, 0.6, 0.2)


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(10 * torch.log10(4 / ((a.float() - b.float()) ** 2).mean()))


def run(loras: list[str], cache_dir: str, count: int = 8, seed: int = 0, min_alignment: float = 0.6):
    base_cfg = load_config("_base", strict=False)
    rows = list(read_index(cache_dir).values())
    scores = load_alignment(cache_dir)
    rows = [r for r in rows if scores.get(r["n"], 1.0) >= min_alignment]
    rows = random.Random(seed).sample(rows, min(count, len(rows)))
    encoder = TextEncoder(resolve_weights("text_encoder", base_cfg), device="cpu", dtype=torch.bfloat16)
    texts = [encoder([r["i"]]) for r in rows]
    del encoder
    base = iris.load_iris(resolve_weights("checkpoint", base_cfg), frame_pairs=base_cfg.get("frame_pairs", 2))
    d = Path(cache_dir)
    copy_db = sum(_psnr(to_tensor(Image.open(d / f"{r['n']}.src.jpg")), to_tensor(Image.open(d / f"{r['n']}.tgt.jpg")))
                  for r in rows) / len(rows)
    print(f"[probe] {len(rows)} pairs; copy-the-source baseline {copy_db:.1f} dB")
    print("[probe] " + "adapter".ljust(28) + "".join(f"s={s:<6}" for s in SIGMAS))
    for path in [None, *loras]:
        model = copy.deepcopy(base)
        if path:
            lora.inject_from_file(model, path)
        model.cuda()
        totals = [0.0] * len(SIGMAS)
        for r, (emb, mask) in zip(rows, texts):
            src = to_tensor(Image.open(d / f"{r['n']}.src.jpg"))[None].cuda().bfloat16()
            tgt = to_tensor(Image.open(d / f"{r['n']}.tgt.jpg"))[None].cuda().bfloat16()
            noise = torch.randn(tgt.shape, generator=torch.Generator("cuda").manual_seed(seed), device="cuda")
            for i, s in enumerate(SIGMAS):
                x_t = (1 - s) * tgt + s * noise.bfloat16()
                with torch.no_grad():
                    v = iris.forward(model, x_t, [src], torch.tensor([s * 1000.0], device="cuda"),
                                     emb.cuda().bfloat16(), mask.cuda())
                totals[i] += _psnr(x_t - s * v, tgt) / len(rows)
        name = "base (no LoRA)" if path is None else str(path)[-28:]
        print("[probe] " + name.ljust(28) + "".join(f"{t:<8.1f}" for t in totals), flush=True)
        del model
        torch.cuda.empty_cache()
