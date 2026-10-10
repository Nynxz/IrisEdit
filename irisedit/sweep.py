"""Render the preview manifest at several CFG scales for one adapter, as a grid."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import config as configlib
from . import lora
from . import model as iris
from .data import ar_dims, to_pil, to_tensor
from .paths import resolve_weights
from .sample import edit
from .text import TextEncoder

CELL = 320


def run(lora_path: str, manifest: str, cfgs: list[float], side: int, steps: int, shift: float, seed: int,
        out: Path) -> Path:
    base = configlib.load("_base", strict=False)
    model = iris.load_iris(resolve_weights("checkpoint", base), frame_pairs=base.get("frame_pairs", 2))
    lora.inject_from_file(model, lora_path)
    model.cuda()
    encoder = TextEncoder(resolve_weights("text_encoder", base))
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]

    font = ImageFont.load_default(size=20)
    label = 34
    sheet = Image.new("RGB", ((1 + len(cfgs)) * CELL, label + len(rows) * (label + CELL)), "white")
    draw = ImageDraw.Draw(sheet)
    for c, title in enumerate(["source"] + [f"cfg {g:g}" for g in cfgs]):
        draw.text((c * CELL + 8, 6), title, fill="black", font=font)
    y = label
    for row in rows:
        img = Image.open(row["image"]).convert("RGB")
        img = img.resize(ar_dims(img.width, img.height, side), Image.LANCZOS)
        cells = [img] + [to_pil(edit(model, encoder, to_tensor(img), row["instruction"], steps=steps, cfg=g,
                                     shift=shift, seed=seed)) for g in cfgs]
        draw.rectangle((0, y, sheet.width, y + label), fill=(235, 235, 235))
        draw.text((8, y + 6), f'"{row["instruction"]}"', fill="black", font=font)
        y += label
        for c, cell in enumerate(cells):
            scale = CELL / max(cell.size)
            cell = cell.resize((round(cell.width * scale), round(cell.height * scale)), Image.LANCZOS)
            sheet.paste(cell, (c * CELL + (CELL - cell.width) // 2, y + (CELL - cell.height) // 2))
        y += CELL
        print(f"[sweep] {row['instruction']}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    return out
