"""Progress grid: one row per preview edit, source then one column per checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELL = 256
LABEL = 34


def build(run_dir: Path, manifest: Path | None, out: Path) -> Path:
    steps = sorted(p for p in (run_dir / "samples").glob("step-*") if p.is_dir())
    if not steps:
        raise SystemExit(f"[grid] no previews in {run_dir / 'samples'}")
    names = sorted(p.name for p in steps[0].glob("*.jpg"))
    instructions = []
    if manifest and manifest.exists():
        instructions = [json.loads(line)["instruction"] for line in manifest.read_text().splitlines() if line.strip()]

    rows = []
    for i, name in enumerate(names):
        first = Image.open(steps[0] / name)
        w, h = first.width // 2, first.height
        scale = CELL / max(w, h)
        size = (round(w * scale), round(h * scale))
        cells = [first.crop((0, 0, w, h)).resize(size, Image.LANCZOS)]
        for step in steps:
            path = step / name
            cells.append(Image.open(path).crop((w, 0, 2 * w, h)).resize(size, Image.LANCZOS) if path.exists()
                         else Image.new("RGB", size, "white"))
        rows.append((instructions[i] if i < len(instructions) else name, cells))

    cols = 1 + len(steps)
    width = cols * CELL
    height = LABEL + sum(LABEL + CELL for _ in rows)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    for c, title in enumerate(["source"] + [f"step {int(s.name.split('-')[1])}" for s in steps]):
        draw.text((c * CELL + 8, 6), title, fill="black", font=font)
    y = LABEL
    for text, cells in rows:
        draw.rectangle((0, y, width, y + LABEL), fill=(235, 235, 235))
        draw.text((8, y + 6), f'"{text}"', fill="black", font=font)
        y += LABEL
        for c, cell in enumerate(cells):
            sheet.paste(cell, (c * CELL + (CELL - cell.width) // 2, y + (CELL - cell.height) // 2))
        y += CELL
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    return out
