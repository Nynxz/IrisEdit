"""`irisedit` command: cache, train, edit, status."""
from __future__ import annotations

from pathlib import Path

import click

from . import config as configlib


@click.group()
def cli():
    """Train Iris-3B into an instruction image editor."""


@cli.command()
@click.argument("data_config")
@click.option("--out", "out_dir", type=click.Path(path_type=Path), help="cache dir (default: data/cache_<name>)")
@click.option("--workers", default=16, show_default=True, help="threads decoding/resizing/writing images")
@click.option("--fresh-cursor", is_flag=True, help="re-walk each source from row 0")
def cache(data_config, out_dir, workers, fresh_cursor):
    """Stream a data config's sources into a pixel cache."""
    from .cache import run

    cfg = configlib.load(data_config)
    suffix = cfg["name"].removeprefix("data").strip("_") or "main"
    run(cfg, out_dir or configlib.REPO / "data" / f"cache_{suffix}",
        workers=workers, fresh_cursor=fresh_cursor)


@cli.command()
@click.argument("run_config", default="generalist")
@click.option("--fresh", is_flag=True, help="ignore out_dir/state.pt and start at step 0")
def train(run_config, fresh):
    """Train (or resume) an edit LoRA."""
    from .train import run

    run(configlib.load(run_config), fresh=fresh)


@cli.command()
@click.argument("image", type=click.Path(exists=True))
@click.argument("instruction")
@click.option("--lora", "lora_path", required=True, type=click.Path(exists=True))
@click.option("--out", "out_path", default="out/edit.png", show_default=True)
@click.option("--side", default=512, show_default=True, help="sqrt of the output area")
@click.option("--steps", default=50, show_default=True)
@click.option("--cfg", "guidance", default=3.0, show_default=True)
@click.option("--shift", default=None, type=float, help="default: 3.5 at side<=768, else 4")
@click.option("--strength", default=1.0, show_default=True)
@click.option("--seed", default=0, show_default=True)
def edit(image, instruction, lora_path, out_path, side, steps, guidance, shift, strength, seed):
    """Edit one image with a trained LoRA."""
    from PIL import Image

    from . import lora
    from . import model as iris
    from .data import ar_dims, to_pil, to_tensor
    from .paths import resolve_weights
    from .sample import edit as run_edit
    from .text import TextEncoder

    base = configlib.load("_base", strict=False)
    model = iris.load_iris(resolve_weights("checkpoint", base), frame_pairs=base.get("frame_pairs", 2))
    lora.inject_from_file(model, lora_path)
    lora.set_strength(model, strength)
    model.cuda()
    encoder = TextEncoder(resolve_weights("text_encoder", base))
    img = Image.open(image).convert("RGB")
    img = img.resize(ar_dims(img.width, img.height, side), Image.LANCZOS)
    out = run_edit(model, encoder, to_tensor(img), instruction, steps=steps, cfg=guidance,
                   shift=shift or (3.5 if side <= 768 else 4.0), seed=seed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    to_pil(out).save(out_path)
    click.echo(f"wrote {out_path}")


@cli.command()
def status():
    """Caches and runs in this repo."""
    from .data import read_index

    for d in sorted((configlib.REPO / "data").glob("cache_*")):
        click.echo(f"cache {d.name}: {len(read_index(d))} pairs")
    for d in sorted((configlib.REPO / "out").glob("*/")):
        ckpts = sorted(d.glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
        click.echo(f"run {d.name}: {len(ckpts)} checkpoints" + (f", latest {ckpts[-1].name}" if ckpts else ""))


@cli.command()
@click.argument("run_config")
@click.option("--out", "out_path", default=None, help="default: <out_dir>/progress.jpg")
def grid(run_config, out_path):
    """Progress grid of a run's previews: source, then one column per checkpoint."""
    from .grid import build

    cfg = configlib.load(run_config)
    run_dir = Path(cfg["out_dir"])
    manifest = Path(cfg["preview_manifest"]) if cfg.get("preview_manifest") else None
    click.echo(f"wrote {build(run_dir, manifest, Path(out_path) if out_path else run_dir / 'progress.jpg')}")


@cli.command()
@click.argument("lora_path", type=click.Path(exists=True))
@click.option("--manifest", default="examples/preview.jsonl", show_default=True)
@click.option("--cfg", "cfgs", default="3,5,7", show_default=True, help="comma-separated CFG scales")
@click.option("--side", default=768, show_default=True)
@click.option("--steps", default=30, show_default=True)
@click.option("--shift", default=3.5, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--out", "out_path", default=None, help="default: next to the adapter, <name>_cfg_sweep.jpg")
def sweep(lora_path, manifest, cfgs, side, steps, shift, seed, out_path):
    """Render the preview edits at several CFG scales for one adapter."""
    from .sweep import run

    out = Path(out_path) if out_path else Path(lora_path).with_name(Path(lora_path).stem + "_cfg_sweep.jpg")
    click.echo(f"wrote {run(lora_path, manifest, [float(c) for c in cfgs.split(',')], side, steps, shift, seed, out)}")


@cli.command()
@click.argument("loras", nargs=-1)
@click.option("--cache", "cache_dir", default="data/cache_main", show_default=True)
@click.option("--count", default=8, show_default=True, help="cached pairs to probe")
@click.option("--seed", default=0, show_default=True)
def probe(loras, cache_dir, count, seed):
    """Per-noise-level prediction quality of adapters on cached pairs (see irisedit/probe.py)."""
    from .probe import run

    run(list(loras), cache_dir, count=count, seed=seed)
