"""Config loading: `extends:` inheritance + strict key validation.

Every config is a flat dict. `extends: _base` reads configs/_base.yaml first and the
child's keys override it. Unknown keys are a hard error, so a misspelled setting
can't sit in a config doing nothing.
"""
from __future__ import annotations

import difflib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO / "configs"

RUN_KEYS = {
    "name", "extends",
    # backbone
    "checkpoint", "text_encoder", "grad_checkpointing", "frame_pairs",
    # starting point
    "base_lora", "bootstrap_lora",
    # adapter
    "lora_rank", "lora_alpha", "lora_dropout", "lora_targets", "lora_scopes",
    # data
    "cache_dir", "min_alignment", "batch_size", "num_workers",
    # optimisation
    "lr", "weight_decay", "grad_accum", "max_grad_norm", "lr_schedule", "warmup_steps", "min_lr",
    # flow objective
    "flow_shift", "t_logitnorm_mean", "t_logitnorm_std", "t_high_frac", "t_high_min",
    "prompt_dropout", "edit_loss_weight",
    # schedule / output
    "max_steps", "seed", "save_every", "log_every", "out_dir",
    # previews
    "preview_manifest", "preview_every", "preview_count", "preview_side", "preview_steps", "preview_cfg",
    "preview_seed",
}

DATA_KEYS = {"sources", "size"}

KNOWN_KEYS = RUN_KEYS | DATA_KEYS


def _suggest(key: str) -> str:
    close = difflib.get_close_matches(key, sorted(KNOWN_KEYS), n=1, cutoff=0.7)
    return f" (did you mean `{close[0]}`?)" if close else ""


def resolve(name: str | Path, relative_to: Path = CONFIG_DIR) -> Path:
    """A config reference: a path, or a bare name under configs/."""
    for cand in (Path(name), relative_to / str(name), relative_to / f"{name}.yaml",
                 CONFIG_DIR / str(name), CONFIG_DIR / f"{name}.yaml"):
        if cand.is_file():
            return cand
    raise SystemExit(f"[config] no config '{name}'")


def _read(path: Path, seen: list[Path]) -> dict:
    if path in seen:
        raise SystemExit(f"[config] circular extends: {' -> '.join(p.name for p in [*seen, path])}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"[config] {path} must be a YAML mapping")
    parent_name = raw.pop("extends", None)
    if parent_name is None:
        return raw
    parent = _read(resolve(str(parent_name), path.parent), [*seen, path])
    parent.update(raw)
    return parent


def load(name: str | Path, strict: bool = True) -> dict:
    path = resolve(name)
    cfg = _read(path, [])
    if strict:
        bad = [f"  `{k}`: not a recognised setting{_suggest(k)}" for k in sorted(cfg) if k not in KNOWN_KEYS]
        if bad:
            raise SystemExit(f"[config] {path} has keys nothing reads:\n" + "\n".join(bad))
    cfg.setdefault("name", path.stem)
    return cfg
