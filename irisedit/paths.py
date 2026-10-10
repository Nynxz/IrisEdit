"""Where the weights come from.

Order: env var (IRISEDIT_CHECKPOINT / IRISEDIT_TEXT_ENCODER), then `irisedit.toml` in
the repo root, then the config value. `hf:<owner>/<repo>/<file>` downloads that file
into the Hugging Face cache; a bare repo id is passed through (text encoder).
"""
from __future__ import annotations

import os
import tomllib

from .config import REPO

ENV = {"checkpoint": "IRISEDIT_CHECKPOINT", "text_encoder": "IRISEDIT_TEXT_ENCODER"}


def local_overrides() -> dict:
    path = REPO / "irisedit.toml"
    return tomllib.loads(path.read_text()) if path.exists() else {}


def resolve_weights(key: str, cfg: dict) -> str:
    spec = os.environ.get(ENV[key]) or local_overrides().get(key) or cfg.get(key)
    if not spec:
        raise SystemExit(f"[paths] no `{key}` set (config, irisedit.toml or ${ENV[key]})")
    if spec.startswith("hf:"):
        from huggingface_hub import hf_hub_download

        owner, repo, filename = spec[3:].split("/", 2)
        return hf_hub_download(f"{owner}/{repo}", filename)
    return spec
