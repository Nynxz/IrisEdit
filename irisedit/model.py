"""Iris-3B with in-context reference images.

References are patchified with the model's own patch embedder and appended after
the target's patch tokens. They share the target's (x, y) RoPE grid and differ only
on the frame axis: target = frame 0, reference i = frame i + 1. The frame axis takes
the `frame_pairs` slowest RoPE frequency pairs, which rotate by at most ~0.003 rad
across an image at frame 0, so the pretrained target geometry is effectively
unchanged. References only pass through the patch trunk; the pixel stage decodes
the target alone.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor"))

import iris3b.models.blocks  # noqa: E402,F401  registers block types
from iris3b.config import ModelConfig  # noqa: E402
from iris3b.models.dit import IrisDiT  # noqa: E402
from iris3b.nn.rope import rope_2d  # noqa: E402

TEXT_LEN = 300


def load_iris(path: str, frame_pairs: int = 2, dtype=torch.bfloat16) -> IrisDiT:
    """The released architecture (ModelConfig defaults) with weights from `path`."""
    cfg = replace(ModelConfig(), rope_frame_pairs=frame_pairs)
    with torch.device("meta"):
        model = IrisDiT(cfg)
    sd = load_file(path)
    model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True, assign=True)
    model.requires_grad_(False)
    return model.eval()


def _rope(model: IrisDiT, grid: tuple[int, int], frame: int, device) -> torch.Tensor:
    cfg = model.cfg
    return rope_2d(cfg.hidden_size // cfg.num_heads, grid[0], grid[1], theta=cfg.rope_theta,
                   scale=cfg.rope_scale, aspect=cfg.rope_aspect, frame_pairs=cfg.rope_frame_pairs,
                   frame_theta=cfg.rope_frame_theta, frame_index=frame).to(device)


def _run(fn, *args, ckpt: bool):
    if ckpt and torch.is_grad_enabled():
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


def forward(model: IrisDiT, x: torch.Tensor, refs: list[torch.Tensor], t: torch.Tensor,
            y: torch.Tensor, y_mask: torch.Tensor, grad_ckpt: bool = False) -> torch.Tensor:
    """Velocity for target `x` [B,3,H,W] given clean `refs` (list of [B,3,h,w], [-1,1]).

    t: model time [B] in [0, 1000]; y: [B, 300, 12, 2560]; y_mask: [B, 300].
    """
    cfg = model.cfg
    p = cfg.patch_size
    batch, _, height, width = x.shape
    grid = (height // p, width // p)
    n_target = grid[0] * grid[1]

    tokens = [model.s_embedder(F.unfold(x, kernel_size=p, stride=p).transpose(1, 2))]
    ropes = [_rope(model, grid, 0, x.device)]
    for i, ref in enumerate(refs):
        tokens.append(model.s_embedder(F.unfold(ref.to(x.dtype), kernel_size=p, stride=p).transpose(1, 2)))
        ropes.append(_rope(model, (ref.shape[-2] // p, ref.shape[-1] // p), i + 1, x.device))
    s = torch.cat(tokens, dim=1)
    rope_img = torch.cat(ropes, dim=0)

    t_emb = model.t_embedder(t)
    cond = F.silu(t_emb)
    y = model.y_embedder(y[:, :TEXT_LEN], y_mask[:, :TEXT_LEN])
    y = y + model.y_pos_embedding[:, : y.shape[1]].to(y.dtype)
    rope_txt = model._fetch_rope_txt(y.shape[1], x.device)

    for block in model.blocks:
        s, y = _run(block, s, y, cond, rope_img, rope_txt, ckpt=grad_ckpt)

    s = F.silu(t_emb + s[:, :n_target])
    s_cond = s.reshape(batch * n_target, -1)
    pixels = model.pixel_embedder(x)
    rope_pix = model._fetch_rope_pix(grid, x.device)
    for block in model.pixel_blocks:
        pixels = _run(block, pixels, s_cond, rope_pix, grid, ckpt=grad_ckpt)
    out = model.final_layer(pixels)
    out = out.reshape(batch, n_target, p * p, -1).permute(0, 3, 2, 1).reshape(batch, -1, n_target)
    return F.fold(out, output_size=(height, width), kernel_size=p, stride=p)
