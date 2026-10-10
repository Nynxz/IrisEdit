"""Edit sampling: the reference DPM-Solver++(2M) on the flow schedule, with the
reference image attached to both CFG branches (unconditional = empty instruction)."""
from __future__ import annotations

import torch

from . import model as iris
from .model import IrisDiT
from iris3b.flow.solver import FlowDPMSolver


@torch.no_grad()
def edit(model: IrisDiT, encoder, source: torch.Tensor, instruction: str, steps: int = 50, cfg: float = 3.0,
         shift: float = 3.0, seed: int = 0) -> torch.Tensor:
    """source: [3, H, W] in [-1, 1] (H, W multiples of 16). Returns the edit, same size."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    ref = source[None].to(device, dtype)
    emb, mask = encoder([instruction, ""])
    emb = emb.to(device, dtype)
    cond, uncond = emb[:1], emb[1:]
    cond_mask, cfg_mask = mask[:1], torch.cat([mask[1:], mask[:1]])

    def model_fn(x, t_model, y):
        refs = [ref.expand(x.shape[0], -1, -1, -1)]
        y_mask = cond_mask if y is cond else cfg_mask
        return iris.forward(model, x.to(dtype), refs, t_model, y, y_mask).float()

    gen = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(1, 3, *source.shape[-2:], generator=gen, device=device)
    solver = FlowDPMSolver(model_fn, cfg_scale=cfg)
    out = solver.sample(z, cond, uncond if cfg != 1.0 else None, steps=steps, order=2, shift=shift)
    return out[0].clamp(-1, 1)
