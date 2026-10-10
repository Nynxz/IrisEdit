"""LoRA for Iris-3B, saved in the ComfyUI/kohya layout.

Checkpoints hold `diffusion_model.<module>.lora_down.weight / lora_up.weight / alpha`.
Iris's trunk SwiGLU width is 6826; ComfyUI-Iris pads it to 6912 at load (exact, the
extra units are zero) so int8/convrot weights work, so saved trunk-MLP adapters are
zero-padded to match. Padding is undone on load.
"""
from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

PREFIX = "diffusion_model."
MLP_MULTIPLE = 256
_TRUNK_MLP = re.compile(r"^blocks\.\d+\.mlp(_[xy])?\.(w1|w2|w3)$")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r, self.alpha, self.scaling, self.strength = r, float(alpha), alpha / r, 1.0
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.base(x)
        h = self.drop(x).float() @ self.lora_A.t() @ self.lora_B.t()
        return out + (h * (self.scaling * self.strength)).to(out.dtype)


def _targets(model: nn.Module, targets, scopes):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear) \
                and any(name.startswith(f"{s}.") for s in scopes) and any(name.endswith(t) for t in targets):
            yield name, module


def _wrap(model, name, module, rank, alpha, dropout):
    parent, attr = name.rsplit(".", 1)
    setattr(model.get_submodule(parent), attr, LoRALinear(module, rank, alpha, dropout))


def inject(model: nn.Module, targets, scopes, rank: int, alpha: float, dropout: float = 0.0) -> int:
    found = list(_targets(model, targets, scopes))
    if not found:
        raise SystemExit(f"[lora] no linears matched targets={targets} scopes={scopes}")
    for name, module in found:
        _wrap(model, name, module, rank, alpha, dropout)
    return len(found)


def modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def parameters(model: nn.Module):
    return [p for m in modules(model).values() for p in (m.lora_A, m.lora_B)]


def _pad(name: str, down: torch.Tensor, up: torch.Tensor):
    m = _TRUNK_MLP.match(name)
    if not m:
        return down, up
    if m.group(2) == "w2":
        return torch.nn.functional.pad(down, (0, -down.shape[1] % MLP_MULTIPLE)), up
    return down, torch.nn.functional.pad(up, (0, 0, 0, -up.shape[0] % MLP_MULTIPLE))


def save(model: nn.Module, path: str, metadata: dict | None = None, dtype=torch.bfloat16):
    state = {}
    for name, m in modules(model).items():
        down, up = _pad(name, m.lora_A.detach().float().cpu(), m.lora_B.detach().float().cpu())
        state[f"{PREFIX}{name}.lora_down.weight"] = down.to(dtype).contiguous()
        state[f"{PREFIX}{name}.lora_up.weight"] = up.to(dtype).contiguous()
        state[f"{PREFIX}{name}.alpha"] = torch.tensor(m.alpha)
    save_file(state, path, metadata={k: str(v) for k, v in (metadata or {}).items()})


def read_pairs(path: str) -> dict[str, dict]:
    """{module: {down, up, alpha}} with ComfyUI padding removed."""
    out: dict[str, dict] = {}
    for k, v in load_file(path).items():
        if not k.startswith(PREFIX):
            continue
        body = k[len(PREFIX):]
        for suffix, field in ((".lora_down.weight", "down"), (".lora_up.weight", "up"), (".alpha", "alpha")):
            if body.endswith(suffix):
                out.setdefault(body[: -len(suffix)], {})[field] = v
    return out


def _fit(name: str, t: torch.Tensor, shape) -> torch.Tensor:
    if tuple(t.shape) == tuple(shape):
        return t
    if _TRUNK_MLP.match(name) and all(a >= b for a, b in zip(t.shape, shape)):
        return t[tuple(slice(0, s) for s in shape)]
    raise SystemExit(f"[lora] shape mismatch at {name}: file {tuple(t.shape)} vs model {tuple(shape)}")


def load_into(model: nn.Module, path: str) -> int:
    """Load adapter weights into already-injected LoRALinear modules."""
    pairs, mods = read_pairs(path), modules(model)
    if set(pairs) != set(mods):
        raise SystemExit(f"[lora] {path} does not match the injected adapter "
                         f"({len(set(pairs) - set(mods))} extra, {len(set(mods) - set(pairs))} missing)")
    with torch.no_grad():
        for name, ab in pairs.items():
            m = mods[name]
            m.lora_A.copy_(_fit(name, ab["down"], m.lora_A.shape))
            m.lora_B.copy_(_fit(name, ab["up"], m.lora_B.shape))
    return len(pairs)


def inject_from_file(model: nn.Module, path: str) -> int:
    """Inference: inject exactly the modules the file names, at its rank/alpha, and load it."""
    for name, ab in read_pairs(path).items():
        _wrap(model, name, model.get_submodule(name), ab["down"].shape[0], float(ab.get("alpha", ab["down"].shape[0])), 0.0)
    return load_into(model, path)


def merge(model: nn.Module, path: str, scaling: float = 1.0) -> int:
    """Bake an adapter into the backbone weights (for training a specialist on top)."""
    n = 0
    with torch.no_grad():
        for name, ab in read_pairs(path).items():
            lin = model.get_submodule(name)
            rank = ab["down"].shape[0]
            down = _fit(name, ab["down"].float(), (rank, lin.in_features))
            up = _fit(name, ab["up"].float(), (lin.out_features, rank))
            s = scaling * float(ab.get("alpha", rank)) / rank
            lin.weight.data = (lin.weight.float() + (up @ down).to(lin.weight.device) * s).to(lin.weight.dtype)
            n += 1
    return n


def state(model: nn.Module) -> dict:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def set_strength(model: nn.Module, strength: float):
    for m in modules(model).values():
        m.strength = float(strength)
