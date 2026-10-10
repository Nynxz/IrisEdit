"""Edit-LoRA training loop.

Per step: noise the target at a logit-normal timestep (shifted like Iris's own
training), run the in-context forward with the clean source as reference, and take
MSE against the velocity (noise - target). The instruction is dropped to the empty
caption with probability `prompt_dropout`, which trains the unconditional branch CFG
uses. Resumable: out_dir/state.pt holds LoRA + optimizer + step + RNG, and
Ctrl-C / SIGTERM saves and stops at the next step boundary.
"""
from __future__ import annotations

import json
import math
import signal
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import lora
from . import model as iris
from .data import EditPairs, ShapeBucketSampler, ar_dims, collate, to_pil, to_tensor
from .paths import resolve_weights

_STOP = {"flag": False}


def _on_signal(signum, _frame):
    _STOP["flag"] = True
    print(f"\n[train] signal {signum}: saving at the next step boundary", flush=True)


def lr_factor(step: int, cfg: dict) -> float:
    warmup = cfg.get("warmup_steps", 0)
    if step < warmup:
        return (step + 1) / warmup
    if cfg.get("lr_schedule", "cosine") == "constant":
        return 1.0
    progress = min(1.0, (step - warmup) / max(1, cfg["max_steps"] - warmup))
    floor = cfg.get("min_lr", 0.0) / cfg["lr"]
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def sample_t(batch: int, cfg: dict, device) -> torch.Tensor:
    """Shifted logit-normal sigma in (0, 1), plus an optional share from the pure-noise end.

    In pixel space an image's layout survives a lot of noise, so with Iris's own schedule
    (almost no samples above sigma 0.98) the model learns to read the layout from the faint
    target left in x_t instead of from the reference. Sampling starts from pure noise, where
    that signal is absent, and the layout gets invented. `t_high_frac` of the batch is drawn
    uniformly from [t_high_min, 1] so the reference has to carry the layout."""
    u = torch.sigmoid(torch.randn(batch, device=device) * cfg.get("t_logitnorm_std", 1.0)
                      + cfg.get("t_logitnorm_mean", 0.0))
    shift = cfg.get("flow_shift", 3.5)
    sigma = shift * u / (1 + (shift - 1) * u)
    frac = cfg.get("t_high_frac", 0.0)
    if frac:
        lo = cfg.get("t_high_min", 0.97)
        high = lo + (1 - lo) * torch.rand(batch, device=device)
        sigma = torch.where(torch.rand(batch, device=device) < frac, high, sigma)
    return sigma


def edit_weights(src: torch.Tensor, tgt: torch.Tensor, strength: float) -> torch.Tensor:
    """Per-pixel loss weights [B, 1, H, W] that favour where the target differs from the source.

    In most pairs only a small region changes, so an unweighted loss is dominated by
    reproducing the source and the model drifts towards copying. Weights are 1 + strength on
    changed pixels (smoothed, dilated) and 1 elsewhere, normalised to mean 1 per sample, so
    pairs that change everywhere are unaffected and the overall loss scale stays the same."""
    with torch.no_grad():
        diff = (tgt.float() - src.float()).abs().mean(1, keepdim=True)
        diff = F.avg_pool2d(diff, 9, stride=1, padding=4)
        mask = F.max_pool2d((diff > 0.08).float(), 17, stride=1, padding=8)
        w = 1 + strength * mask
        return w / w.mean(dim=(2, 3), keepdim=True)


def metadata(cfg: dict, step: int) -> dict:
    return {
        "irisedit_layout": "text|target(frame=0)|reference(frame=1)",
        "irisedit_frame_pairs": cfg.get("frame_pairs", 2),
        "irisedit_rank": cfg["lora_rank"],
        "irisedit_alpha": cfg["lora_alpha"],
        "irisedit_step": step,
        "irisedit_run": cfg["name"],
        "ss_base_model_version": "iris-3b",
        "ss_network_module": "networks.lora",
        "ss_network_dim": cfg["lora_rank"],
        "ss_network_alpha": cfg["lora_alpha"],
    }


def save_state(out_dir: Path, model, opt, step: int):
    tmp = out_dir / "state.pt.tmp"
    torch.save({"step": step, "lora": lora.state(model), "optimizer": opt.state_dict(),
                "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()}, tmp)
    tmp.replace(out_dir / "state.pt")


def load_state(path: Path, model, opt) -> int:
    s = torch.load(path, map_location="cpu", weights_only=False)
    params = dict(model.named_parameters())
    missing = [n for n, p in params.items() if p.requires_grad and n not in s["lora"]]
    if missing:
        raise SystemExit(f"[train] {path} does not match this adapter ({len(missing)} params missing); use --fresh")
    with torch.no_grad():
        for n, v in s["lora"].items():
            params[n].copy_(v)
    opt.load_state_dict(s["optimizer"])
    torch.set_rng_state(s["rng_cpu"])
    torch.cuda.set_rng_state_all(s["rng_cuda"])
    return int(s["step"])


def load_previews(path: str | None, count: int):
    if not path:
        return []
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows[:count]


def render_previews(model, encoder, rows, out_dir: Path, step: int, cfg: dict):
    from PIL import Image

    from .sample import edit

    dest = out_dir / "samples" / f"step-{step:07d}"
    dest.mkdir(parents=True, exist_ok=True)
    model.eval()
    for i, row in enumerate(rows):
        img = Image.open(row["image"]).convert("RGB")
        img = img.resize(ar_dims(img.width, img.height, cfg.get("preview_side", 512)), Image.LANCZOS)
        out = edit(model, encoder, to_tensor(img), row["instruction"], steps=cfg.get("preview_steps", 30),
                   cfg=cfg.get("preview_cfg", 3.0), shift=cfg.get("flow_shift", 3.5),
                   seed=cfg.get("preview_seed", 0))
        pair = Image.new("RGB", (img.width * 2, img.height))
        pair.paste(img, (0, 0))
        pair.paste(to_pil(out), (img.width, 0))
        pair.save(dest / f"{i:02d}.jpg", quality=92)
    model.train()
    print(f"[train] previews -> {dest}", flush=True)


def run(cfg: dict, fresh: bool = False):
    torch.manual_seed(cfg.get("seed", 0))
    device = torch.device("cuda")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    ds = EditPairs(cfg["cache_dir"], cfg.get("min_alignment"))
    print(f"[train] {len(ds)} cached pairs, {len(set(ds.shapes()))} sizes"
          + (f" ({ds.dropped} dropped below alignment {cfg['min_alignment']})" if ds.dropped else ""), flush=True)

    checkpoint = resolve_weights("checkpoint", cfg)
    print(f"[train] loading Iris-3B from {checkpoint}", flush=True)
    model = iris.load_iris(checkpoint, frame_pairs=cfg.get("frame_pairs", 2))
    if cfg.get("base_lora"):
        print(f"[train] merged base LoRA ({lora.merge(model, cfg['base_lora'])} layers): {cfg['base_lora']}")
    n = lora.inject(model, cfg["lora_targets"], cfg["lora_scopes"], cfg["lora_rank"], cfg["lora_alpha"],
                    cfg.get("lora_dropout", 0.0))
    model.to(device)
    params = lora.parameters(model)
    print(f"[train] LoRA on {n} linears, {sum(p.numel() for p in params) / 1e6:.1f}M trainable", flush=True)

    from .text import TextEncoder

    encoder = TextEncoder(resolve_weights("text_encoder", cfg), device=device)
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0), fused=True)

    step = 0
    state_path = out_dir / "state.pt"
    if state_path.exists() and not fresh:
        step = load_state(state_path, model, opt)
        print(f"[train] resumed at step {step}", flush=True)
    elif cfg.get("bootstrap_lora"):
        lora.load_into(model, cfg["bootstrap_lora"])
        print(f"[train] starting from {cfg['bootstrap_lora']}", flush=True)
    for g in opt.param_groups:
        g["initial_lr"] = cfg["lr"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, cfg), last_epoch=step - 1)

    accum = cfg.get("grad_accum", 1)
    batch_size = cfg.get("batch_size", 1)
    sampler = ShapeBucketSampler(ds.shapes(), batch_size, seed=cfg.get("seed", 0),
                                 start_index=step * accum * batch_size)
    batches = iter(DataLoader(ds, batch_sampler=sampler, num_workers=cfg.get("num_workers", 4),
                              collate_fn=collate, pin_memory=True, persistent_workers=True))
    previews = load_previews(cfg.get("preview_manifest"), cfg.get("preview_count", 4))
    preview_every = cfg.get("preview_every") or cfg["save_every"]
    drop = cfg.get("prompt_dropout", 0.1)
    edit_weight = cfg.get("edit_loss_weight", 0.0)
    if edit_weight:
        print(f"[train] edit-region loss weighting: changed pixels x{1 + edit_weight:g} (normalised)", flush=True)
    grad_ckpt = cfg.get("grad_checkpointing", True)

    def save(name: str):
        lora.save(model, str(out_dir / name), metadata(cfg, step))
        save_state(out_dir, model, opt, step)

    if previews and step == 0:
        render_previews(model, encoder, previews, out_dir, step, cfg)

    model.train()
    start, t0, running = step, time.time(), None
    torch.cuda.reset_peak_memory_stats()
    while step < cfg["max_steps"]:
        for _ in range(accum):
            batch = next(batches)
            src = batch["source"].to(device, torch.bfloat16, non_blocking=True)
            tgt = batch["target"].to(device, torch.bfloat16, non_blocking=True)
            prompts = ["" if torch.rand(1).item() < drop else p for p in batch["instruction"]]
            emb, mask = encoder(prompts)
            sigma = sample_t(tgt.shape[0], cfg, device)
            noise = torch.randn_like(tgt)
            s = sigma.view(-1, 1, 1, 1).to(tgt.dtype)
            x_t = (1 - s) * tgt + s * noise
            pred = iris.forward(model, x_t, [src], sigma * 1000, emb.to(torch.bfloat16), mask, grad_ckpt=grad_ckpt)
            err = (pred.float() - (noise - tgt).float()).pow(2)
            loss = err.mean() if not edit_weight else (err * edit_weights(src, tgt, edit_weight)).mean()
            (loss / accum).backward()
        if cfg.get("max_grad_norm"):
            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()

        if step % cfg.get("log_every", 50) == 0 or step == start + 1:
            rate = (step - start) / (time.time() - t0)
            print(f"[train] step {step}/{cfg['max_steps']} loss {loss.item():.4f} (avg {running:.4f}) "
                  f"lr {opt.param_groups[0]['lr']:.2e} {rate:.2f} it/s "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
        if step % cfg["save_every"] == 0:
            save(f"step{step}.safetensors")
            print(f"[train] saved step{step}.safetensors", flush=True)
        if previews and step % preview_every == 0:
            render_previews(model, encoder, previews, out_dir, step, cfg)
        if _STOP["flag"]:
            save("last.safetensors")
            print(f"[train] stopped at step {step}; rerun to continue", flush=True)
            return
    save("last.safetensors")
    print(f"[train] done -> {out_dir / 'last.safetensors'}", flush=True)
