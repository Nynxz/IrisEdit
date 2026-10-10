"""How closely base + extracted LoRA reproduces a full Iris-3B fine-tune.

    .venv/bin/python scripts/lora_fidelity.py restoration upscaler/model.safetensors \
        upscaler/empty_prompt.safetensors lora.safetensors image.png [--rank-tag 128]

restoration: one pass at sigma 0.5 on a 1024 tile of the 4x bicubic upsample, output
x - 0.5 * v (the released Restorer, without tiling or colour fix).
depth: one pass at sigma 1.0 on the RGB image (folded checkpoint, see ComfyUI-Iris
scripts/fold_depth.py), output = the model output (relative log depth, all channels equal).
Prints PSNR of base+LoRA and of plain base against the full fine-tune.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from irisedit import model as iris  # noqa: E402
from irisedit.config import load as load_config  # noqa: E402
from irisedit.data import to_tensor  # noqa: E402
from irisedit.paths import resolve_weights  # noqa: E402


def merge(model, lora_path):
    sd = load_file(lora_path)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in sd.items():
            body = k[len("diffusion_model."):]
            if body.endswith(".lora_down.weight"):
                module = body[: -len(".lora_down.weight")]
                up = sd[f"diffusion_model.{module}.lora_up.weight"].float()
                down = v.float()
                w = params[f"{module}.weight"]
                delta = (up @ down)[: w.shape[0], : w.shape[1]] * float(sd[f"diffusion_model.{module}.alpha"]) / down.shape[0]
                w.add_(delta.to(w.dtype))
            elif body.endswith(".diff_b"):
                p = params[body[: -len(".diff_b")] + ".bias"]
                p.add_(v.to(p.dtype))
            elif body.endswith(".diff"):
                name = body[: -len(".diff")]
                p = params[name + ".weight"] if name + ".weight" in params else params[name]
                p.add_(v.to(p.dtype))


def run(model, task, x, emb, mask):
    model.cuda()
    with torch.no_grad():
        if task == "restoration":
            v = iris.forward(model, x, [], torch.tensor([500.0], device="cuda"), emb, mask)
            out = x.float() - 0.5 * v.float()
        else:
            out = iris.forward(model, x, [], torch.tensor([1000.0], device="cuda"), emb, mask).float()
    model.cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["restoration", "depth"])
    ap.add_argument("finetune")
    ap.add_argument("prompt")
    ap.add_argument("lora")
    ap.add_argument("image")
    args = ap.parse_args()
    cfg = load_config("_base", strict=False)

    img = Image.open(args.image).convert("RGB")
    if args.task == "restoration":
        side = min(img.size)
        img = img.crop((0, 0, side, side)).resize((256, 256), Image.LANCZOS)
        x = F.interpolate(to_tensor(img)[None], size=(1024, 1024), mode="bicubic", align_corners=False)
    else:
        img = img.crop((0, 0, min(img.size), min(img.size))).resize((768, 768), Image.LANCZOS)
        x = to_tensor(img)[None]
    x = x.cuda().bfloat16()
    prompt = load_file(args.prompt)
    emb = prompt["embeddings"].cuda().bfloat16()
    mask = prompt["mask"].cuda().long()

    ft = iris.load_iris(args.finetune, frame_pairs=0)
    target = run(ft, args.task, x, emb, mask)
    del ft
    torch.cuda.empty_cache()

    base = iris.load_iris(resolve_weights("checkpoint", cfg), frame_pairs=0)
    plain = run(base, args.task, x, emb, mask)
    merge(base, args.lora)
    lora_out = run(base, args.task, x, emb, mask)

    rng = float(target.max() - target.min())
    psnr = lambda a: float(10 * torch.log10(rng ** 2 / ((a - target) ** 2).mean()))
    print(f"[fidelity] {args.task}: base+LoRA vs full fine-tune {psnr(lora_out):.1f} dB | plain base {psnr(plain):.1f} dB")


if __name__ == "__main__":
    main()
