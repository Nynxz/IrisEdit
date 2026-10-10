# IrisEdit

Train [Iris-3B](https://github.com/speridlabs/iris-3b) into an instruction image editor with an
in-context edit LoRA, the same recipe as [NK2E](https://github.com/Nynxz/NK2E) for Krea 2.

The source image is appended to the model's patch tokens as a clean reference, tagged on a
RoPE frame axis (target = frame 0, reference = frame 1). Iris's authors reserved that axis
for exactly this; it uses the 2 slowest RoPE frequency pairs, which barely rotate across an
image, so turning it on leaves the pretrained model unchanged (0.01% output difference in
fp32). The reference only passes through the patch trunk; the pixel stage decodes the target
alone. Loss is on the target only, and only a LoRA is trained.

Iris works in pixel space, so there is no VAE anywhere: the data cache is resized image
pairs, and training is plain bf16 on a 32GB card (~18GB at 768px).

Trained adapters load in ComfyUI with [ComfyUI-Iris](https://github.com/Nynxz/ComfyUI-Iris)
and stock nodes (Load Image, VAE Encode `pixel_space`, ReferenceLatent, LoraLoaderModelOnly).

## Setup

```bash
uv venv --python 3.12 && uv pip install -e .
```

Weights download from the Hugging Face Hub on first use (`speridlabs/iris-3b`,
`Qwen/Qwen3-VL-4B-Instruct`). To use files you already have, create `irisedit.toml`:

```toml
checkpoint = "/path/to/iris-3b.safetensors"          # the FP32 release or a bf16 copy
text_encoder = "/path/to/qwen3vl_4b_bf16.safetensors" # ComfyUI single file, or an HF repo/dir
```

## Commands

```
irisedit cache <data-config>      stream a dataset mix into data/cache_<name>
irisedit train [run-config]       train or resume an edit LoRA (default: generalist)
irisedit probe <lora>...          per-noise-level prediction quality on cached pairs
irisedit grid <run-config>        progress grid of a run's previews
irisedit sweep <lora>             previews at several CFG scales
irisedit edit <img> <instr> --lora <file>
irisedit status                   caches and runs
```

`train` resumes `out_dir/state.pt` if present. Ctrl-C saves and stops; rerun to continue,
or raise `max_steps` and rerun to train longer. Every `save_every` steps it writes a
ComfyUI-format LoRA and renders the edits in `examples/preview.jsonl` to `out_dir/samples/`.
`./run_generalist.sh [run-config]` finishes the cache, waits for free VRAM and trains.

## Recipe

```bash
irisedit cache data_smoke && irisedit train generalist_smoke   # check the pipeline (minutes)
irisedit cache data                                             # ~49k pairs at 768px
irisedit train generalist_v3                                    # the working recipe
```

The data mix (`configs/data.yaml`) is MagicBrush, OmniEdit (o_score >= 8), NHR-Edit
(category-capped) and HQ-Edit, all around 1 MP natively, cached at 768px.

## What the runs taught us

Each of these broke a run before it was fixed, so the configs encode them:

- **Train at 768, not 512.** The released checkpoint was fine-tuned at ~1 MP and draws a
  16 px patch grid below ~768, even without a LoRA.
- **Sample the pure-noise end** (`t_high_frac: 0.3`, `t_high_min: 0.97`). This is the big one.
  In pixel space an image's layout survives a lot of noise, and Iris's own schedule almost never
  trains above sigma 0.98. The model learned to read the layout from the noisy target instead
  of the reference, then redrew the scene when sampling from pure noise. `irisedit probe` shows
  it: prediction quality at sigma 1.0 collapsed to 11-14 dB (below simply copying the source,
  ~18-22 dB) while sigma 0.9 looked fine. With the fix it reaches the copy baseline within
  1000 steps and edits keep the composition.
- **Drop pairs that don't keep the composition** (`min_alignment: 0.4`, see `irisedit/align.py`).
  HQ-Edit's before and after images are generated separately (edge correlation ~0.03 vs ~0.9 for
  the others), and training on them teaches the model to redraw.
- **Read OmniEdit across its shards** (`interleave: 16`). It is stored sorted by edit type, so
  streaming the start gives only "add object" edits.
- **Weight changed pixels mildly** (`edit_loss_weight: 1`). Stronger weighting (4) starved the
  unchanged pixels and pushed towards redrawing.

Instructions are used as written, with no template around them, so the text you type in
ComfyUI's CLIP Text Encode is exactly what the model was trained on. In ComfyUI, attach the
reference (ReferenceLatent) to the negative as well as the positive. CFG around 2 works best.

## LoRA format

kohya/ComfyUI keys (`diffusion_model.<module>.lora_down/lora_up/alpha`). Iris's trunk MLP
width is 6826, which ComfyUI-Iris pads to 6912 at load so int8/convrot weights work; saved
MLP adapters are zero-padded to match, which is exact.

## License

Apache-2.0. Includes the Iris-3B model code (`vendor/iris3b`, Apache-2.0); see `NOTICE`.
Training data keeps its own licenses.
