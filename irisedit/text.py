"""Qwen3-VL-4B text encoding, identical to iris3b.text.qwen3_vl.

The system/user prefix runs through the decoder but is sliced off; the caption plus
the assistant-turn suffix is kept, padded to 300 tokens, and hidden states are taken
after layers 2, 5, ..., 35. The empty caption is the CFG null Iris was trained with.

`source` is a Hugging Face repo id / local HF directory, or a single .safetensors file
in the ComfyUI layout (`model.language_model.*` keys, e.g. qwen3vl_4b_bf16.safetensors).
"""
from __future__ import annotations

import torch

HF_REPO = "Qwen/Qwen3-VL-4B-Instruct"
PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial "
    "relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
MAX_LENGTH = 300


def _load_decoder(source: str, dtype):
    if source.endswith(".safetensors"):
        from safetensors.torch import load_file
        from transformers import AutoConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

        config = AutoConfig.from_pretrained(HF_REPO).text_config
        with torch.device("meta"):
            decoder = Qwen3VLTextModel(config)
        prefix = "model.language_model."
        sd = {k[len(prefix):]: v.to(dtype) for k, v in load_file(source).items() if k.startswith(prefix)}
        missing, unexpected = decoder.load_state_dict(sd, strict=False, assign=True)
        if unexpected or any("rotary" not in m for m in missing):
            raise SystemExit(f"[text] {source} does not look like Qwen3-VL-4B ({missing[:3]}, {unexpected[:3]})")
        decoder.rotary_emb = type(decoder.rotary_emb)(config=config)  # non-persistent buffer, rebuild off meta
        return decoder
    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(source, dtype=dtype)
    return model.get_decoder()


class TextEncoder:
    def __init__(self, source: str = HF_REPO, device="cuda", dtype=torch.bfloat16):
        from transformers import AutoTokenizer

        tok_source = HF_REPO if source.endswith(".safetensors") else source
        self.tokenizer = AutoTokenizer.from_pretrained(tok_source)
        self.prefix = torch.tensor(self.tokenizer.encode(PREFIX, add_special_tokens=False))
        self.suffix = torch.tensor(self.tokenizer.encode(SUFFIX, add_special_tokens=False))
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.device = torch.device(device)
        self.decoder = _load_decoder(source, dtype).to(self.device).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """-> embeddings [B, 300, 12, 2560], mask [B, 300] (int, 1 = real token)."""
        budget = MAX_LENGTH - len(self.suffix)
        start = len(self.prefix)
        ids = torch.full((len(prompts), start + MAX_LENGTH), self.pad_id, dtype=torch.long)
        attn = torch.zeros_like(ids)
        ids[:, :start] = self.prefix
        for row, caption in enumerate(self.tokenizer(prompts, add_special_tokens=False)["input_ids"]):
            n = min(len(caption), budget)
            ids[row, start:start + n] = torch.tensor(caption[:n], dtype=torch.long)
            ids[row, start + n:start + n + len(self.suffix)] = self.suffix
            attn[row, :start + n + len(self.suffix)] = 1
        out = self.decoder(input_ids=ids.to(self.device), attention_mask=attn.to(self.device),
                           use_cache=False, output_hidden_states=True, return_dict=True)
        mask = attn[:, start:].to(self.device)
        emb = torch.stack([out.hidden_states[i][:, start:] for i in LAYERS], dim=2)
        return emb * mask[:, :, None, None].to(emb.dtype), mask
