#!/usr/bin/env python3
# quantize_int4.py — per-channel INT4 (range) weight quantization
# + WikiText-2 PPL evaluation.
#
# Reproduces the INT4 (per-channel range) row in Table 2 and the INT4 row in
# Table 3 (Qwen-7B ablation). Used by 04_ablation.sh.
#
# Per-channel range scaling: scale = qmax / max(|w_row|), qmax=7 for signed INT4.

import argparse, json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import transformers.modeling_utils as modeling_utils

transformers.logging.set_verbosity_error()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-12


def is_lin(m):
    return isinstance(m, (nn.Linear, modeling_utils.Conv1D))


def skip(name, m):
    return (name == "lm_head" or name.endswith(".lm_head") or isinstance(m, nn.Embedding))


@torch.no_grad()
def int4_per_channel_range(W: torch.Tensor, nbits: int = 4):
    """Per-output-row signed INT quantize: scale = qmax / amax(row)."""
    qmin = -(1 << (nbits - 1))
    qmax = (1 << (nbits - 1)) - 1
    dtype, dev = W.dtype, W.device
    W_f = W.detach().float()
    amax = W_f.abs().amax(dim=-1, keepdim=True).clamp_min(EPS)          # [Cout, 1]
    scale = qmax / amax                                                  # [Cout, 1]
    q = torch.clamp(torch.round(W_f * scale), qmin, qmax) / scale
    return q.to(dtype=dtype, device=dev)


@torch.no_grad()
def eval_wikitext2_ppl(model, tok, seqlen, forward_dtype):
    model.eval()
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt", add_special_tokens=False).input_ids
    n = ids.numel() // seqlen
    dev = next(model.parameters()).device
    nll = 0.0
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[forward_dtype]
    with torch.cuda.amp.autocast(enabled=(forward_dtype != "fp32"), dtype=dt):
        for i in range(n):
            b = ids[:, i*seqlen:(i+1)*seqlen].to(dev)
            logits = model(b).logits
            sl = logits[:, :-1, :].contiguous().float()
            lab = b[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lab.view(-1))
            nll += loss.item() * seqlen
    return math.exp(nll / (n * seqlen))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF id or local dir")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--save_dir", required=True, help="Output dir (writes metrics.json)")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    a = p.parse_args()

    os.makedirs(a.save_dir, exist_ok=True)
    td = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]

    print(f"[Load] {a.model}  dtype={a.dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=td, trust_remote_code=a.trust_remote_code, low_cpu_mem_usage=True
    ).to(DEV)
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True, trust_remote_code=a.trust_remote_code)

    n_quantized = 0
    for name, mod in model.named_modules():
        if skip(name, mod) or not is_lin(mod):
            continue
        if mod.weight.dim() != 2:
            continue
        mod.weight.data = int4_per_channel_range(mod.weight.data)
        n_quantized += 1
    print(f"[Quantize] INT4 (per-channel range) on {n_quantized} Linear/Conv1D layers", flush=True)

    ppl = eval_wikitext2_ppl(model, tok, a.seqlen, a.dtype)
    print(f"[Result] wikitext2_ppl = {ppl:.4f}", flush=True)

    with open(os.path.join(a.save_dir, "metrics.json"), "w") as f:
        json.dump({
            "scheme": "int4_per_channel_range",
            "bits": 4,
            "granularity": "per_channel",
            "wikitext2_ppl": float(ppl),
            "seqlen": a.seqlen,
            "dtype_forward": a.dtype,
            "n_quantized_layers": int(n_quantized),
        }, f, indent=2)
    print(f"[Saved] {a.save_dir}/metrics.json", flush=True)


if __name__ == "__main__":
    main()
