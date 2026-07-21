#!/usr/bin/env python3
# eval_ppl.py — Load a saved checkpoint (or HF model) and eval WikiText-2 PPL
#                   with optional per-Linear FP8 (E4M3) activation quantization.
#
# Used by 06_act_quant.sh to reproduce Table VI (weight × activation quant).

import argparse, math, torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import transformers.modeling_utils as modeling_utils
from qtorch_plus.quant import float_quantize

transformers.logging.set_verbosity_error()
DEV = "cuda" if torch.cuda.is_available() else "cpu"


FP8_E4M3_MAX = 240.0  # qtorch_plus float_quantize(exp=4,man=3) finite max (IEEE convention)


def make_act_hook(exp_bits, man_bits):
    """Forward-pre hook: quantize activation with per-token dynamic FP(exp,man) scale.

    Matches the reference implementation used to generate Table 6 paper values:
      amax = |x|.max(dim=-1)                        # per-token
      scale = amax / FP8_MAX                         # per-token dynamic scale
      q = float_quantize((x/scale).clamp(-M, M)) * scale

    This is the standard MX-FP8-style dynamic per-token quant used in modern
    FP8 tensor pipelines. Naive per-element float_quantize (no per-token
    rescale) saturates on activation outliers and inflates PPL.
    """
    fmax = FP8_E4M3_MAX if (exp_bits == 4 and man_bits == 3) else float("inf")

    def hook(module, inputs):
        if not inputs:
            return inputs
        x = inputs[0]
        if not torch.is_tensor(x):
            return inputs
        xf = x.contiguous().float()
        amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = amax / fmax
        q = float_quantize(
            (xf / scale).clamp(-fmax, fmax),
            exp=exp_bits, man=man_bits, rounding="nearest"
        ) * scale
        return (q.to(x.dtype),) + tuple(inputs[1:])
    return hook


def is_lin(m):
    return isinstance(m, (nn.Linear, modeling_utils.Conv1D))


def skip(name, m):
    return (name == "lm_head" or name.endswith(".lm_head") or isinstance(m, nn.Embedding))


@torch.no_grad()
def ppl_wikitext(model, tok, seqlen):
    model.eval()
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt", add_special_tokens=False).input_ids
    n = ids.numel() // seqlen
    dev = next(model.parameters()).device
    nll = 0.0
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
    p.add_argument("--model", required=True, help="HF id or local checkpoint dir")
    p.add_argument("--act_quant", choices=["none", "fp8_e4m3"], default="none")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    a = p.parse_args()

    td = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    print(f"[Load] {a.model}  dtype={a.dtype}  act_quant={a.act_quant}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=td, trust_remote_code=a.trust_remote_code, low_cpu_mem_usage=True).to(DEV)
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True, trust_remote_code=a.trust_remote_code)

    if a.act_quant == "fp8_e4m3":
        hook = make_act_hook(exp_bits=4, man_bits=3)
        k = 0
        for name, mod in model.named_modules():
            if skip(name, mod) or not is_lin(mod):
                continue
            mod.register_forward_pre_hook(hook)
            k += 1
        print(f"[Act quant] FP8-E4M3 hook on {k} Linear/Conv1D modules", flush=True)

    ppl = ppl_wikitext(model, tok, a.seqlen)
    print(f"[Result] wikitext2_ppl = {ppl:.4f}", flush=True)


if __name__ == "__main__":
    main()
