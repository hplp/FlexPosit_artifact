#!/usr/bin/env python3
# mxfp8_group_wise.py
# MXFP8 PG (block-wise, group_size=32, OCP MX-spec) weight-only quantization PPL on WikiText-2.
#
# Element format: fp8_e4m3. Shared scale: E8M0 power-of-two, one per block of 32 along the
# input (last) dim of each Linear/Conv1D weight. Weights only; activations FP16/BF16.
#
# Two backends, auto-selected at runtime:
#   [A] microsoft/microxcaling (`mx`) — canonical OCP MX impl. Preferred if importable.
#         quantize_mx_op(W, specs, elem_format='fp8_e4m3', axes=[-1], block_size=32)
#   [B] qtorch_plus fallback — block-wise E4M3 with corrected format max.
#         A sanity test on float_quantize(exp=4,man=3) decides the E4M3 max convention:
#           max ~240  -> IEEE  -> MXFP8_MAX=240, emax_elem=7
#           max ~448  -> OCP   -> MXFP8_MAX=448, emax_elem=8
#         OCP shared scale: 2^(floor(log2(block_amax)) - emax_elem).
#
# PPL: WikiText-2 raw test, chunked non-overlapping, seqlen 1024 (gpt2-*) / 2048 (else),
#      forward dtype fp16 (bf16 for llama-2-7b). Skip lm_head + embeddings.

import argparse, math, gc, torch
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import transformers.modeling_utils as modeling_utils

transformers.logging.set_verbosity_error()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-12
GROUP = 32

# (short, hf_id, seqlen, trust_remote_code, use_fast, dtype)
MODELS = [
    ("gpt2-large",      "gpt2-large",                          1024, False, True,  torch.float16),
    ("gpt2-xl",         "gpt2-xl",                             1024, False, True,  torch.float16),
    ("phi-2",           "microsoft/phi-2",                     2048, True,  True,  torch.float16),
    ("opt-2.7b",        "facebook/opt-2.7b",                   2048, False, True,  torch.float16),
    ("llama-2-7b",      "meta-llama/Llama-2-7b-hf",            2048, False, True,  torch.bfloat16),
    ("qwen2.5-7b",      "Qwen/Qwen2.5-7B",                     2048, True,  False, torch.float16),
    ("deepseek-llm-7b", "deepseek-ai/deepseek-llm-7b-base",    2048, True,  False, torch.float16),
    ("mistral-7b",      "mistralai/Mistral-7B-v0.1",           2048, False, True,  torch.float16),
    ("qwen2.5-14b",     "Qwen/Qwen2.5-14B",                    2048, True,  False, torch.float16),
]

# Buggy PG numbers reported on yg9bq's server (for the report's context column). NaN if unknown.
BUGGY_PG = {
    "gpt2-large": 27.71, "qwen2.5-7b": 14.10, "mistral-7b": 99.62,
    "llama-2-7b": 139.49, "deepseek-llm-7b": 71.17,
}
# Existing PC MXFP8 (non-PoT, scale=amax/448) for the PG<=PC sanity check.
REF_PC = {
    "gpt2-large": 20.56, "gpt2-xl": 17.94, "phi-2": 11.45, "opt-2.7b": 13.94,
    "qwen2.5-7b": 7.82, "deepseek-llm-7b": 8.34, "mistral-7b": 7.24, "qwen2.5-14b": 6.79,
}

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
BACKEND = None       # "mx" or "qtorch"
_MX_QUANT = None     # callable W->W_q (mx path)
MXFP8_MAX = None     # qtorch path
EMAX_ELEM = None     # qtorch path


def init_backend():
    global BACKEND, _MX_QUANT, MXFP8_MAX, EMAX_ELEM
    # ---- Try microxcaling first ----
    try:
        from mx import finalize_mx_specs, quantize_mx_op
        specs = finalize_mx_specs({
            "scale_bits": 8,
            "w_elem_format": "fp8_e4m3",
            "block_size": GROUP,
            "bfloat": 16,
            "custom_cuda": False,   # pure-torch path: robust, no JIT CUDA compile needed
        })

        def _q(W):
            x = W.detach().float()
            y = quantize_mx_op(x, specs, elem_format="fp8_e4m3", axes=[-1], block_size=GROUP)
            return y.to(W.dtype)

        # smoke test
        _ = _q(torch.randn(8, GROUP, device=DEV))
        BACKEND = "mx"; _MX_QUANT = _q
        print("[backend] microxcaling (mx): fp8_e4m3, block_size=32, axes=[-1], custom_cuda=False", flush=True)
        return
    except Exception as e:
        print(f"[backend] microxcaling unavailable ({type(e).__name__}: {e}); falling back to qtorch_plus", flush=True)

    # ---- qtorch_plus fallback: decide E4M3 max convention ----
    from qtorch_plus.quant import float_quantize
    x = torch.tensor([100., 200., 240., 245., 300., 400., 500., 1000.], device=DEV)
    out = float_quantize(x, exp=4, man=3, rounding="nearest")
    mx_out = float(out.max().item())
    print(f"[sanity] float_quantize(exp=4,man=3) on {x.tolist()} -> {out.tolist()}", flush=True)
    print(f"[sanity] max finite encoded = {mx_out}", flush=True)
    if mx_out <= 300.0:      # ~240 => IEEE convention
        MXFP8_MAX, EMAX_ELEM = 240.0, 7
        print("[sanity] => qtorch uses IEEE E4M3 (max 240): MXFP8_MAX=240, emax_elem=7 (CORRECTED)", flush=True)
    else:                    # ~448 => OCP convention
        MXFP8_MAX, EMAX_ELEM = 448.0, 8
        print("[sanity] => qtorch uses OCP E4M3 (max 448): MXFP8_MAX=448, emax_elem=8", flush=True)
    BACKEND = "qtorch"


@torch.no_grad()
def pg_quant(W):
    """Block-wise (group=32 along input dim) MXFP8 quantization of a 2-D weight."""
    if BACKEND == "mx":
        return _MX_QUANT(W)

    # qtorch fallback
    from qtorch_plus.quant import float_quantize
    out_f, in_f = W.shape
    flat = W.detach().float()
    pad = (GROUP - in_f % GROUP) % GROUP
    if pad:
        flat = F.pad(flat, (0, pad))
    nblk = flat.shape[1] // GROUP
    blocks = flat.view(out_f, nblk, GROUP)
    amax = blocks.abs().amax(dim=2, keepdim=True).clamp(min=EPS)          # [out, nblk, 1]
    shared_exp = torch.floor(torch.log2(amax)) - EMAX_ELEM               # OCP MX shared scale exponent
    scale = torch.pow(2.0, shared_exp)
    q = float_quantize((blocks / scale).clamp(-MXFP8_MAX, MXFP8_MAX),
                       exp=4, man=3, rounding="nearest") * scale
    q = q.view(out_f, nblk * GROUP)[:, :in_f].contiguous()
    return q.to(W.dtype)


def is_lin(m):
    return isinstance(m, (nn.Linear, modeling_utils.Conv1D))


def skip(name, m):
    return (name == "lm_head" or name.endswith(".lm_head")
            or isinstance(m, nn.Embedding))


@torch.no_grad()
def ppl_wikitext(model, tok, seqlen, fwd_dtype):
    model.eval()
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt", add_special_tokens=False).input_ids
    n = ids.numel() // seqlen
    dev = next(model.parameters()).device
    nll = 0.0
    with torch.cuda.amp.autocast(enabled=True, dtype=fwd_dtype):
        for i in range(n):
            b = ids[:, i*seqlen:(i+1)*seqlen].to(dev)
            logits = model(b).logits
            sl = logits[:, :-1, :].contiguous().float()
            lab = b[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lab.view(-1))
            nll += loss.item() * seqlen
    return math.exp(nll / (n * seqlen))


def run_one(short, hf_id, seqlen, trust, use_fast, dtype):
    print(f"\n=== {short}  ({hf_id})  seqlen={seqlen}  dtype={dtype} ===", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=dtype, trust_remote_code=trust, low_cpu_mem_usage=True).to(DEV)
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust, use_fast=use_fast)
    k = 0
    for name, mod in m.named_modules():
        if skip(name, mod) or not is_lin(mod):
            continue
        mod.weight.data = pg_quant(mod.weight)
        k += 1
    ppl = ppl_wikitext(m, tok, seqlen, dtype)
    flag = ""
    if math.isnan(ppl) or ppl > 1000:
        flag = "   <<< GUARD: NaN or PPL>1000"
    print(f"[{short}] MXFP8 PG g={GROUP} ({BACKEND}): quantized {k} layers | PPL = {ppl:.4f}{flag}", flush=True)
    del m, tok
    gc.collect(); torch.cuda.empty_cache()
    return k, ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of short names (default: all 9)")
    ap.add_argument("--csv_out", default=None,
                    help="If set, append (model,method,wikitext2_ppl) rows to this CSV. "
                         "Header written iff file does not exist.")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None

    print("==================== MXFP8 PG (block-wise, group_size=32, OCP MX-spec) ====================", flush=True)
    init_backend()

    results = []
    for short, hf_id, seqlen, trust, use_fast, dtype in MODELS:
        if only and short not in only:
            continue
        try:
            k, ppl = run_one(short, hf_id, seqlen, trust, use_fast, dtype)
        except Exception as e:
            print(f"[{short}] FAILED: {repr(e)}", flush=True)
            k, ppl = -1, float("nan")
            gc.collect(); torch.cuda.empty_cache()
        results.append((short, k, ppl))

    print(f"\n========= MXFP8 PG g={GROUP} PPL (WikiText-2) | backend={BACKEND} =========", flush=True)
    print(f"{'model':18s} {'PG-g32 PPL':>11s} {'was-buggy PG':>13s} {'ref PC':>8s} {'PG<=PC?':>8s}")
    for short, k, ppl in results:
        buggy = BUGGY_PG.get(short, float("nan"))
        refpc = REF_PC.get(short, float("nan"))
        ok = "" if math.isnan(refpc) or math.isnan(ppl) else ("yes" if ppl <= refpc + 1e-6 else "NO")
        bs = f"{buggy:13.2f}" if not math.isnan(buggy) else f"{'n.a.':>13s}"
        ps = f"{refpc:8.2f}" if not math.isnan(refpc) else f"{'n.a.':>8s}"
        print(f"{short:18s} {ppl:11.4f} {bs} {ps} {ok:>8s}")

    if a.csv_out:
        import csv, os
        write_header = not os.path.exists(a.csv_out)
        with open(a.csv_out, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["model", "method", "wikitext2_ppl"])
            for short, _, ppl in results:
                if not math.isnan(ppl):
                    w.writerow([short, "mxfp8_pg_g32", f"{ppl:.4f}"])
        print(f"[csv] wrote {len([r for r in results if not math.isnan(r[2])])} MXFP8-PG rows to {a.csv_out}", flush=True)


if __name__ == "__main__":
    main()
