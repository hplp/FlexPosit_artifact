#!/usr/bin/env python3
# posit_quant_pc.py
# Fixed-nsize per-layer; per-channel search for (scale, es).
# GPU-friendly: batched per-channel search (no float(tensor) syncs inside loops).
# Saves model + log + metrics.json (chunked non-overlapping PPL).

import argparse, json, os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import transformers.modeling_utils as modeling_utils

from qtorch_plus.quant import posit_quantize, float_quantize

# GPTQ-Hessian helpers are imported lazily inside the gptq_hessian code path
# (below) so this driver runs cleanly without posit_gptq_hessian.py present.
# Users who pass --quant_mode gptq_hessian will hit an ImportError at that point.

transformers.logging.set_verbosity_error()
EPS = 1e-8

# -------------------- Model presets --------------------
MODEL_PRESETS = {
    
    "gpt2":         {"hf_id": "gpt2"},
    "gpt2-medium":  {"hf_id": "gpt2-medium"},
    "gpt2-large":   {"hf_id": "gpt2-large"},
    "gpt2-xl":      {"hf_id": "gpt2-xl"},

    "opt-125m":     {"hf_id": "facebook/opt-125m"},
    "opt-350m":     {"hf_id": "facebook/opt-350m"},
    "opt-1.3b":     {"hf_id": "facebook/opt-1.3b"},
    "opt-2.7b":     {"hf_id": "facebook/opt-2.7b"},
    "opt-6.7b":     {"hf_id": "facebook/opt-6.7b"},

    "bloom-7b1":    {"hf_id": "bigscience/bloom-7b1"},

    "phi-2":        {"hf_id": "microsoft/phi-2"},
    "yi-6b":        {"hf_id": "01-ai/Yi-6B", "trust_remote_code": True},
    "llama-2-7b":   {"hf_id": "meta-llama/Llama-2-7b-hf"},
    "llama-3-8b":   {"hf_id": "meta-llama/Meta-Llama-3-8B"},
    
    "qwen2.5-14b": {
    "hf_id": "Qwen/Qwen2.5-14B",
    "trust_remote_code": True,
    "use_fast_tokenizer": False,
    "requires_auth": False,
},
    "llama-2-7b":   {"hf_id": "meta-llama/Llama-2-7b-hf"},
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B",
        "trust_remote_code": True,
        "use_fast_tokenizer": False,
        "requires_auth": False,
    },
    "qwen2-7b": {
        "hf_id": "Qwen/Qwen2-7B",
        "trust_remote_code": True,
        "use_fast_tokenizer": False,
        "requires_auth": False,
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "trust_remote_code": False,
        "use_fast_tokenizer": True,
        "requires_auth": False,
    },
    "deepseek-llm-7b": {
        "hf_id": "deepseek-ai/deepseek-llm-7b-base",
        "trust_remote_code": True,
        "use_fast_tokenizer": False,
        "requires_auth": False,
    },
    "phi-3-mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "trust_remote_code": True,
        "use_fast_tokenizer": True,
        "requires_auth": False,
    },
    "phi-3-small": {
        "hf_id": "microsoft/Phi-3-small-8k-instruct",
        "trust_remote_code": True,
        "use_fast_tokenizer": True,
        "requires_auth": False,
    },
    # GATED
    "llama-2-13b": {
        "hf_id": "meta-llama/Llama-2-13b-hf",
        "trust_remote_code": False,
        "use_fast_tokenizer": True,
        "requires_auth": True,
    },
}

# -------------------- Args --------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_PRESETS.keys()), default="mistral-7b")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # HF token (for gated models); can also use env HF_TOKEN
    p.add_argument("--hf_token", default=None)

    # fixed nsize (per layer), per-channel search for (scale, es)
    p.add_argument("--nsize", type=int, default=4)
    p.add_argument(
        "--weight_format",
        choices=[
            "posit4", "posit5",
        ],
        default="posit4",
        help="Weight format: posit4 or posit5 (per-channel PoT scale search)."
    )
    p.add_argument("--olive_w_low", type=int, default=75,
                   help="OliVe per-channel alpha-search lower bound (i*0.01), reference default 75.")
    p.add_argument("--olive_w_up", type=int, default=150,
                   help="OliVe per-channel alpha-search upper bound (i*0.01), reference default 150.")
    p.add_argument("--group_size", type=int, default=128,
                   help="Group size for group-wise weight formats (unused by posit4/posit5 baselines) (OCP MXFP4 typically uses 32).")
    p.add_argument("--search_metric", choices=["sqnr", "mse"], default="sqnr",
                   help="Search objective (legacy).")
    p.add_argument("--es_candidates", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--log2_min", type=int, default=-8)
    p.add_argument("--log2_max", type=int, default=9)

    # GPU batching for per-channel search (important!)
    p.add_argument("--ch_batch", type=int, default=64,
                   help="How many output channels to search at once per layer. Increase if you have headroom.")

    # activation quant (optional; off by default)
    p.add_argument("--use_act_quant", action="store_true", default=False)
    p.add_argument("--act_exp", type=int, default=4)
    p.add_argument("--act_man", type=int, default=3)

    # quantization scope
    p.add_argument("--skip_lm_head", action="store_true", default=True)
    p.add_argument("--quantize_embeddings", action="store_true", default=False)

    # Chunked non-overlapping WikiText-2 PPL
    p.add_argument("--ppl_seqlen", type=int, default=2048)

    # baseline = current batched SQNR Posit; gptq_hessian = GPTQ Cholesky + column error propagation
    p.add_argument("--quant_mode", choices=["baseline", "gptq_hessian"], default="baseline")

    # WikiText-2 *train* calibration for Hessian (forward hooks), GPTQ-style online H
    p.add_argument("--hessian_calib_nsamples", type=int, default=128,
                   help="Number of non-overlapping train windows for H.")
    p.add_argument("--hessian_calib_seqlen", type=int, default=1024)
    p.add_argument("--hessian_percdamp", type=float, default=0.01)
    p.add_argument("--gptq_blocksize", type=int, default=128)
    p.add_argument("--hessian_seed", type=int, default=0)

    # saving
    p.add_argument("--save_dir", required=True)
    p.add_argument("--save_log_name", default="quant_log.json")
    return p.parse_args()

def get_torch_dtype(tag: str):
    if tag == "fp16":
        return torch.float16
    if tag == "bf16":
        return torch.bfloat16
    return torch.float32

# -------------------- Act quant hook --------------------
def make_act_hook(use_act: bool, exp_bits: int, man_bits: int):
    if not use_act:
        def passthrough(_m, inputs): return inputs
        return passthrough

    def linear_activation(x: torch.Tensor):
        x_fp32 = x.float()  # float_quantize requires fp32 input
        q_fp32 = float_quantize(x_fp32, exp=exp_bits, man=man_bits, rounding="nearest")
        return q_fp32.to(x.dtype)  # cast back to module's operating dtype

    def hook(_m, inputs):
        return (linear_activation(inputs[0]),)

    return hook

# -------------------- Layer filters --------------------
def is_quant_linear(mod: nn.Module):
    if isinstance(mod, nn.Linear):
        return True
    if isinstance(mod, modeling_utils.Conv1D):
        return True
    name = mod.__class__.__name__.lower()
    if "linear" in name and hasattr(mod, "weight") and isinstance(getattr(mod, "weight"), torch.Tensor):
        return True
    return False

def should_skip_layer(name: str, mod: nn.Module, skip_lm_head: bool, quantize_embeddings: bool):
    if skip_lm_head and (name == "lm_head" or name.endswith(".lm_head")):
        return True
    if isinstance(mod, nn.Embedding) and not quantize_embeddings:
        return True
    return False

def summarize_format_usage(ch_meta):
    """
    Summarize per-channel selected format statistics.
    Supports:
      - mixed meta: {"selected_format": "..."}
      - single-format meta: {"format": "..."}
    """
    total = len(ch_meta)
    counts = {}
    for item in ch_meta:
        fmt = item.get("selected_format", item.get("format", "unknown"))
        counts[fmt] = counts.get(fmt, 0) + 1
    ratios = {k: (v / total if total > 0 else 0.0) for k, v in counts.items()}
    return {"total_channels": total, "counts": counts, "ratios": ratios}

# -------------------- Group-wise INT-4 helper (legacy) --------------------
BITMOD3_CODEBOOKS = {
    "fp3_er_pos": [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
    "fp3_er_neg": [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0],
    "fp3_ea_pos": [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0],
    "fp3_ea_neg": [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0],
}

BITMOD4_CODEBOOKS = {
    "fp4_er_pos": [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    "fp4_er_neg": [-12.0, -10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
    "fp4_ea_pos": [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0],
    "fp4_ea_neg": [-16.0, -12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
}

@torch.no_grad()
def _quantize_with_codebook_per_row(x: torch.Tensor, codebook_vals):
    """
    x: [B, K] fp32.
    Per-row dynamic scale + nearest codebook value quantization.
    """
    dev = x.device
    cb = torch.tensor(codebook_vals, device=dev, dtype=torch.float32)  # [L]
    qmax = torch.max(torch.abs(cb))
    rmax = torch.amax(torch.abs(x), dim=1, keepdim=True)               # [B,1]
    scale = (rmax / qmax).clamp(min=1e-5, max=1e4)
    xn = x / scale
    dist = torch.abs(xn.unsqueeze(-1) - cb.view(1, 1, -1))             # [B,K,L]
    idx = torch.argmin(dist, dim=-1)                                    # [B,K]
    qn = cb[idx]
    return qn * scale

# -------------------- GPU-friendly per-channel search --------------------
@torch.no_grad()
def _per_channel_sqnr_search_batched(
    flat: torch.Tensor,
    nsize: int,
    es_cands,
    sweep_scales,
    ch_batch: int,
    return_quantized: bool,
):
    """
    flat: [Cout, K] fp32 on device.
    If return_quantized: returns (q_out [Cout,K], per_ch_meta list).
    Else: returns (scale_vec [Cout], es_vec [Cout] int64, per_ch_meta list).
    """
    dev = flat.device
    Cout, K = flat.shape
    max_es = max(0, nsize - 1)
    es_list = [int(e) for e in es_cands if int(e) <= max_es] or [0]
    scales = torch.tensor([float(s) for s in sweep_scales], device=dev, dtype=torch.float32)

    q_out = torch.empty_like(flat) if return_quantized else None
    per_ch_meta = []
    scale_out = torch.empty((Cout,), device=dev, dtype=torch.float32)
    es_out = torch.empty((Cout,), device=dev, dtype=torch.int64)

    for c0 in range(0, Cout, ch_batch):
        c1 = min(Cout, c0 + ch_batch)
        X = flat[c0:c1]
        B = X.size(0)
        sp = torch.sum(X * X, dim=1) + EPS

        best_sqnr = torch.full((B,), -1e30, device=dev, dtype=torch.float32)
        best_q = torch.zeros((B, K), device=dev, dtype=torch.float32)
        best_es = torch.zeros((B,), device=dev, dtype=torch.int32)
        best_l2 = torch.zeros((B,), device=dev, dtype=torch.int32)

        for s in scales:
            Xs = X * s
            for es in es_list:
                q = posit_quantize(Xs, nsize=nsize, es=int(es), scale=1.0) / s
                noise = torch.sum((X - q) ** 2, dim=1) + EPS
                sqnr = 10.0 * torch.log10(sp / noise)
                mask = sqnr > best_sqnr
                if mask.any():
                    best_sqnr = torch.where(mask, sqnr, best_sqnr)
                    best_q = torch.where(mask[:, None], q, best_q)
                    best_es = torch.where(mask, torch.tensor(es, device=dev, dtype=torch.int32), best_es)
                    l2 = int(round(math.log2(float(s.item()))))
                    best_l2 = torch.where(mask, torch.tensor(l2, device=dev, dtype=torch.int32), best_l2)

        if return_quantized:
            q_out[c0:c1] = best_q

        sch = torch.pow(2.0, best_l2.to(dtype=torch.float32))
        scale_out[c0:c1] = sch
        es_out[c0:c1] = best_es.to(torch.int64)

        best_sqnr_cpu = best_sqnr.detach().cpu().tolist()
        best_es_cpu = best_es.detach().cpu().tolist()
        best_l2_cpu = best_l2.detach().cpu().tolist()
        for i in range(B):
            per_ch_meta.append({
                "channel": int(c0 + i),
                "sqnr": float(best_sqnr_cpu[i]),
                "log2_scale": int(best_l2_cpu[i]),
                "es": int(best_es_cpu[i]),
                "nsize": int(nsize),
            })

    if return_quantized:
        return q_out, per_ch_meta
    return scale_out, es_out, per_ch_meta


@torch.no_grad()
def per_channel_scales_es_batched(
    layer_weight: torch.Tensor,
    nsize: int,
    es_cands,
    sweep_scales,
    ch_batch: int,
):
    """SQNR-optimal per-output-channel scale (float) and es (int) for GPTQ+Posit path."""
    dev = layer_weight.device
    W = layer_weight.detach().to(dtype=torch.float32, device=dev)
    flat = W.view(W.size(0), -1)
    scale_vec, es_vec, meta = _per_channel_sqnr_search_batched(
        flat, nsize, es_cands, sweep_scales, ch_batch, return_quantized=False
    )
    return scale_vec, es_vec, meta


@torch.no_grad()
def per_channel_quantize_fixed_nsize_batched(layer_weight: torch.Tensor,
                                             nsize: int,
                                             es_cands,
                                             sweep_scales,
                                             ch_batch: int):
    """
    layer_weight: [Cout, K] tensor on model device.
    Returns:
      q_w: quantized weight, same shape/device/dtype as layer_weight
      per_ch_meta: list of dicts (channel, log2_scale, es, sqnr)  (small)
    """
    dev = layer_weight.device
    out_dtype = layer_weight.dtype
    W = layer_weight.detach().to(dtype=torch.float32, device=dev)
    flat = W.view(W.size(0), -1)
    q_out, per_ch_meta = _per_channel_sqnr_search_batched(
        flat, nsize, es_cands, sweep_scales, ch_batch, return_quantized=True
    )
    q_w = q_out.view_as(W).to(dtype=out_dtype, device=dev)
    return q_w, per_ch_meta

@torch.no_grad()
def per_channel_int_grouped_sqnr_batched(layer_weight: torch.Tensor,
                                    sweep_scales,
                                    ch_batch: int,
                                    search_metric: str = "sqnr",
                                    codebooks=None,
                                    format_label: str = "int4_gp"):
    """
    Per-channel legacy INT-4 search (SQNR or MSE objective):
      datatype candidates come from codebooks.
      sweep over power-of-two pre-scales.
    """
    if codebooks is None:
        codebooks = BITMOD4_CODEBOOKS
    dev = layer_weight.device
    out_dtype = layer_weight.dtype
    W = layer_weight.detach().to(dtype=torch.float32, device=dev)
    Cout = W.size(0)
    flat = W.view(Cout, -1)
    K = flat.size(1)

    scales_list = [float(s) for s in sweep_scales]
    scales_tensor = torch.tensor(scales_list, device=dev, dtype=torch.float32)
    log2_scales_list = [int(round(math.log2(s))) for s in scales_list]
    variants = list(codebooks.keys())

    q_out = torch.empty_like(flat)
    per_ch_meta = []

    for c0 in range(0, Cout, ch_batch):
        c1 = min(Cout, c0 + ch_batch)
        X = flat[c0:c1]
        B = X.size(0)
        sp = torch.sum(X * X, dim=1) + EPS

        if search_metric == "sqnr":
            best_score = torch.full((B,), -1e30, device=dev, dtype=torch.float32)
        else:
            best_score = torch.full((B,), 1e30, device=dev, dtype=torch.float32)
        best_q = torch.zeros((B, K), device=dev, dtype=torch.float32)
        best_l2 = torch.zeros((B,), device=dev, dtype=torch.int32)
        best_vid = torch.zeros((B,), device=dev, dtype=torch.int32)

        for sidx, s in enumerate(scales_tensor):
            Xs = X * s
            for vidx, vname in enumerate(variants):
                q = _quantize_with_codebook_per_row(Xs, codebooks[vname]) / s
                noise = torch.sum((X - q) ** 2, dim=1) + EPS
                sqnr = 10.0 * torch.log10(sp / noise)
                if search_metric == "sqnr":
                    score = sqnr
                    mask = score > best_score
                else:
                    score = noise / float(K)  # per-channel MSE
                    mask = score < best_score
                if mask.any():
                    best_score = torch.where(mask, score, best_score)
                    best_q = torch.where(mask[:, None], q, best_q)
                    best_l2 = torch.where(mask, torch.tensor(log2_scales_list[sidx], device=dev, dtype=torch.int32), best_l2)
                    best_vid = torch.where(mask, torch.tensor(vidx, device=dev, dtype=torch.int32), best_vid)

        q_out[c0:c1] = best_q
        score_cpu = best_score.detach().cpu().tolist()
        l2_cpu = best_l2.detach().cpu().tolist()
        vid_cpu = best_vid.detach().cpu().tolist()
        for i in range(B):
            item = {
                "channel": int(c0 + i),
                "log2_scale": int(l2_cpu[i]),
                "variant": variants[int(vid_cpu[i])],
                "format": format_label,
            }
            if search_metric == "sqnr":
                item["sqnr"] = float(score_cpu[i])
            else:
                item["mse"] = float(score_cpu[i])
            per_ch_meta.append(item)

    q_w = q_out.view_as(W).to(dtype=out_dtype, device=dev)
    return q_w, per_ch_meta

@torch.no_grad()
def per_channel_int4_grouped_sqnr_batched(layer_weight: torch.Tensor,
                                     sweep_scales,
                                     ch_batch: int,
                                     search_metric: str = "sqnr"):
    return per_channel_int_grouped_sqnr_batched(
        layer_weight=layer_weight,
        sweep_scales=sweep_scales,
        ch_batch=ch_batch,
        search_metric=search_metric,
        codebooks=BITMOD4_CODEBOOKS,
        format_label="int4_gp",
    )

@torch.no_grad()
def per_channel_int3_grouped_sqnr_batched(layer_weight: torch.Tensor,
                                     sweep_scales,
                                     ch_batch: int,
                                     search_metric: str = "sqnr"):
    return per_channel_int_grouped_sqnr_batched(
        layer_weight=layer_weight,
        sweep_scales=sweep_scales,
        ch_batch=ch_batch,
        search_metric=search_metric,
        codebooks=BITMOD3_CODEBOOKS,
        format_label="int3_gp",
    )

@torch.no_grad()
def per_channel_gw_int_sqnr_batched(layer_weight: torch.Tensor,
                                              sweep_scales,
                                              ch_batch: int,
                                              group_size: int = 128,
                                              search_metric: str = "sqnr",
                                              codebooks=None,
                                              format_label: str = "int4_gp_g128"):
    """
    Group-wise legacy INT-4 search (SQNR or MSE objective).
    - Each output channel is split into contiguous groups along input dimension.
    - For each group, search best (scale, bitmod subtype) by SQNR.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if codebooks is None:
        codebooks = BITMOD4_CODEBOOKS

    dev = layer_weight.device
    out_dtype = layer_weight.dtype
    W = layer_weight.detach().to(dtype=torch.float32, device=dev)
    Cout = W.size(0)
    flat = W.view(Cout, -1)
    K = flat.size(1)
    num_groups = (K + group_size - 1) // group_size

    scales_tensor = torch.tensor([float(s) for s in sweep_scales], device=dev, dtype=torch.float32)
    variants = list(codebooks.keys())

    q_out = torch.empty_like(flat)
    per_ch_meta = []

    for c0 in range(0, Cout, ch_batch):
        c1 = min(Cout, c0 + ch_batch)
        X = flat[c0:c1]  # [B, K]
        B = X.size(0)
        Q = torch.empty_like(X)
        score_sum = torch.zeros((B,), device=dev, dtype=torch.float32)

        for g in range(num_groups):
            s = g * group_size
            e = min((g + 1) * group_size, K)
            Xg = X[:, s:e]  # [B, G]
            sp = torch.sum(Xg * Xg, dim=1) + EPS

            if search_metric == "sqnr":
                best_score = torch.full((B,), -1e30, device=dev, dtype=torch.float32)
            else:
                best_score = torch.full((B,), 1e30, device=dev, dtype=torch.float32)
            best_q = torch.zeros_like(Xg)

            # NOTE: the per-group scale sweep is a no-op here — _quantize_with_codebook_per_row
            # self-normalizes each group by its own amax, so every sweep scale yields a
            # bit-identical q. We therefore evaluate variants once (scale=1), which is
            # output-identical to the old triple loop but ~len(sweep_scales)x faster.
            for vname in variants:
                q = _quantize_with_codebook_per_row(Xg, codebooks[vname])
                noise = torch.sum((Xg - q) ** 2, dim=1) + EPS
                sqnr = 10.0 * torch.log10(sp / noise)
                if search_metric == "sqnr":
                    score = sqnr
                    mask = score > best_score
                else:
                    score = noise / float(e - s)  # per-group MSE
                    mask = score < best_score
                if mask.any():
                    best_score = torch.where(mask, score, best_score)
                    best_q = torch.where(mask[:, None], q, best_q)

            Q[:, s:e] = best_q
            score_sum += best_score

        q_out[c0:c1] = Q
        score_avg_cpu = (score_sum / float(num_groups)).detach().cpu().tolist()
        for i in range(B):
            item = {
                "channel": int(c0 + i),
                "format": format_label,
                "group_size": int(group_size),
                "num_groups": int(num_groups),
            }
            if search_metric == "sqnr":
                item["sqnr_avg"] = float(score_avg_cpu[i])
            else:
                item["mse_avg"] = float(score_avg_cpu[i])
            per_ch_meta.append(item)

    q_w = q_out.view_as(W).to(dtype=out_dtype, device=dev)
    return q_w, per_ch_meta

@torch.no_grad()
def per_channel_gw_int4_sqnr_batched(layer_weight: torch.Tensor,
                                               sweep_scales,
                                               ch_batch: int,
                                               group_size: int = 128,
                                               search_metric: str = "sqnr"):
    return per_channel_gw_int_sqnr_batched(
        layer_weight=layer_weight,
        sweep_scales=sweep_scales,
        ch_batch=ch_batch,
        group_size=group_size,
        search_metric=search_metric,
        codebooks=BITMOD4_CODEBOOKS,
        format_label="int4_gp_g128",
    )

@torch.no_grad()
def per_channel_gw_int3_sqnr_batched(layer_weight: torch.Tensor,
                                               sweep_scales,
                                               ch_batch: int,
                                               group_size: int = 128,
                                               search_metric: str = "sqnr"):
    return per_channel_gw_int_sqnr_batched(
        layer_weight=layer_weight,
        sweep_scales=sweep_scales,
        ch_batch=ch_batch,
        group_size=group_size,
        search_metric=search_metric,
        codebooks=BITMOD3_CODEBOOKS,
        format_label="int3_gp_g128",
    )

# MXFP4 (OCP-style): FP4 E2M1 via qtorch float_quantize(exp=2, man=1), microblock-wise scale search.
MXFP4_EXP = 2
MXFP4_MAN = 1

@torch.no_grad()
def per_channel_groupwise_mxfp4_e2m1_sqnr_batched(layer_weight: torch.Tensor,
                                                  sweep_scales,
                                                  ch_batch: int,
                                                  group_size: int,
                                                  search_metric: str = "sqnr",
                                                  format_label: str = "mxfp4"):
    """
    Microblock-wise MXFP4-style weight quant: each contiguous group of `group_size` weights
    along the input dimension shares a per-group power-of-two scale; values are FP4 E2M1.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")

    dev = layer_weight.device
    out_dtype = layer_weight.dtype
    W = layer_weight.detach().to(dtype=torch.float32, device=dev)
    Cout = W.size(0)
    flat = W.view(Cout, -1)
    K = flat.size(1)
    num_groups = (K + group_size - 1) // group_size

    scales_tensor = torch.tensor([float(s) for s in sweep_scales], device=dev, dtype=torch.float32)

    q_out = torch.empty_like(flat)
    per_ch_meta = []

    for c0 in range(0, Cout, ch_batch):
        c1 = min(Cout, c0 + ch_batch)
        X = flat[c0:c1]
        B = X.size(0)
        Q = torch.empty_like(X)
        score_sum = torch.zeros((B,), device=dev, dtype=torch.float32)

        sc = scales_tensor.view(1, -1, 1)  # [1, S, 1]
        for g in range(num_groups):
            s = g * group_size
            e = min((g + 1) * group_size, K)
            Xg = X[:, s:e]                       # [B, gs]
            gs = e - s
            sp = torch.sum(Xg * Xg, dim=1) + EPS  # [B]

            # Vectorized scale sweep: evaluate all candidate scales at once.
            Xs = Xg.unsqueeze(1) * sc            # [B, S, gs]
            q = float_quantize(
                Xs, exp=MXFP4_EXP, man=MXFP4_MAN, rounding="nearest"
            ) / sc                                # [B, S, gs]
            noise = torch.sum((Xg.unsqueeze(1) - q) ** 2, dim=2) + EPS  # [B, S]
            if search_metric == "sqnr":
                score = 10.0 * torch.log10(sp.unsqueeze(1) / noise)     # [B, S]
                best_idx = torch.argmax(score, dim=1)                   # [B]
            else:
                score = noise / float(gs)
                best_idx = torch.argmin(score, dim=1)
            best_score = score.gather(1, best_idx[:, None]).squeeze(1)  # [B]
            best_q = q.gather(1, best_idx.view(-1, 1, 1).expand(-1, 1, gs)).squeeze(1)

            Q[:, s:e] = best_q
            score_sum += best_score

        q_out[c0:c1] = Q
        score_avg_cpu = (score_sum / float(num_groups)).detach().cpu().tolist()
        for i in range(B):
            item = {
                "channel": int(c0 + i),
                "format": format_label,
                "group_size": int(group_size),
                "num_groups": int(num_groups),
                "fp4": "e2m1",
            }
            if search_metric == "sqnr":
                item["sqnr_avg"] = float(score_avg_cpu[i])
            else:
                item["mse_avg"] = float(score_avg_cpu[i])
            per_ch_meta.append(item)

    q_w = q_out.view_as(W).to(dtype=out_dtype, device=dev)
    return q_w, per_ch_meta


# ---------------------------------------------------------------------------
# OliVe (ISCA'23) Outlier-Victim-Pair 4-bit weight quantization.
# Faithful port of clevercool/ANT-Quantization olive_quantization/antquant/
# quant_modules.py (Quantizer): adaptive int/flint normal codebook + abfloat
# outlier codebook, per-channel alpha MSE search, and outlier-victim-pair
# encoding (an outlier zeroes its pair partner). Weight-only, per-output-channel,
# signed 4-bit. Defaults match their LLM config: w_low=75, w_up=150 (step 2),
# x_max = max(|mean+3std|, |mean-3std|), threshold between normal/outlier = 32.
# ---------------------------------------------------------------------------
def _olive_int_grid(device):
    B = 3  # 4-bit signed -> 3 magnitude bits
    vals = [0.0]
    for i in range(1, 2 ** B):
        vals += [float(i), -float(i)]
    t = torch.tensor(sorted(vals), device=device, dtype=torch.float32)
    t *= 32.0 / (2 ** B)
    return t


def _olive_flint_grid(device, exp_base=0):
    value_bit = 3
    neg_exp_num = value_bit - 1
    pos_exp_num = value_bit - 1
    exp_max = pos_exp_num + exp_base
    vals = [0.0]
    for i in range(0, neg_exp_num + 1):
        exp_bit = i + 2
        exp_value = -(exp_bit - 1)
        mant_bit = value_bit - exp_bit
        for j in range(int(2 ** mant_bit)):
            v = 2 ** exp_value * (1 + 2 ** (-mant_bit) * j)
            vals += [v, -v]
    exp_bit = 2
    mant_bit = value_bit - exp_bit
    for j in range(int(2 ** mant_bit)):
        v = 2 ** (exp_base) * (1 + 2 ** (-mant_bit) * j)
        vals += [v, -v]
    for i in range(1, pos_exp_num):
        exp_bit = i + 2
        mant_bit = value_bit - exp_bit
        for j in range(int(2 ** mant_bit)):
            v = 2 ** (i + exp_base) * (1 + 2 ** (-mant_bit) * j)
            vals += [v, -v]
    vals += [2 ** exp_max, -(2 ** exp_max)]
    t = torch.tensor(sorted(vals), device=device, dtype=torch.float32)
    t *= 32.0 / (2 ** exp_max)
    return t


def _olive_outlier_grid(device, exp_bit=2, exp_base=5):
    value_bit = 3
    mant_bit = value_bit - exp_bit
    vals = []
    for i in range(exp_base, exp_base + 2 ** exp_bit):
        for j in range(int(2 ** mant_bit)):
            if i == exp_base and j == 0:
                continue
            v = 2 ** i * (1 + 2 ** (-mant_bit) * j)
            vals += [v, -v]
    return torch.tensor(sorted(vals), device=device, dtype=torch.float32)


def _nearest_grid(x, grid):
    """Nearest-neighbour quantize x onto sorted 1-D grid (returns dequantized values)."""
    idx = torch.searchsorted(grid, x).clamp(0, grid.numel() - 1)
    idx_lo = (idx - 1).clamp(0, grid.numel() - 1)
    q_hi = grid[idx]
    q_lo = grid[idx_lo]
    return torch.where((x - q_lo).abs() <= (q_hi - x).abs(), q_lo, q_hi)


def _olive_ovp_zero(q):
    """Outlier-victim-pair zeroing on rows of q [B, K] (pairs along K, row-major).
    An outlier (|q|>32) forces its pair-partner to 0. Matches the reference logic."""
    B, K = q.shape
    flat = q.reshape(-1)
    mask = flat.abs() > 32.0
    victim_odd = torch.roll(mask, 1, -1).clone()
    victim_odd[::2] = False
    victim_even = torch.roll(mask & (~victim_odd), -1, -1).clone()
    victim_even[1::2] = False
    victim = victim_even | victim_odd
    out = flat * (~victim)
    return out.view(B, K), mask.view(B, K)


@torch.no_grad()
def _olive_quant_one_mode(X, normal_grid, outlier_grid, base_alpha, lb, ub):
    """Per-channel OliVe quant for one normal codebook. X:[B,K], base_alpha:[B].
    Returns best (dequantized q [B,K], per-row mse [B], outlier-count [B])."""
    dev = X.device
    gmax = normal_grid.abs().max()
    combined = torch.cat([normal_grid, outlier_grid]).sort().values
    B, K = X.shape
    best_mse = torch.full((B,), 1e30, device=dev)
    best_q = torch.zeros_like(X)
    best_nout = torch.zeros((B,), device=dev)
    for i in range(lb, ub, 2):
        alpha = base_alpha * (i * 0.01)
        scale = (alpha / gmax).clamp_min(EPS)[:, None]
        q = _nearest_grid(X / scale, combined)
        qv, omask = _olive_ovp_zero(q)
        deq = qv * scale
        mse = ((deq - X) ** 2).mean(1)
        upd = mse < best_mse
        if upd.any():
            best_mse = torch.where(upd, mse, best_mse)
            best_q = torch.where(upd[:, None], deq, best_q)
            best_nout = torch.where(upd, omask.float().sum(1), best_nout)
    return best_q, best_mse, best_nout


@torch.no_grad()
def per_channel_olive4_batched(layer_weight: torch.Tensor, ch_batch: int,
                               w_low: int = 75, w_up: int = 150,
                               format_label: str = "olive4"):
    dev = layer_weight.device
    out_dtype = layer_weight.dtype
    W = layer_weight.detach().to(dtype=torch.float32, device=dev)
    Cout = W.size(0)
    flat = W.view(Cout, -1)
    int_grid = _olive_int_grid(dev)
    flint_grid = _olive_flint_grid(dev)
    outlier_grid = _olive_outlier_grid(dev)

    # adaptive int/flint: choose per-TENSOR by total MSE (as in reference)
    mode_results = {}
    for mode, ng in (("int", int_grid), ("flint", flint_grid)):
        q_full = torch.empty_like(flat)
        nout_full = torch.empty((Cout,), device=dev)
        mse_total = 0.0
        for c0 in range(0, Cout, ch_batch):
            c1 = min(Cout, c0 + ch_batch)
            X = flat[c0:c1]
            mean = X.mean(1)
            std = X.std(1)
            base_alpha = torch.maximum((mean + 3 * std).abs(), (mean - 3 * std).abs()).clamp_min(EPS)
            q, mse, nout = _olive_quant_one_mode(X, ng, outlier_grid, base_alpha, w_low, w_up)
            q_full[c0:c1] = q
            nout_full[c0:c1] = nout
            mse_total += float(mse.sum().item())
        mode_results[mode] = (q_full, nout_full, mse_total)

    chosen = min(mode_results, key=lambda m: mode_results[m][2])
    q_full, nout_full, _ = mode_results[chosen]

    nout_cpu = nout_full.detach().cpu().tolist()
    per_ch_meta = [{"channel": int(i), "format": format_label,
                    "olive_mode": chosen, "num_outliers": int(nout_cpu[i])}
                   for i in range(Cout)]
    q_w = q_full.view_as(W).to(dtype=out_dtype, device=dev)
    return q_w, per_ch_meta


@torch.no_grad()
def per_channel_mix_int4_positN_sqnr_batched(layer_weight: torch.Tensor,
                                                 posit_nsize: int,
                                                 posit_label: str,
                                                 es_cands,
                                                 sweep_scales,
                                                 ch_batch: int):
    """
    Per-channel mixed search:
      1) best posit-N
      2) best bitmod4
      3) pick higher SQNR for each channel
    """
    q_pos, meta_pos = per_channel_quantize_fixed_nsize_batched(
        layer_weight, nsize=posit_nsize, es_cands=es_cands, sweep_scales=sweep_scales, ch_batch=ch_batch
    )
    q_bm, meta_bm = per_channel_int4_grouped_sqnr_batched(
        layer_weight, sweep_scales=sweep_scales, ch_batch=ch_batch
    )

    dev = layer_weight.device
    flat_pos = q_pos.view(q_pos.size(0), -1)
    flat_bm = q_bm.view(q_bm.size(0), -1)
    sqnr_pos = torch.tensor([m["sqnr"] for m in meta_pos], device=dev, dtype=torch.float32)
    sqnr_bm = torch.tensor([m["sqnr"] for m in meta_bm], device=dev, dtype=torch.float32)
    pick_pos = sqnr_pos >= sqnr_bm
    q_mix = torch.where(pick_pos[:, None], flat_pos, flat_bm).view_as(q_pos).to(dtype=layer_weight.dtype, device=dev)

    pick_pos_cpu = pick_pos.detach().cpu().tolist()
    per_ch_meta = []
    for i in range(len(meta_pos)):
        per_ch_meta.append({
            "channel": int(i),
            "selected_format": posit_label if pick_pos_cpu[i] else "bitmod4",
            "sqnr_selected": float(meta_pos[i]["sqnr"] if pick_pos_cpu[i] else meta_bm[i]["sqnr"]),
            f"sqnr_{posit_label}": float(meta_pos[i]["sqnr"]),
            "sqnr_bitmod4": float(meta_bm[i]["sqnr"]),
            f"{posit_label}_log2_scale": int(meta_pos[i]["log2_scale"]),
            f"{posit_label}_es": int(meta_pos[i]["es"]),
            "bitmod_log2_scale": int(meta_bm[i]["log2_scale"]),
            "variant": meta_bm[i]["variant"],
        })
    return q_mix, per_ch_meta

@torch.no_grad()
def per_channel_mix_bitmod4_posit4_sqnr_batched(layer_weight: torch.Tensor,
                                                 es_cands,
                                                 sweep_scales,
                                                 ch_batch: int):
    return per_channel_mix_int4_positN_sqnr_batched(
        layer_weight=layer_weight,
        posit_nsize=4,
        posit_label="posit4",
        es_cands=es_cands,
        sweep_scales=sweep_scales,
        ch_batch=ch_batch,
    )

# -------------------- Chunked non-overlapping Perplexity --------------------
@torch.no_grad()
def eval_wikitext_ppl(model, tokenizer, seqlen: int, forward_dtype: str):
    model.eval()

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(test["text"]), return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids  # CPU

    nsamples = ids.numel() // seqlen
    if nsamples == 0:
        raise ValueError(f"Not enough tokens for seqlen={seqlen}")

    try:
        dev = next(p.device for p in model.parameters() if p.device.type != "meta")
    except StopIteration:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp = (forward_dtype in ["fp16", "bf16"]) and (dev.type == "cuda")
    amp_dtype = torch.float16 if forward_dtype == "fp16" else torch.bfloat16

    if dev.type == "cuda":
        autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype)
    else:
        class _NoOp:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        autocast_ctx = _NoOp()

    nll_sum = 0.0
    for i in tqdm(range(nsamples), desc="PPL", unit="chunk"):
        batch = ids[:, i * seqlen:(i + 1) * seqlen].to(dev)
        with autocast_ctx:
            logits = model(batch).logits  # (B, T, V)

        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1))
        nll_sum += (loss.item() * seqlen)

        if dev.type == "cuda":
            del batch, logits, shift_logits, shift_labels, loss
            torch.cuda.empty_cache()

    ppl = math.exp(nll_sum / (nsamples * seqlen))
    return float(ppl)


def build_calib_batches_wt2_train(tokenizer, n_seqs: int, seqlen: int, seed: int = 0):
    """Non-overlapping WikiText-2 train windows."""
    random.seed(seed)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    windows = []
    for i in range(0, len(ids) - seqlen + 1, seqlen):
        windows.append(ids[i : i + seqlen].clone().unsqueeze(0))
    if not windows:
        raise ValueError("Hessian calibration: train split shorter than --hessian_calib_seqlen")
    if len(windows) > n_seqs:
        windows = random.sample(windows, n_seqs)
    return windows


def make_fwd_autocast(device: torch.device, forward_dtype: str):
    use_amp = (forward_dtype in ["fp16", "bf16"]) and (device.type == "cuda")
    amp_dtype = torch.float16 if forward_dtype == "fp16" else torch.bfloat16
    if device.type == "cuda":
        return lambda: torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype)

    class _NoOp:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    return lambda: _NoOp()


# -------------------- Main --------------------
def main():
    args = get_args()
    device = torch.device(args.device)
    torch_dtype = get_torch_dtype(args.dtype)

    preset = MODEL_PRESETS[args.model]
    hf_id = preset["hf_id"]
    trust_remote = preset.get("trust_remote_code", False)
    use_fast = preset.get("use_fast_tokenizer", True)
    requires_auth = preset.get("requires_auth", False)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", None)
    if requires_auth and not hf_token:
        raise RuntimeError(
            f"Model '{args.model}' is gated and requires an HF token.\n"
            f"Set env HF_TOKEN or pass --hf_token <token>."
        )

    print(f"[Load] {args.model} -> {hf_id}  (dtype={args.dtype}, device={device})")
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote,
        token=hf_token
    ).to(device)

    tok = AutoTokenizer.from_pretrained(
        hf_id,
        use_fast=use_fast,
        trust_remote_code=trust_remote,
        token=hf_token
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    # Auto-cap ppl_seqlen by model context if available
    max_ctx = getattr(model.config, "max_position_embeddings", None)
    if isinstance(max_ctx, int) and max_ctx > 0 and args.ppl_seqlen > max_ctx:
        print(f"[Info] ppl_seqlen {args.ppl_seqlen} > max_position_embeddings {max_ctx}; capping to {max_ctx}")
        args.ppl_seqlen = max_ctx

    # Candidate scales (powers of two)
    sweep_scales = [2.0 ** k for k in range(args.log2_min, args.log2_max + 1)]
    act_hook = make_act_hook(args.use_act_quant, args.act_exp, args.act_man)

    # Collect target layers
    target_layers = []
    for name, mod in model.named_modules():
        if should_skip_layer(name, mod, skip_lm_head=args.skip_lm_head, quantize_embeddings=args.quantize_embeddings):
            continue
        if is_quant_linear(mod) and hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
            if mod.weight.dim() == 2:
                target_layers.append((name, mod))

    # nsize is implied by weight_format in the baseline (weight-only) path.
    if args.quant_mode == "baseline":
        target_nsize_by_format = {
            "posit4": 4, "posit5": 5,
            "bitmod4": 4, "bitmod4_g128": 4,
            "bitmod3": 3, "bitmod3_g128": 3,
            "mix_bitmod4_posit4": 4, "mix_bitmod4_posit5": 5,
            "mxfp4": 4, "mxfp4_g128": 4,
            "olive4": 4,
        }
        target_nsize = target_nsize_by_format.get(args.weight_format, args.nsize)
        if args.nsize != target_nsize:
            print(f"[Warn] weight_format={args.weight_format} expects nsize={target_nsize}; overriding nsize={args.nsize} -> {target_nsize}")
            args.nsize = target_nsize

    print(f"\n[Quantize] mode={args.quant_mode} | weight_format={args.weight_format} | "
          f"nsize={args.nsize} | es∈{args.es_candidates}, log2∈[{args.log2_min},{args.log2_max}], ch_batch={args.ch_batch}")
    print(f"Layers to quantize: {len(target_layers)}")

    # GPTQ-Hessian calibration (only for the gptq_hessian posit path)
    hessian_maps = {}
    hook_handles = []
    if args.quant_mode == "gptq_hessian":
        # Lazy import — file may not be present in the AE artifact.
        from posit_gptq_hessian import (
            assign_weight_from_gptq_matrix,
            gptq_posit_quantize_weight,
            linear_weight_to_gptq_matrix,
            register_hessian_hooks,
            remove_hooks,
            run_calibration_forward,
        )
        calib_batches = build_calib_batches_wt2_train(
            tok, args.hessian_calib_nsamples, args.hessian_calib_seqlen, seed=args.hessian_seed
        )
        print(
            f"[Hessian] WikiText-2 train: {len(calib_batches)} chunks × "
            f"seqlen={args.hessian_calib_seqlen} (seed={args.hessian_seed})"
        )
        hessian_maps, _ns, hook_handles = register_hessian_hooks(model, target_layers, device)
        try:
            run_calibration_forward(model, calib_batches, device, make_fwd_autocast(device, args.dtype))
        finally:
            remove_hooks(hook_handles)
            hook_handles = []
        print(f"[Hessian] Collected GPTQ-style H for {len(hessian_maps)} layers")

    quant_log = {"quant_mode": args.quant_mode}
    global_usage_counts = {}
    global_total_channels = 0
    with torch.no_grad():
        for name, mod in tqdm(target_layers, desc="Quantizing layers"):
            # ---- GPTQ-Hessian posit path ----
            if args.quant_mode == "gptq_hessian":
                H = hessian_maps.get(name)
                if H is None:
                    raise RuntimeError(f"Missing Hessian for layer {name}")
                scale_vec, es_vec, per_ch = per_channel_scales_es_batched(
                    mod.weight, nsize=args.nsize, es_cands=args.es_candidates,
                    sweep_scales=sweep_scales, ch_batch=args.ch_batch
                )
                Wm, transpose_back = linear_weight_to_gptq_matrix(mod.weight.data, mod)
                shared = isinstance(mod, modeling_utils.Conv1D)
                Q = gptq_posit_quantize_weight(
                    Wm, H, nsize=args.nsize,
                    scale_vec=scale_vec, es_vec=es_vec,
                    percdamp=args.hessian_percdamp,
                    blocksize=args.gptq_blocksize,
                    shared_scale_per_column=shared,
                    out_dtype=mod.weight.dtype,
                )
                assign_weight_from_gptq_matrix(mod, Q, transpose_back, mod.weight.dtype)
                if args.use_act_quant and not getattr(mod, "_act_quant_hooked", False):
                    mod.register_forward_pre_hook(act_hook)
                    setattr(mod, "_act_quant_hooked", True)
                quant_log[name] = {
                    "mode": "gptq_hessian_posit",
                    "shape": list(mod.weight.shape),
                    "nsize": int(args.nsize),
                    "hessian_shape": list(H.shape),
                    "shared_scale_per_column": shared,
                    "channels": per_ch,
                }
                continue

            # ---- baseline weight-only path: weight_format dispatch ----
            if args.weight_format == "posit4":
                q_w, per_ch = per_channel_quantize_fixed_nsize_batched(
                    mod.weight, nsize=4, es_cands=args.es_candidates,
                    sweep_scales=sweep_scales, ch_batch=args.ch_batch
                )
            elif args.weight_format == "posit5":
                q_w, per_ch = per_channel_quantize_fixed_nsize_batched(
                    mod.weight, nsize=5, es_cands=args.es_candidates,
                    sweep_scales=sweep_scales, ch_batch=args.ch_batch
                )
            elif args.weight_format == "bitmod4":
                q_w, per_ch = per_channel_int4_grouped_sqnr_batched(
                    mod.weight, sweep_scales=sweep_scales, ch_batch=args.ch_batch, search_metric=args.search_metric
                )
            elif args.weight_format == "bitmod4_g128":
                q_w, per_ch = per_channel_gw_int4_sqnr_batched(
                    mod.weight, sweep_scales=sweep_scales, ch_batch=args.ch_batch,
                    group_size=args.bitmod_group_size, search_metric=args.search_metric
                )
            elif args.weight_format == "bitmod3":
                q_w, per_ch = per_channel_int3_grouped_sqnr_batched(
                    mod.weight, sweep_scales=sweep_scales, ch_batch=args.ch_batch, search_metric=args.search_metric
                )
            elif args.weight_format == "bitmod3_g128":
                q_w, per_ch = per_channel_gw_int3_sqnr_batched(
                    mod.weight, sweep_scales=sweep_scales, ch_batch=args.ch_batch,
                    group_size=args.bitmod_group_size, search_metric=args.search_metric
                )
            elif args.weight_format == "mxfp4":
                q_w, per_ch = per_channel_groupwise_mxfp4_e2m1_sqnr_batched(
                    mod.weight, sweep_scales=sweep_scales, ch_batch=args.ch_batch,
                    group_size=32, search_metric=args.search_metric, format_label="mxfp4_g32",
                )
            elif args.weight_format == "mxfp4_g128":
                q_w, per_ch = per_channel_groupwise_mxfp4_e2m1_sqnr_batched(
                    mod.weight, sweep_scales=sweep_scales, ch_batch=args.ch_batch,
                    group_size=args.bitmod_group_size, search_metric=args.search_metric,
                    format_label="mxfp4_g128",
                )
            elif args.weight_format == "olive4":
                q_w, per_ch = per_channel_olive4_batched(
                    mod.weight, ch_batch=args.ch_batch,
                    w_low=args.olive_w_low, w_up=args.olive_w_up,
                )
            elif args.weight_format == "mix_bitmod4_posit4":
                q_w, per_ch = per_channel_mix_int4_positN_sqnr_batched(
                    mod.weight, posit_nsize=4, posit_label="posit4",
                    es_cands=args.es_candidates,
                    sweep_scales=sweep_scales, ch_batch=args.ch_batch
                )
            else:
                q_w, per_ch = per_channel_mix_int4_positN_sqnr_batched(
                    mod.weight, posit_nsize=5, posit_label="posit5",
                    es_cands=args.es_candidates,
                    sweep_scales=sweep_scales, ch_batch=args.ch_batch
                )
            mod.weight.data = q_w

            if args.use_act_quant and not getattr(mod, "_act_quant_hooked", False):
                mod.register_forward_pre_hook(act_hook)
                setattr(mod, "_act_quant_hooked", True)

            layer_usage = summarize_format_usage(per_ch)
            for k, v in layer_usage["counts"].items():
                global_usage_counts[k] = global_usage_counts.get(k, 0) + int(v)
            global_total_channels += int(layer_usage["total_channels"])
            quant_log[name] = {
                "mode": f"weight_only_{args.weight_format}",
                "shape": list(q_w.shape),
                "nsize": int(args.nsize),
                "format_usage": layer_usage,
                "channels": per_ch,
            }

    if global_total_channels > 0:
        global_usage_ratios = {
            k: (v / global_total_channels if global_total_channels > 0 else 0.0)
            for k, v in global_usage_counts.items()
        }
        quant_log["_summary"] = {
            "weight_format": args.weight_format,
            "format_usage": {
                "total_channels": global_total_channels,
                "counts": global_usage_counts,
                "ratios": global_usage_ratios,
            },
        }
    # Save model + tokenizer + log
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir, safe_serialization=True)
    tok.save_pretrained(args.save_dir)
    with open(os.path.join(args.save_dir, args.save_log_name), "w") as f:
        json.dump(quant_log, f, indent=2)

    print(f"[Saved] model+tokenizer -> {args.save_dir}")
    print(f"[Saved] log -> {os.path.join(args.save_dir, args.save_log_name)}")
    format_usage_summary = quant_log.get("_summary", {}).get(
        "format_usage", {"total_channels": 0, "counts": {}, "ratios": {}}
    )
    print(f"[Format Usage] total={format_usage_summary['total_channels']} "
          f"counts={format_usage_summary['counts']} ratios={format_usage_summary['ratios']}")

    # Chunked non-overlapping PPL
    ppl = eval_wikitext_ppl(model, tok, seqlen=args.ppl_seqlen, forward_dtype=args.dtype)
    print(f"\nPerplexity ({args.model}, {args.weight_format}, nsize={args.nsize}) [seqlen={args.ppl_seqlen}, fwd={args.dtype}]: {ppl:.4f}")

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump({
            "ppl": ppl,
            "model": args.model,
            "nsize": int(args.nsize),
            "weight_format": args.weight_format,
            "format_usage": format_usage_summary,
            "ppl_seqlen": args.ppl_seqlen,
            "dtype_forward": args.dtype,
            "quant_mode": args.quant_mode,
            "hessian_calib_nsamples": args.hessian_calib_nsamples if args.quant_mode == "gptq_hessian" else None,
            "hessian_calib_seqlen": args.hessian_calib_seqlen if args.quant_mode == "gptq_hessian" else None,
        }, f, indent=2)

    print(f"[Saved] metrics -> {os.path.join(args.save_dir, 'metrics.json')}")

if __name__ == "__main__":
    main()
