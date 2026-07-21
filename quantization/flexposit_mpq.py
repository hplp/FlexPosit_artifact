# #!/usr/bin/env python3
# # apply_mixed_posit_from_sensitivity.py
# # Mixed-precision posit quantization guided by GPU-accelerated per-window sensitivity CSV.
# # Works with CSV columns from sensitivity_from_saved_accel.py:
# #   layer, win_start, win_end, ppl_new, delta_ppl, seconds,
# #   override_nsize, channel_window, eval_batch_size, l2_diff_vs_baseline
# #
# # Modes:
# #   (A) Default: upgrade all windows with ΔPPL < 0
# #   (B) Budget:  select top windows by ΔPPL to meet --target_avg_bits
# #   (C) PPL goal: iteratively apply best windows until PPL <= --ppl_goal
# #   (S) Sweep (budget): sweep avg bits from base->upgrade in steps; eval PPL at each step
# #
# # Examples:
# # Default (upgrade all negative-Δ windows):
# #   python3 apply_mixed_posit_from_sensitivity.py \
# #     --base_dir quant_out_gpt2_posit4 \
# #     --fp32_reference_dir gpt2-large \
# #     --sensitivity_csv quant_out_gpt2_posit4/sensitivity_n5_cw8/sensitivity.csv \
# #     --out_dir quant_out_gpt2_mixposit_default
# #
# # Budget:
# #   python3 apply_mixed_posit_from_sensitivity.py \
# #     --base_dir quant_out_gpt2_posit4 \
# #     --fp32_reference_dir gpt2-large \
# #     --sensitivity_csv quant_out_gpt2_posit4/sensitivity_n5_cw8/sensitivity.csv \
# #     --out_dir quant_out_gpt2_mixposit_budget_4p50 \
# #     --target_avg_bits 4.50 --base_bits 4 --upgrade_bits 5
# #
# # PPL goal:
# #   python3 apply_mixed_posit_from_sensitivity.py \
# #     --base_dir quant_out_gpt2_posit4 \
# #     --fp32_reference_dir gpt2-large \
# #     --sensitivity_csv quant_out_gpt2_posit4/sensitivity_n5_cw8/sensitivity.csv \
# #     --out_dir quant_out_gpt2_mixposit_ppl23 \
# #     --ppl_goal 23.0 --seqlen 1024
# #
# # Sweep (budget):
# #   python3 apply_mixed_posit_from_sensitivity.py \
# #     --base_dir quant_out_gpt2_posit4 \
# #     --fp32_reference_dir gpt2-large \
# #     --sensitivity_csv quant_out_gpt2_posit4/sensitivity_n5_cw8/sensitivity.csv \
# #     --out_dir quant_out_gpt2_mixposit_sweep \
# #     --base_bits 4 --upgrade_bits 5 \
# #     --sweep_bits_start 4.0 --sweep_bits_end 5.0 --sweep_bits_step 0.1

# import argparse, os, csv, json, math, time
# from typing import List, Tuple, Dict, Any
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from datasets import load_dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import transformers
# import transformers.modeling_utils as modeling_utils
# import matplotlib
# matplotlib.use("Agg")  # headless backend for servers
# import matplotlib.pyplot as plt

# # Your posit quantize kernel
# from qtorch_plus.quant import posit_quantize

# transformers.logging.set_verbosity_error()
# EPS = 1e-8


# # ------------------- Args -------------------
# def get_args():
#     p = argparse.ArgumentParser()
#     # I/O
#     p.add_argument("--base_dir", required=True,
#                    help="Path to base model dir (already-quantized model saved via save_pretrained)")
#     p.add_argument("--fp32_reference_dir", required=True,
#                    help="HF id or local dir for FP32 reference weights")
#     p.add_argument("--sensitivity_csv", required=True,
#                    help="CSV produced by sensitivity_from_saved_accel.py")
#     p.add_argument("--out_dir", required=True,
#                    help="Output dir for the mixed-precision model")

#     # quantization knobs
#     p.add_argument("--override_nsize", type=int, default=5,
#                    help="posit nsize for the upgraded windows")
#     p.add_argument("--es_candidates", type=int, nargs="+", default=[0,1,2],
#                    help="posit es candidates")
#     p.add_argument("--log2_min", type=int, default=-8,
#                    help="min log2(scale) included in sweep")
#     p.add_argument("--log2_max", type=int, default=9,
#                    help="max log2(scale) included in sweep")

#     # compute/runtime
#     p.add_argument("--dtype", choices=["fp16","fp32"], default="fp16",
#                    help="forward precision for running the model")
#     p.add_argument("--device_map", choices=["none","auto"], default="none",
#                    help="device map for loading base model (auto for big models)")
#     p.add_argument("--skip_lm_head", action="store_true", default=True)
#     p.add_argument("--quantize_embeddings", action="store_true", default=False)

#     # Mode A: budget (target average bits)
#     p.add_argument("--target_avg_bits", type=float, default=None,
#                    help="Target average bits across ALL windows (None disables budget mode)")
#     p.add_argument("--base_bits", type=float, default=4.0,
#                    help="Assumed bits for non-upgraded windows")
#     p.add_argument("--upgrade_bits", type=float, default=5.0,
#                    help="Assumed bits for upgraded windows")
#     p.add_argument("--allow_positive_to_meet_budget", action="store_true", default=True,
#                    help="Allow ΔPPL >= 0 windows to be picked if needed to meet budget")

#     # Mode B: ppl goal
#     p.add_argument("--ppl_goal", type=float, default=None,
#                    help="If set, iteratively apply best windows until PPL <= goal")
#     p.add_argument("--seqlen", type=int, default=1024,
#                    help="Chunk length for WikiText-2 eval")
#     p.add_argument("--eval_dtype", choices=["fp16","fp32"], default="fp16",
#                    help="forward precision for eval PPL; loss in fp32")

#     # eval-at-end (optional for default/budget)
#     p.add_argument("--eval_final_ppl", action="store_true", default=False,
#                    help="Compute final PPL after upgrades (WikiText-2)")

#     # NEW: Mode S: sweep avg bits
#     p.add_argument("--sweep_bits_start", type=float, default=None,
#                    help="Start avg bits for sweep (default: base_bits)")
#     p.add_argument("--sweep_bits_end", type=float, default=None,
#                    help="End avg bits for sweep (default: upgrade_bits)")
#     p.add_argument("--sweep_bits_step", type=float, default=0.1,
#                    help="Step for sweep (default 0.1)")

#     # utility
#     p.add_argument("--dry_run", action="store_true", default=False,
#                    help="Plan/print only; do not modify or save model")

#     return p.parse_args()


# def get_torch_dtype(tag: str):
#     return torch.float16 if tag == "fp16" else torch.float32


# # ------------------- Model helpers -------------------


# def is_quant_linear(mod):
#     if isinstance(mod, nn.Linear):
#         return True
#     if isinstance(mod, modeling_utils.Conv1D):
#         return True
#     return False


# def should_skip_layer(name: str, mod: nn.Module, skip_lm_head: bool, quantize_embeddings: bool):
#     if skip_lm_head and (name == "lm_head" or name.endswith(".lm_head")):
#         return True
#     if isinstance(mod, nn.Embedding) and not quantize_embeddings:
#         return True
#     return False


# # ------------------- Read sensitivity CSV (per-window) -------------------
# def read_sensitivity_windows(csv_path: str) -> List[Tuple[str, int, int, float, int]]:
#     """
#     Return rows as: (layer, win_start, win_end, delta_ppl, channel_window)
#     Compatible with the GPU-accelerated sensitivity CSV.
#     """
#     rows = []
#     with open(csv_path, "r") as f:
#         r = csv.DictReader(f)
#         for row in r:
#             layer = (row.get("layer") or "").strip()
#             if not layer:
#                 continue
#             try:
#                 ws = int(row.get("win_start"))
#                 we = int(row.get("win_end"))
#                 dp = float(row.get("delta_ppl"))
#                 cw = int(row.get("channel_window", 1))
#             except Exception:
#                 continue
#             if not math.isnan(dp):
#                 rows.append((layer, ws, we, dp, max(1, cw)))
#     return rows


# # ------------------- Quant helpers (CPU search) -------------------
# @torch.no_grad()
# def _best_scale_es_for_vec(w_vec: torch.Tensor, nsize: int, es_cands, sweep_scales):
#     """Return (scale, es) maximizing SQNR for one channel vector."""
#     sp = torch.sum(w_vec ** 2) + EPS
#     best_sqnr = -float("inf")
#     best = (1.0, 0)

#     max_es = max(0, nsize - 1)
#     es_list = [e for e in es_cands if e <= max_es] or [0]

#     for sc in sweep_scales:
#         ws = w_vec * sc
#         for es in es_list:
#             q = posit_quantize(ws, nsize=nsize, es=int(es), scale=1.0) / sc
#             noise = torch.sum((w_vec - q) ** 2) + EPS
#             sqnr = 10.0 * torch.log10(sp / noise)
#             if float(sqnr) > best_sqnr:
#                 best_sqnr = float(sqnr)
#                 best = (float(sc), int(es))
#     return best


# @torch.no_grad()
# def quantize_window_cpu_from_fp32(fp32_w: torch.Tensor,
#                                   nsize: int,
#                                   start: int,
#                                   end: int,
#                                   es_cands,
#                                   sweep_scales) -> torch.Tensor:
#     """
#     Quantize ONLY rows [start:end) FROM FP32 reference and return a tensor
#     with SAME SHAPE as fp32_w where [start:end) are quantized, rest unchanged FP32.
#     """
#     W = fp32_w.detach().float().cpu()
#     if W.ndim != 2:
#         raise RuntimeError(f"Expect 2D weight, got {tuple(W.shape)}")
#     Cout, K = W.shape
#     start = max(0, min(start, Cout))
#     end = max(start, min(end, Cout))
#     if start >= end:
#         return W

#     flat = W.view(Cout, -1)
#     block = flat[start:end]  # [Cw, K]
#     q_block = torch.empty_like(block)
#     for i in range(block.size(0)):
#         w_vec = block[i]
#         sc, es = _best_scale_es_for_vec(w_vec, nsize=nsize, es_cands=es_cands, sweep_scales=sweep_scales)
#         q_block[i] = posit_quantize(w_vec * sc, nsize=nsize, es=es, scale=1.0) / sc

#     out = flat.clone()
#     out[start:end] = q_block
#     return out.view_as(W)


# # ------------------- Eval (chunked non-overlapping WT2) -------------------
# @torch.no_grad()
# def encode_wikitext2_cpu(tokenizer) -> torch.Tensor:
#     test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
#     enc = tokenizer("\n\n".join(test["text"]), return_tensors="pt", add_special_tokens=False)
#     return enc.input_ids  # CPU [1, T]


# @torch.no_grad()
# def eval_ppl_with_ids(model, ids_cpu: torch.Tensor, seqlen: int, use_fp16_fwd: bool) -> float:
#     model.eval()
#     nsamples = ids_cpu.numel() // seqlen
#     if nsamples == 0:
#         raise ValueError(f"Not enough tokens for seqlen={seqlen}")

#     try:
#         dev = next(p.device for p in model.parameters() if p.device.type != "meta")
#     except StopIteration:
#         dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     if dev.type == "cuda":
#         autocast_ctx = torch.cuda.amp.autocast(enabled=use_fp16_fwd, dtype=torch.float16)
#     else:
#         class _NoOp:
#             def __enter__(self): return None
#             def __exit__(self, *a): return False
#         autocast_ctx = _NoOp()

#     nll_sum = 0.0
#     for i in range(nsamples):
#         batch = ids_cpu[:, i*seqlen:(i+1)*seqlen].to(dev)
#         with autocast_ctx:
#             logits = model(batch).logits
#         shift_logits = logits[:, :-1, :].contiguous().float()
#         shift_labels = batch[:, 1:].contiguous()
#         loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
#                                shift_labels.view(-1))
#         nll_sum += (loss.item() * seqlen)
#         if dev.type == "cuda":
#             del batch, logits, shift_logits, shift_labels, loss
#             torch.cuda.empty_cache()

#     return float(math.exp(nll_sum / (nsamples * seqlen)))


# # ------------------- Utility -------------------
# def collect_quantizable_layers(model, skip_lm_head: bool, quantize_embeddings: bool) -> List[str]:
#     names = []
#     for name, mod in model.named_modules():
#         if should_skip_layer(name, mod, skip_lm_head, quantize_embeddings):
#             continue
#         if is_quant_linear(mod) and hasattr(mod, "weight") and mod.weight is not None and mod.weight.dim() == 2:
#             names.append(name)
#     return names


# def apply_windows_to_model(model,
#                            ref_sd: Dict[str, torch.Tensor],
#                            windows: List[Tuple[str, int, int, int]],
#                            nsize: int,
#                            es_cands,
#                            sweep_scales,
#                            skip_lm_head: bool,
#                            quantize_embeddings: bool) -> Dict[str, Any]:
#     """
#     Apply a set of windows (layer, start, end, cw) to the model in-place.
#     Returns a dict summarizing upgrades per layer.
#     """
#     # Group windows by layer to reduce device <-> host shuffles
#     by_layer: Dict[str, List[Tuple[int,int,int]]] = {}
#     for ly, ws, we, cw in windows:
#         by_layer.setdefault(ly, []).append((ws, we, cw))

#     upgraded = {}
#     for name, mod in model.named_modules():
#         if name not in by_layer:
#             continue
#         if should_skip_layer(name, mod, skip_lm_head, quantize_embeddings):
#             continue
#         if not (is_quant_linear(mod) and hasattr(mod, "weight") and mod.weight is not None and mod.weight.dim() == 2):
#             continue

#         key = name + ".weight"
#         if key not in ref_sd:
#             print(f"[Warn] Missing FP32 ref for {key}; skip layer {name}.")
#             continue

#         fp32_w = ref_sd[key]
#         # Start from CURRENT layer weights (not FP32), stitch window-by-window
#         base_w_cpu = mod.weight.detach().float().cpu()
#         Q_full = base_w_cpu.clone()

#         for (ws, we, _cw) in by_layer[name]:
#             # quantize rows ws:we from FP32 ref using nsize/es/scale chosen per channel
#             Q_win = quantize_window_cpu_from_fp32(fp32_w, nsize=nsize,
#                                                   start=ws, end=we,
#                                                   es_cands=es_cands,
#                                                   sweep_scales=sweep_scales)
#             Q_full[ws:we] = Q_win[ws:we]

#         with torch.no_grad():
#             mod.weight.data = Q_full.to(mod.weight.device, dtype=mod.weight.dtype)

#         upgraded[name] = {
#             "n_windows": len(by_layer[name]),
#             "windows": [(ws, we) for (ws, we, _cw) in by_layer[name]],
#             "nsize": nsize,
#             "shape": list(fp32_w.shape),
#         }

#     return upgraded


# # ------------------- Main -------------------
# def main():
#     args = get_args()
#     os.makedirs(args.out_dir, exist_ok=True)

#     q_dtype = get_torch_dtype(args.dtype)
#     eval_fp16 = (args.eval_dtype == "fp16")
#     device_map = None if args.device_map == "none" else "auto"
#     sweep_scales = [2.0 ** k for k in range(args.log2_min, args.log2_max + 1)]

#     # 1) Read per-window sensitivity and sort by ΔPPL ascending (most negative first)
#     sens_rows = read_sensitivity_windows(args.sensitivity_csv)  # (layer, ws, we, dp, cw)
#     if not sens_rows:
#         print("[Error] No valid sensitivity rows found.")
#         return
#     sens_sorted = sorted(sens_rows, key=lambda x: (x[3], x[0], x[1], x[2]))  # by delta_ppl, then layer/start/end
#     total_windows_all = len(sens_sorted)

#     # 2) Load tokenizer (for optional eval)
#     # tok = AutoTokenizer.from_pretrained(args.base_dir, use_fast=True)
#     tok = AutoTokenizer.from_pretrained(args.base_dir, use_fast=False, trust_remote_code=True)

#     if tok.pad_token is None:
#         tok.pad_token = tok.eos_token

#     # 3) Load FP32 reference (CPU)
#     print(f"[Load FP32 reference] {args.fp32_reference_dir}")
#     ref_model = AutoModelForCausalLM.from_pretrained(
#         args.fp32_reference_dir, torch_dtype=torch.float32, device_map=None, low_cpu_mem_usage=True, trust_remote_code=True
#     ).cpu()
#     ref_sd = {k: v.detach().cpu() for k, v in ref_model.state_dict().items()}
#     del ref_model

#     # 4) Probe model to filter only real quantizable modules
#     probe = AutoModelForCausalLM.from_pretrained(
#         args.base_dir, torch_dtype=q_dtype,
#         device_map=(device_map if args.dry_run else (None if args.device_map == "none" else "auto")),
#         low_cpu_mem_usage=True
#     )
#     valid_layers = set(collect_quantizable_layers(probe, args.skip_lm_head, args.quantize_embeddings))
#     del probe
#     sens_sorted = [row for row in sens_sorted if row[0] in valid_layers]
#     if not sens_sorted:
#         print("[Error] No sensitivity rows match quantizable modules in the base model.")
#         return

#     # Helper: unique windows in ΔPPL order, optionally filtering positives
#     def candidate_windows(rows, allow_positive):
#         seen = set()
#         out = []
#         for (ly, ws, we, dp, cw) in rows:
#             if (dp >= 0.0) and (not allow_positive):
#                 continue
#             key = (ly, ws, we)
#             if key in seen:
#                 continue
#             seen.add(key)
#             out.append((ly, ws, we, cw, dp))
#         return out

#     # ---------- NEW: Mode S: Sweep (budget) ----------
#     do_sweep = (args.sweep_bits_start is not None) or (args.sweep_bits_end is not None)
#     if (args.ppl_goal is None) and do_sweep:
#         start_bits = args.sweep_bits_start if args.sweep_bits_start is not None else args.base_bits
#         end_bits   = args.sweep_bits_end   if args.sweep_bits_end   is not None else args.upgrade_bits
#         if end_bits < start_bits:
#             start_bits, end_bits = end_bits, start_bits
#         steps = int(round((end_bits - start_bits) / max(1e-12, args.sweep_bits_step))) + 1
#         targets = [round(start_bits + i*args.sweep_bits_step, 6) for i in range(steps)]
#         targets[0]   = max(targets[0], args.base_bits)
#         targets[-1]  = min(targets[-1], args.upgrade_bits)
#         print(f"\n[PLAN: Sweep Mode]")
#         print(f"  sweep_bits: {targets[0]:.3f} → {targets[-1]:.3f}  step={args.sweep_bits_step}")

#         # Build candidate window list once (ΔPPL ascending; positives optional)
#         cand = candidate_windows(sens_sorted, allow_positive=args.allow_positive_to_meet_budget)
#         total_cand = len(cand)
#         if total_cand == 0:
#             print("[Error] No candidate windows available for sweep.")
#             return

#         def k_needed(avg_bits):
#             # k = ceil(N * (avg - base) / (upgrade - base))
#             frac = (avg_bits - args.base_bits) / max(1e-12, (args.upgrade_bits - args.base_bits))
#             return int(math.ceil(max(0.0, min(1.0, frac)) * total_cand))

#         if args.dry_run:
#             print(f"  total_candidate_windows={total_cand} (of {total_windows_all} total)")
#             print("  First 5 targets:", ", ".join(f"{t:.2f}" for t in targets[:5]))
#             print("[Dry-run] Exiting without modification.")
#             return

#         # Load base model to modify
#         model = AutoModelForCausalLM.from_pretrained(
#             args.base_dir, torch_dtype=q_dtype, device_map=device_map, low_cpu_mem_usage=True
#         )
#         if device_map is None and torch.cuda.is_available():
#             model = model.to("cuda")

#         # Prepare eval ids once
#         print("[Eval] Tokenizing WikiText-2 test...")
#         ids_cpu = encode_wikitext2_cpu(tok)
#         use_fp16 = (args.eval_dtype == "fp16")

#         # CSV + figure data
#         csv_path = os.path.join(args.out_dir, "ppl_vs_avg_bits.csv")
#         xs, ys = [], []

#         with open(csv_path, "w", newline="") as fcsv:
#             writer = csv.DictWriter(fcsv, fieldnames=[
#                 "step", "target_avg_bits", "achieved_avg_bits",
#                 "windows_applied", "channels_applied", "ppl"
#             ])
#             writer.writeheader()

#             applied_k = 0
#             applied_ch = 0

#             # baseline eval (0 upgrades in this session)
#             ppl = eval_ppl_with_ids(model, ids_cpu, seqlen=args.seqlen, use_fp16_fwd=use_fp16)
#             achieved_bits = args.base_bits + (applied_k / total_cand) * (args.upgrade_bits - args.base_bits)
#             writer.writerow({
#                 "step": 0,
#                 "target_avg_bits": round(targets[0], 6),
#                 "achieved_avg_bits": round(achieved_bits, 6),
#                 "windows_applied": applied_k,
#                 "channels_applied": applied_ch,
#                 "ppl": f"{ppl:.6f}"
#             })
#             print(f"[Sweep 0] k={applied_k}/{total_cand}  achieved={achieved_bits:.3f} bits  PPL={ppl:.4f}")
#             xs.append(achieved_bits); ys.append(ppl)

#             for i, tgt in enumerate(targets, start=1):
#                 k_tgt = min(total_cand, max(0, k_needed(tgt)))
#                 if k_tgt > applied_k:
#                     # apply next slice of windows (incremental, in-place)
#                     to_apply = cand[applied_k:k_tgt]
#                     _ = apply_windows_to_model(
#                         model, ref_sd,
#                         [(ly, ws, we, cw) for (ly, ws, we, cw, _dp) in to_apply],
#                         nsize=args.override_nsize,
#                         es_cands=args.es_candidates, sweep_scales=sweep_scales,
#                         skip_lm_head=args.skip_lm_head, quantize_embeddings=args.quantize_embeddings
#                     )
#                     applied_k = k_tgt
#                     applied_ch += sum(int(cw) for (_ly, _ws, _we, cw, _dp) in to_apply)

#                 achieved_bits = args.base_bits + (applied_k / total_cand) * (args.upgrade_bits - args.base_bits)
#                 ppl = eval_ppl_with_ids(model, ids_cpu, seqlen=args.seqlen, use_fp16_fwd=use_fp16)

#                 writer.writerow({
#                     "step": i,
#                     "target_avg_bits": round(tgt, 6),
#                     "achieved_avg_bits": round(achieved_bits, 6),
#                     "windows_applied": applied_k,
#                     "channels_applied": applied_ch,
#                     "ppl": f"{ppl:.6f}"
#                 })
#                 print(f"[Sweep {i}] target={tgt:.3f}  k={applied_k}/{total_cand}  "
#                       f"achieved={achieved_bits:.3f} bits  PPL={ppl:.4f}")
#                 xs.append(achieved_bits); ys.append(ppl)

#         # Save figure
#         try:
#             fig_path = os.path.join(args.out_dir, "ppl_vs_avg_bits.png")
#             plt.figure()
#             plt.plot(xs, ys, marker="o")
#             plt.xlabel("Achieved Avg Bits")
#             plt.ylabel("Perplexity (PPL)")
#             plt.title("PPL vs Achieved Avg Bits (Sweep)")
#             plt.grid(True)
#             plt.tight_layout()
#             plt.savefig(fig_path, dpi=150)
#             plt.close()
#             print(f"[Figure] Saved: {fig_path}")
#         except Exception as e:
#             print(f"[Warn] Failed to save plot: {e}")

#         # Save final model + summary
#         print(f"[Save] -> {args.out_dir}")
#         model.save_pretrained(args.out_dir, safe_serialization=True)
#         tok.save_pretrained(args.out_dir)
#         with open(os.path.join(args.out_dir, "mixposit_sweep_log.json"), "w") as f:
#             json.dump({
#                 "mode": "sweep",
#                 "base_dir": args.base_dir,
#                 "fp32_reference_dir": args.fp32_reference_dir,
#                 "sensitivity_csv": args.sensitivity_csv,
#                 "override_nsize": args.override_nsize,
#                 "es_candidates": args.es_candidates,
#                 "log2_min": args.log2_min,
#                 "log2_max": args.log2_max,
#                 "base_bits": args.base_bits,
#                 "upgrade_bits": args.upgrade_bits,
#                 "allow_positive_to_meet_budget": args.allow_positive_to_meet_budget,
#                 "total_candidate_windows": total_cand,
#                 "total_windows_all": total_windows_all,
#                 "ppl_csv": "ppl_vs_avg_bits.csv",
#                 "ppl_png": "ppl_vs_avg_bits.png"
#             }, f, indent=2)
#         print("[OK] Sweep complete.")
#         return

#     # ------- Mode C: Default (ΔPPL < 0 windows) -------
#     if args.target_avg_bits is None and args.ppl_goal is None:
#         # unchanged...
#         neg_rows = [(ly, ws, we, dp, cw) for (ly, ws, we, dp, cw) in sens_sorted if dp < 0.0]
#         # unique by (layer,start,end)
#         seen = set(); chosen_windows = []
#         for (ly, ws, we, dp, cw) in neg_rows:
#             key = (ly, ws, we)
#             if key in seen: continue
#             seen.add(key)
#             chosen_windows.append((ly, ws, we, cw))
#         windows_selected = len(chosen_windows)
#         total_windows = len({(ly, ws, we) for (ly, ws, we, _dp, _cw) in sens_sorted})
#         achieved_avg_bits = args.base_bits + (min(windows_selected, total_windows) / total_windows) * (args.upgrade_bits - args.base_bits)

#         print("\n[PLAN: Default Mode]")
#         print(f"  total_windows={total_windows}")
#         print(f"  windows_selected={windows_selected}")
#         print(f"  achieved_avg_bits={achieved_avg_bits:.3f}")

#         if args.dry_run:
#             print("[Dry-run] Exiting without modification.")
#             return

#         # Load base model to modify
#         model = AutoModelForCausalLM.from_pretrained(
#             args.base_dir, torch_dtype=q_dtype, device_map=device_map, low_cpu_mem_usage=True
#         )
#         if device_map is None and torch.cuda.is_available():
#             model = model.to("cuda")

#         upgraded = apply_windows_to_model(
#             model, ref_sd, chosen_windows, nsize=args.override_nsize,
#             es_cands=args.es_candidates, sweep_scales=sweep_scales,
#             skip_lm_head=args.skip_lm_head, quantize_embeddings=args.quantize_embeddings
#         )

#         final_ppl = None
#         if args.eval_final_ppl:
#             print("[Eval] Computing final PPL on WikiText-2...")
#             ids_cpu = encode_wikitext2_cpu(tok)
#             final_ppl = eval_ppl_with_ids(model, ids_cpu, seqlen=args.seqlen, use_fp16_fwd=eval_fp16)
#             print(f"[Final PPL] {final_ppl:.4f}")

#         print(f"[Save] -> {args.out_dir}")
#         model.save_pretrained(args.out_dir, safe_serialization=True)
#         tok.save_pretrained(args.out_dir)
#         with open(os.path.join(args.out_dir, "mixposit_default_log.json"), "w") as f:
#             json.dump({
#                 "mode": "default_negative_only",
#                 "base_bits": args.base_bits,
#                 "upgrade_bits": args.upgrade_bits,
#                 "total_windows": total_windows,
#                 "windows_selected": windows_selected,
#                 "achieved_avg_bits": achieved_avg_bits,
#                 "upgraded": upgraded,
#                 "final_ppl": final_ppl
#             }, f, indent=2)
#         print("[OK] Saved mixed-precision model and log.")
#         return

#     # ------- Mode A: Budget -------
#     if args.ppl_goal is None and args.target_avg_bits is not None:
#         if not (args.base_bits <= args.target_avg_bits <= args.upgrade_bits):
#             raise ValueError(f"--target_avg_bits must be within [{args.base_bits}, {args.upgrade_bits}]")

#         total_windows = len({(ly, ws, we) for (ly, ws, we, _dp, _cw) in sens_sorted})
#         need_windows = int(math.ceil(
#             total_windows * (args.target_avg_bits - args.base_bits) /
#             max(1e-12, (args.upgrade_bits - args.base_bits))
#         ))

#         picked = []
#         seen = set()
#         for (ly, ws, we, dp, cw) in sens_sorted:
#             if len(picked) >= need_windows:
#                 break
#             if (dp < 0.0) or args.allow_positive_to_meet_budget:
#                 key = (ly, ws, we)
#                 if key in seen: continue
#                 seen.add(key)
#                 picked.append((ly, ws, we, cw))

#         windows_selected = len(picked)
#         achieved_avg_bits = args.base_bits + (min(windows_selected, total_windows) / total_windows) * (args.upgrade_bits - args.base_bits)

#         print("\n[PLAN: Budget Mode]")
#         print(f"  total_windows={total_windows}, needed_windows={need_windows}")
#         print(f"  windows_selected={windows_selected}")
#         print(f"  achieved_avg_bits={achieved_avg_bits:.3f} (target={args.target_avg_bits:.3f})")

#         if args.dry_run:
#             print("[Dry-run] Exiting without modification.")
#             return

#         model = AutoModelForCausalLM.from_pretrained(
#             args.base_dir, torch_dtype=q_dtype, device_map=device_map, low_cpu_mem_usage=True
#         )
#         if device_map is None and torch.cuda.is_available():
#             model = model.to("cuda")

#         total_channels = sum(int(cw) for (_, _, _, cw) in picked) if picked else 0
#         done_windows = 0
#         done_channels = 0

#         upgraded_by_layer: Dict[str, Dict[str, Any]] = {}

#         for (ly, ws, we, cw) in picked:
#             _ = apply_windows_to_model(
#                 model, ref_sd, [(ly, ws, we, cw)], nsize=args.override_nsize,
#                 es_cands=args.es_candidates, sweep_scales=sweep_scales,
#                 skip_lm_head=args.skip_lm_head, quantize_embeddings=args.quantize_embeddings
#             )
#             done_windows += 1
#             done_channels += int(cw)

#             print(f"[Budget] progress: {done_windows}/{windows_selected} windows, "
#                   f"{done_channels}/{total_channels} channels")

#             if ly not in upgraded_by_layer:
#                 key = ly + ".weight"
#                 shape = list(ref_sd[key].shape) if key in ref_sd else []
#                 upgraded_by_layer[ly] = {"n_windows": 0, "windows": [], "nsize": args.override_nsize, "shape": shape}
#             upgraded_by_layer[ly]["n_windows"] += 1
#             upgraded_by_layer[ly]["windows"].append((ws, we))

#         # Always compute final PPL for Budget mode
#         print("[Eval] Computing final PPL on WikiText-2 (Budget mode)...")
#         ids_cpu = encode_wikitext2_cpu(tok)
#         final_ppl = eval_ppl_with_ids(model, ids_cpu, seqlen=args.seqlen, use_fp16_fwd=eval_fp16)
#         print(f"[Final PPL] {final_ppl:.4f}")

#         print(f"[Save] -> {args.out_dir}")
#         model.save_pretrained(args.out_dir, safe_serialization=True)
#         tok.save_pretrained(args.out_dir)
#         with open(os.path.join(args.out_dir, "mixposit_budget_log.json"), "w") as f:
#             json.dump({
#                 "mode": "budget",
#                 "target_avg_bits": args.target_avg_bits,
#                 "achieved_avg_bits": achieved_avg_bits,
#                 "base_bits": args.base_bits,
#                 "upgrade_bits": args.upgrade_bits,
#                 "total_windows": total_windows,
#                 "windows_selected": windows_selected,
#                 "channels_selected": total_channels,
#                 "final_windows_applied": done_windows,
#                 "final_channels_applied": done_channels,
#                 "upgraded": upgraded_by_layer,
#                 "final_ppl": final_ppl
#             }, f, indent=2)
#         print("[OK] Saved mixed-precision model and log.")
#         return

#     # ------- Mode B: PPL Goal -------
#     if args.ppl_goal is not None:
#         model = AutoModelForCausalLM.from_pretrained(
#             args.base_dir, torch_dtype=q_dtype, device_map=device_map, low_cpu_mem_usage=True
#         )
#         if device_map is None and torch.cuda.is_available():
#             model = model.to("cuda")

#         print("[Eval] Tokenizing WikiText-2 test...")
#         ids_cpu = encode_wikitext2_cpu(tok)
#         print("[Eval] Computing baseline PPL...]")
#         ppl = eval_ppl_with_ids(model, ids_cpu, seqlen=args.seqlen, use_fp16_fwd=eval_fp16)
#         print(f"[Baseline] PPL = {ppl:.4f}")

#         # Prepare CSV logging
#         curve_csv_path = os.path.join(args.out_dir, "ppl_vs_windows.csv")
#         os.makedirs(args.out_dir, exist_ok=True)
#         with open(curve_csv_path, "w", newline="") as fcsv:
#             writer = csv.DictWriter(
#                 fcsv,
#                 fieldnames=["step", "windows_upgraded", "channels_upgraded", "layer", "win_start", "win_end", "ppl"]
#             )
#             writer.writeheader()
#             writer.writerow({
#                 "step": 0,
#                 "windows_upgraded": 0,
#                 "channels_upgraded": 0,
#                 "layer": "",
#                 "win_start": "",
#                 "win_end": "",
#                 "ppl": f"{ppl:.6f}"
#             })

#         upgraded_windows: List[Tuple[str,int,int,int]] = []
#         start_t = time.time()
#         windows_upgraded = 0
#         channels_upgraded = 0
#         step = 0

#         # For plotting: PPL vs cumulative channels
#         x_list = [0]         # channels_upgraded (baseline)
#         y_list = [ppl]       # ppl (baseline)

#         # iterate windows by ΔPPL ascending; apply one, re-eval; stop when goal met
#         for (ly, ws, we, dp, cw) in sens_sorted:
#             if ppl <= args.ppl_goal:
#                 break
#             if (dp >= 0.0) and (not args.allow_positive_to_meet_budget):
#                 continue
#             # apply this single window
#             upgraded_windows.append((ly, ws, we, cw))
#             _ = apply_windows_to_model(
#                 model, ref_sd, [(ly, ws, we, cw)], nsize=args.override_nsize,
#                 es_cands=args.es_candidates, sweep_scales=sweep_scales,
#                 skip_lm_head=args.skip_lm_head, quantize_embeddings=args.quantize_embeddings
#             )
#             ppl = eval_ppl_with_ids(model, ids_cpu, seqlen=args.seqlen, use_fp16_fwd=eval_fp16)
#             print(f"  [{ly} {ws}:{we}) -> PPL {ppl:.4f}")

#             # Update counters and logs
#             windows_upgraded += 1
#             channels_upgraded += int(cw)
#             step += 1

#             with open(curve_csv_path, "a", newline="") as fcsv:
#                 writer = csv.DictWriter(
#                     fcsv,
#                     fieldnames=["step", "windows_upgraded", "channels_upgraded", "layer", "win_start", "win_end", "ppl"]
#                 )
#                 writer.writerow({
#                     "step": step,
#                     "windows_upgraded": windows_upgraded,
#                     "channels_upgraded": channels_upgraded,
#                     "layer": ly,
#                     "win_start": ws,
#                     "win_end": we,
#                     "ppl": f"{ppl:.6f}"
#                 })

#             # Append to plot lists
#             x_list.append(channels_upgraded)
#             y_list.append(ppl)

#         elapsed = time.time() - start_t
#         total_windows = len({(ly, ws, we) for (ly, ws, we, _dp, _cw) in sens_sorted})
#         achieved_avg_bits = args.base_bits + (min(windows_upgraded, total_windows) / total_windows) * (args.upgrade_bits - args.base_bits)

#         # Save plot: PPL vs number of channels upgraded
#         plot_path = os.path.join(args.out_dir, "ppl_vs_channels.png")
#         try:
#             plt.figure()
#             plt.plot(x_list, y_list, marker="o")
#             plt.xlabel("# Channels Upgraded")
#             plt.ylabel("Perplexity (PPL)")
#             plt.title("PPL vs Channels Upgraded (PPL-goal mode)")
#             plt.grid(True)
#             plt.tight_layout()
#             plt.savefig(plot_path, dpi=150)
#             plt.close()
#             print(f"[Curve] Plot saved to: {plot_path}")
#         except Exception as e:
#             print(f"[Warn] Failed to save plot: {e}")

#         print("\n[RESULT: PPL Goal Mode]")
#         print(f"  ppl_goal={args.ppl_goal:.4f}, final_ppl={ppl:.4f}, time={elapsed/60:.1f} min")
#         print(f"  windows_upgraded={windows_upgraded}/{total_windows}")
#         print(f"  achieved_avg_bits={achieved_avg_bits:.3f}  (base={args.base_bits}, upgrade={args.upgrade_bits})")
#         print(f"  [Curve] CSV saved to: {curve_csv_path}")

#         if args.dry_run:
#             print("[Dry-run] Exiting without saving.")
#             return

#         print(f"[Save] -> {args.out_dir}")
#         model.save_pretrained(args.out_dir, safe_serialization=True)
#         tok.save_pretrained(args.out_dir)
#         with open(os.path.join(args.out_dir, "mixposit_pplgoal_log.json"), "w") as f:
#             json.dump({
#                 "mode": "ppl_goal",
#                 "ppl_goal": args.ppl_goal,
#                 "final_ppl": ppl,
#                 "elapsed_sec": elapsed,
#                 "base_bits": args.base_bits,
#                 "upgrade_bits": args.upgrade_bits,
#                 "total_windows": total_windows,
#                 "windows_upgraded": windows_upgraded,
#                 "channels_upgraded": channels_upgraded,
#                 "upgraded_windows": [{"layer": ly, "start": ws, "end": we, "channel_window": cw}
#                                      for (ly, ws, we, cw) in upgraded_windows]
#             }, f, indent=2)
#         print("[OK] Saved model and PPL-goal log.")
#         return

#     print("[Error] Invalid mode selection.")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# apply_mixed_posit_from_sensitivity.py
# Mixed-precision posit quantization with multiple sweep strategies:
#   - sensitivity: pick windows by ΔPPL ascending (old behavior)
#   - random:      pick windows in a randomized order (seeded)
#   - location:    pick windows in model/layer order, earliest → latest
#
# Modes:
#   Default:  upgrade all windows with ΔPPL < 0
#   Budget:   select top windows by ΔPPL to meet --target_avg_bits
#   PPL goal: iteratively apply best windows until PPL <= --ppl_goal
#   Sweep:    sweep avg bits (e.g., 4→5 step 0.1) with selectable strategy
#
# Examples (Sweep):
#   python3 apply_mixed_posit_from_sensitivity.py \
#     --base_dir quant_out_gpt2_posit4 \
#     --fp32_reference_dir gpt2-large \
#     --sensitivity_csv quant_out_gpt2_posit4/sensitivity_n5_cw8/sensitivity.csv \
#     --out_dir quant_out_gpt2_mixposit_sweep \
#     --base_bits 4 --upgrade_bits 5 \
#     --sweep_bits_start 4.0 --sweep_bits_end 5.0 --sweep_bits_step 0.1 \
#     --sweep_strategy random --random_seed 1234

import argparse, os, csv, json, math, time
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import transformers.modeling_utils as modeling_utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qtorch_plus.quant import posit_quantize

transformers.logging.set_verbosity_error()
EPS = 1e-8

# ------------------- Args -------------------
def get_args():
    p = argparse.ArgumentParser()
    # I/O
    p.add_argument("--base_dir", required=True)
    p.add_argument("--fp32_reference_dir", required=True)
    p.add_argument("--sensitivity_csv", required=True)
    p.add_argument("--out_dir", required=True)

    # quantization knobs
    p.add_argument("--override_nsize", type=int, default=5)
    p.add_argument("--es_candidates", type=int, nargs="+", default=[0,1,2])
    p.add_argument("--log2_min", type=int, default=-8)
    p.add_argument("--log2_max", type=int, default=9)

    # compute/runtime
    p.add_argument("--dtype", choices=["fp16","bf16","fp32"], default="fp16")
    p.add_argument("--device_map", choices=["none","auto"], default="none")
    p.add_argument("--skip_lm_head", action="store_true", default=True)
    p.add_argument("--quantize_embeddings", action="store_true", default=False)

    # Mode A: budget (target average bits)
    p.add_argument("--target_avg_bits", type=float, default=None)
    p.add_argument("--base_bits", type=float, default=4.0)
    p.add_argument("--upgrade_bits", type=float, default=5.0)
    p.add_argument("--allow_positive_to_meet_budget", action="store_true", default=True)
    p.add_argument("--downgrade", action="store_true", default=False,
                   help="Downgrade least-sensitive windows from upgrade_bits to base_bits, instead of "
                        "upgrading most-sensitive windows from base_bits to upgrade_bits. "
                        "--base_dir must be an upgrade_bits (e.g. Posit(5,1)) checkpoint. "
                        "Iterates fewer windows when target > (base+upgrade)/2 (e.g. target 4.9 for [4,5]).")
    p.add_argument("--downgrade_nsize", type=int, default=None,
                   help="nsize to apply when downgrading windows. Defaults to int(base_bits).")

    # Mode B: ppl goal
    p.add_argument("--ppl_goal", type=float, default=None)
    p.add_argument("--seqlen", type=int, default=1024)
    p.add_argument("--eval_dtype", choices=["fp16","fp32"], default="fp16")

    # eval-at-end (optional for default/budget)
    p.add_argument("--eval_final_ppl", action="store_true", default=False)

    # Mode S: sweep avg bits
    p.add_argument("--sweep_bits_start", type=float, default=None)
    p.add_argument("--sweep_bits_end", type=float, default=None)
    p.add_argument("--sweep_bits_step", type=float, default=0.1)

    # NEW: sweep strategy & randomness
    p.add_argument("--sweep_strategy",
                   choices=["sensitivity", "random", "location"],
                   default="sensitivity",
                   help="Window ordering for sweep mode.")
    p.add_argument("--random_seed", type=int, default=20250925,
                   help="Seed for --sweep_strategy random (and any RNG use).")

    # utility
    p.add_argument("--dry_run", action="store_true", default=False)

    return p.parse_args()

def get_torch_dtype(tag: str):
    return torch.float16 if tag == "fp16" else torch.float32

# ------------------- Model helpers -------------------
def is_quant_linear(mod):
    if isinstance(mod, nn.Linear):
        return True
    if isinstance(mod, modeling_utils.Conv1D):  # GPT-2 attention/MLP use Conv1D, not Linear
        return True
    return False

def should_skip_layer(name: str, mod: nn.Module, skip_lm_head: bool, quantize_embeddings: bool):
    if skip_lm_head and (name == "lm_head" or name.endswith(".lm_head")):
        return True
    if isinstance(mod, nn.Embedding) and not quantize_embeddings:
        return True
    return False

# ------------------- Read sensitivity CSV (per-window) -------------------
def read_sensitivity_windows(csv_path: str) -> List[Tuple[str, int, int, float, int]]:
    """
    Return rows as: (layer, win_start, win_end, delta_ppl, channel_window)
    """
    rows = []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            layer = (row.get("layer") or "").strip()
            if not layer:
                continue
            try:
                ws = int(row.get("win_start"))
                we = int(row.get("win_end"))
                dp = float(row.get("delta_ppl"))
                cw = int(row.get("channel_window", 1))
            except Exception:
                continue
            if not math.isnan(dp):
                rows.append((layer, ws, we, dp, max(1, cw)))
    return rows

# ------------------- Quant helpers (CPU search) -------------------
@torch.no_grad()
def _best_scale_es_for_vec(w_vec: torch.Tensor, nsize: int, es_cands, sweep_scales):
    sp = torch.sum(w_vec ** 2) + EPS
    best_sqnr = -float("inf")
    best = (1.0, 0)
    max_es = max(0, nsize - 1)
    es_list = [e for e in es_cands if e <= max_es] or [0]
    for sc in sweep_scales:
        ws = w_vec * sc
        for es in es_list:
            q = posit_quantize(ws, nsize=nsize, es=int(es), scale=1.0) / sc
            noise = torch.sum((w_vec - q) ** 2) + EPS
            sqnr = 10.0 * torch.log10(sp / noise)
            if float(sqnr) > best_sqnr:
                best_sqnr = float(sqnr)
                best = (float(sc), int(es))
    return best

@torch.no_grad()
def quantize_window_cpu_from_fp32(fp32_w: torch.Tensor,
                                  nsize: int,
                                  start: int,
                                  end: int,
                                  es_cands,
                                  sweep_scales) -> torch.Tensor:
    W = fp32_w.detach().float().cpu()
    if W.ndim != 2:
        raise RuntimeError(f"Expect 2D weight, got {tuple(W.shape)}")
    Cout, K = W.shape
    start = max(0, min(start, Cout))
    end = max(start, min(end, Cout))
    if start >= end:
        return W
    flat = W.view(Cout, -1)
    block = flat[start:end]
    q_block = torch.empty_like(block)
    for i in range(block.size(0)):
        w_vec = block[i]
        sc, es = _best_scale_es_for_vec(w_vec, nsize=nsize, es_cands=es_cands, sweep_scales=sweep_scales)
        q_block[i] = posit_quantize(w_vec * sc, nsize=nsize, es=es, scale=1.0) / sc
    out = flat.clone()
    out[start:end] = q_block
    return out.view_as(W)

# ------------------- Eval (chunked non-overlapping WT2) -------------------
@torch.no_grad()
def encode_wikitext2_cpu(tokenizer) -> torch.Tensor:
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(test["text"]), return_tensors="pt", add_special_tokens=False)
    return enc.input_ids

@torch.no_grad()
def eval_ppl_with_ids(model, ids_cpu: torch.Tensor, seqlen: int, use_fp16_fwd: bool) -> float:
    model.eval()
    nsamples = ids_cpu.numel() // seqlen
    if nsamples == 0:
        raise ValueError(f"Not enough tokens for seqlen={seqlen}")
    try:
        dev = next(p.device for p in model.parameters() if p.device.type != "meta")
    except StopIteration:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        autocast_ctx = torch.cuda.amp.autocast(enabled=use_fp16_fwd, dtype=torch.float16)
    else:
        class _NoOp:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        autocast_ctx = _NoOp()
    nll_sum = 0.0
    for i in range(nsamples):
        batch = ids_cpu[:, i*seqlen:(i+1)*seqlen].to(dev)
        with autocast_ctx:
            logits = model(batch).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1))
        nll_sum += (loss.item() * seqlen)
        if dev.type == "cuda":
            del batch, logits, shift_logits, shift_labels, loss
            torch.cuda.empty_cache()
    return float(math.exp(nll_sum / (nsamples * seqlen)))

# ------------------- Utility -------------------
def collect_quantizable_layers(model, skip_lm_head: bool, quantize_embeddings: bool) -> List[str]:
    names = []
    for name, mod in model.named_modules():
        if should_skip_layer(name, mod, skip_lm_head, quantize_embeddings):
            continue
        if is_quant_linear(mod) and hasattr(mod, "weight") and mod.weight is not None and mod.weight.dim() == 2:
            names.append(name)
    return names

def apply_windows_to_model(model,
                           ref_sd: Dict[str, torch.Tensor],
                           windows: List[Tuple[str, int, int, int]],
                           nsize: int,
                           es_cands,
                           sweep_scales,
                           skip_lm_head: bool,
                           quantize_embeddings: bool) -> Dict[str, Any]:
    by_layer: Dict[str, List[Tuple[int,int,int]]] = {}
    for ly, ws, we, cw in windows:
        by_layer.setdefault(ly, []).append((ws, we, cw))
    upgraded = {}
    for name, mod in model.named_modules():
        if name not in by_layer:
            continue
        if should_skip_layer(name, mod, skip_lm_head, quantize_embeddings):
            continue
        if not (is_quant_linear(mod) and hasattr(mod, "weight") and mod.weight is not None and mod.weight.dim() == 2):
            continue
        key = name + ".weight"
        if key not in ref_sd:
            print(f"[Warn] Missing FP32 ref for {key}; skip layer {name}.")
            continue
        fp32_w = ref_sd[key]
        base_w_cpu = mod.weight.detach().float().cpu()
        Q_full = base_w_cpu.clone()
        for (ws, we, _cw) in by_layer[name]:
            Q_win = quantize_window_cpu_from_fp32(fp32_w, nsize=nsize,
                                                  start=ws, end=we,
                                                  es_cands=es_cands,
                                                  sweep_scales=sweep_scales)
            Q_full[ws:we] = Q_win[ws:we]
        with torch.no_grad():
            mod.weight.data = Q_full.to(mod.weight.device, dtype=mod.weight.dtype)
        upgraded[name] = {
            "n_windows": len(by_layer[name]),
            "windows": [(ws, we) for (ws, we, _cw) in by_layer[name]],
            "nsize": nsize,
            "shape": list(fp32_w.shape),
        }
    return upgraded

# -------- NEW: candidate ordering helpers for Sweep --------
def _dedup_windows(rows: List[Tuple[str,int,int,float,int]],
                   allow_positive: bool) -> List[Tuple[str,int,int,float,int]]:
    """Deduplicate by (layer, start, end). Optionally filter ΔPPL >= 0."""
    seen = set()
    out = []
    for (ly, ws, we, dp, cw) in rows:
        if (dp >= 0.0) and (not allow_positive):
            continue
        key = (ly, ws, we)
        if key in seen:
            continue
        seen.add(key)
        out.append((ly, ws, we, dp, cw))
    return out

def _build_layer_order_map(layer_order_list: List[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(layer_order_list)}

def _order_candidates(cands: List[Tuple[str,int,int,float,int]],
                      strategy: str,
                      layer_order_map: Dict[str,int],
                      rng: np.random.Generator) -> List[Tuple[str,int,int,int,float]]:
    """
    Return [(ly, ws, we, cw, dp)] ordered per strategy.
    """
    if strategy == "sensitivity":
        ordered = sorted(cands, key=lambda x: (x[3], x[0], x[1], x[2]))  # by delta_ppl asc, tiebreakers
    elif strategy == "location":
        # layer index, then start, then end
        ordered = sorted(
            cands,
            key=lambda x: (layer_order_map.get(x[0], 1_000_000_000), x[1], x[2])
        )
    elif strategy == "random":
        ordered = list(cands)
        rng.shuffle(ordered)
    else:
        raise ValueError(f"Unknown sweep strategy: {strategy}")
    # map to (ly, ws, we, cw, dp)
    return [(ly, ws, we, cw, dp) for (ly, ws, we, dp, cw) in ordered]

# ------------------- Main -------------------
def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    q_dtype = get_torch_dtype(args.dtype)
    eval_fp16 = (args.eval_dtype == "fp16")
    device_map = None if args.device_map == "none" else "auto"
    sweep_scales = [2.0 ** k for k in range(args.log2_min, args.log2_max + 1)]
    rng = np.random.default_rng(args.random_seed)

    # 1) Read per-window sensitivity and (later) filter to quantizable
    sens_rows = read_sensitivity_windows(args.sensitivity_csv)  # (layer, ws, we, dp, cw)
    if not sens_rows:
        print("[Error] No valid sensitivity rows found.")
        return

    # 2) Load tokenizer (for optional eval)
    tok = AutoTokenizer.from_pretrained(args.base_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 3) Load FP32 reference (CPU)
    print(f"[Load FP32 reference] {args.fp32_reference_dir}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.fp32_reference_dir, torch_dtype=torch.float32, device_map=None, low_cpu_mem_usage=True
    ).cpu()
    ref_sd = {k: v.detach().cpu() for k, v in ref_model.state_dict().items()}
    del ref_model

    # 4) Probe model to (a) get quantizable set and (b) preserve layer order for 'location' sweep
    probe = AutoModelForCausalLM.from_pretrained(
        args.base_dir, torch_dtype=q_dtype,
        device_map=(device_map if args.dry_run else (None if args.device_map == "none" else "auto")),
        low_cpu_mem_usage=True
    )
    layer_order_list = collect_quantizable_layers(probe, args.skip_lm_head, args.quantize_embeddings)
    valid_layers = set(layer_order_list)
    del probe

    # Filter rows to valid layers only
    sens_rows = [row for row in sens_rows if row[0] in valid_layers]
    if not sens_rows:
        print("[Error] No sensitivity rows match quantizable modules in the base model.")
        return

    total_windows_all = len({(ly, ws, we) for (ly, ws, we, _dp, _cw) in sens_rows})

    # ---------- Mode S: Sweep (budget) ----------
    do_sweep = (args.sweep_bits_start is not None) or (args.sweep_bits_end is not None)
    if (args.ppl_goal is None) and do_sweep:
        start_bits = args.sweep_bits_start if args.sweep_bits_start is not None else args.base_bits
        end_bits   = args.sweep_bits_end   if args.sweep_bits_end   is not None else args.upgrade_bits
        if end_bits < start_bits:
            start_bits, end_bits = end_bits, start_bits
        steps = int(round((end_bits - start_bits) / max(1e-12, args.sweep_bits_step))) + 1
        targets = [round(start_bits + i*args.sweep_bits_step, 6) for i in range(steps)]
        targets[0]   = max(targets[0], args.base_bits)
        targets[-1]  = min(targets[-1], args.upgrade_bits)

        # Build candidate window list according to strategy
        # 1) dedup & optional positive filtering
        dedup = _dedup_windows(sens_rows, allow_positive=args.allow_positive_to_meet_budget)
        if not dedup:
            print("[Error] No candidate windows available for sweep.")
            return
        # 2) order
        layer_order_map = _build_layer_order_map(layer_order_list)
        cand = _order_candidates(dedup, strategy=args.sweep_strategy,
                                 layer_order_map=layer_order_map, rng=rng)
        total_cand = len(cand)

        print(f"\n[PLAN: Sweep Mode]")
        print(f"  strategy={args.sweep_strategy}  seed={args.random_seed if args.sweep_strategy=='random' else '-'}")
        print(f"  sweep_bits: {targets[0]:.3f} → {targets[-1]:.3f}  step={args.sweep_bits_step}")
        print(f"  total_candidate_windows={total_cand} (of {total_windows_all} total unique windows)")

        def k_needed(avg_bits):
            frac = (avg_bits - args.base_bits) / max(1e-12, (args.upgrade_bits - args.base_bits))
            return int(math.ceil(max(0.0, min(1.0, frac)) * total_cand))

        if args.dry_run:
            print("  First 5 targets:", ", ".join(f"{t:.2f}" for t in targets[:5]))
            print("[Dry-run] Exiting without modification.")
            return

        # Load base model to modify
        model = AutoModelForCausalLM.from_pretrained(
            args.base_dir, torch_dtype=q_dtype, device_map=device_map, low_cpu_mem_usage=True
        )
        if device_map is None and torch.cuda.is_available():
            model = model.to("cuda")

        # Prepare eval ids once
        print("[Eval] Tokenizing WikiText-2 test...")
        ids_cpu = encode_wikitext2_cpu(tok)
        use_fp16 = (args.eval_dtype == "fp16")

        # CSV + figure data
        os.makedirs(args.out_dir, exist_ok=True)
        csv_path = os.path.join(args.out_dir, "ppl_vs_avg_bits.csv")
        xs, ys = [], []

        with open(csv_path, "w", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=[
                "step", "target_avg_bits", "achieved_avg_bits",
                "windows_applied", "channels_applied", "ppl"
            ])
            writer.writeheader()

            applied_k = 0
            applied_ch = 0

            # baseline eval (0 upgrades in this session)
            ppl = eval_ppl_with_ids(model, ids_cpu, seqlen=args.seqlen, use_fp16_fwd=use_fp16)
            achieved_bits = args.base_bits + (applied_k / total_cand) * (args.upgrade_bits - args.base_bits)
            writer.writerow({
                "step": 0,
                "target_avg_bits": round(targets[0], 6),
                "achieved_avg_bits": round(achieved_bits, 6),
                "windows_applied": applied_k,
                "channels_applied": applied_ch,
                "ppl": f"{ppl:.6f}"
            })
            print(f"[Sweep 0] k={applied_k}/{total_cand}  achieved={achieved_bits:.3f} bits  PPL={ppl:.4f}")
            xs.append(achieved_bits); ys.append(ppl)

            for i, tgt in enumerate(targets, start=1):
                k_tgt = min(total_cand, max(0, k_needed(tgt)))
                if k_tgt > applied_k:
                    # apply next slice of windows (incremental, in-place) following chosen ordering
                    to_apply = cand[applied_k:k_tgt]
                    _ = apply_windows_to_model(
                        model, ref_sd,
                        [(ly, ws, we, cw) for (ly, ws, we, cw, _dp) in to_apply],
                        nsize=args.override_nsize,
                        es_cands=args.es_candidates, sweep_scales=sweep_scales,
                        skip_lm_head=args.skip_lm_head, quantize_embeddings=args.quantize_embeddings
                    )
                    applied_k = k_tgt
                    applied_ch += sum(int(cw) for (_ly, _ws, _we, cw, _dp) in to_apply)

                achieved_bits = args.base_bits + (applied_k / total_cand) * (args.upgrade_bits - args.base_bits)
                ppl = eval_ppl_with_ids(model, ids_cpu, seqlen=args.seqlen, use_fp16_fwd=use_fp16)

                writer.writerow({
                    "step": i,
                    "target_avg_bits": round(tgt, 6),
                    "achieved_avg_bits": round(achieved_bits, 6),
                    "windows_applied": applied_k,
                    "channels_applied": applied_ch,
                    "ppl": f"{ppl:.6f}"
                })
                print(f"[Sweep {i}] target={tgt:.3f}  k={applied_k}/{total_cand}  "
                      f"achieved={achieved_bits:.3f} bits  PPL={ppl:.4f}")
                xs.append(achieved_bits); ys.append(ppl)

        # Save figure
        try:
            fig_path = os.path.join(args.out_dir, "ppl_vs_avg_bits.png")
            plt.figure()
            plt.plot(xs, ys, marker="o")
            plt.xlabel("Achieved Avg Bits")
            plt.ylabel("Perplexity (PPL)")
            plt.title(f"PPL vs Achieved Avg Bits (Sweep: {args.sweep_strategy})")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"[Figure] Saved: {fig_path}")
        except Exception as e:
            print(f"[Warn] Failed to save plot: {e}")

        # Save final model + summary
        print(f"[Save] -> {args.out_dir}")
        model.save_pretrained(args.out_dir, safe_serialization=True)
        tok.save_pretrained(args.out_dir)
        with open(os.path.join(args.out_dir, "mixposit_sweep_log.json"), "w") as f:
            json.dump({
                "mode": "sweep",
                "base_dir": args.base_dir,
                "fp32_reference_dir": args.fp32_reference_dir,
                "sensitivity_csv": args.sensitivity_csv,
                "override_nsize": args.override_nsize,
                "es_candidates": args.es_candidates,
                "log2_min": args.log2_min,
                "log2_max": args.log2_max,
                "base_bits": args.base_bits,
                "upgrade_bits": args.upgrade_bits,
                "allow_positive_to_meet_budget": args.allow_positive_to_meet_budget,
                "total_candidate_windows": total_cand,
                "total_windows_all": total_windows_all,
                "sweep_strategy": args.sweep_strategy,
                "random_seed": args.random_seed if args.sweep_strategy == "random" else None,
                "ppl_csv": "ppl_vs_avg_bits.csv",
                "ppl_png": "ppl_vs_avg_bits.png"
            }, f, indent=2)
        print("[OK] Sweep complete.")
        return

    # ------- Mode C: Default (ΔPPL < 0 windows) -------
    # (unchanged)
    if args.target_avg_bits is None and args.ppl_goal is None:
        sens_sorted = sorted(sens_rows, key=lambda x: (x[3], x[0], x[1], x[2]))
        neg_rows = [(ly, ws, we, dp, cw) for (ly, ws, we, dp, cw) in sens_sorted if dp < 0.0]
        seen = set(); chosen_windows = []
        for (ly, ws, we, dp, cw) in neg_rows:
            key = (ly, ws, we)
            if key in seen: continue
            seen.add(key)
            chosen_windows.append((ly, ws, we, cw))
        windows_selected = len(chosen_windows)
        total_windows = len({(ly, ws, we) for (ly, ws, we, _dp, _cw) in sens_rows})
        achieved_avg_bits = args.base_bits + (min(windows_selected, total_windows) / total_windows) * (args.upgrade_bits - args.base_bits)

        print("\n[PLAN: Default Mode]")
        print(f"  total_windows={total_windows}")
        print(f"  windows_selected={windows_selected}")
        print(f"  achieved_avg_bits={achieved_avg_bits:.3f}")

        if args.dry_run:
            print("[Dry-run] Exiting without modification.")
            return

        model = AutoModelForCausalLM.from_pretrained(
            args.base_dir, torch_dtype=q_dtype, device_map=device_map, low_cpu_mem_usage=True
        )
        if device_map is None and torch.cuda.is_available():
            model = model.to("cuda")

        _ = apply_windows_to_model(
            model, ref_sd, chosen_windows, nsize=args.override_nsize,
            es_cands=args.es_candidates, sweep_scales=sweep_scales,
            skip_lm_head=args.skip_lm_head, quantize_embeddings=args.quantize_embeddings
        )

        final_ppl = None
        if args.eval_final_ppl:
            print("[Eval] Computing final PPL on WikiText-2...")
            ids_cpu = encode_wikitext2_cpu(tok)
            final_ppl = eval_ppl_with_ids(model, ids_cpu, seqlen=args.seqlen, use_fp16_fwd=eval_fp16)
            print(f"[Final PPL] {final_ppl:.4f}")

        print(f"[Save] -> {args.out_dir}")
        model.save_pretrained(args.out_dir, safe_serialization=True)
        tok.save_pretrained(args.out_dir)
        with open(os.path.join(args.out_dir, "mixposit_default_log.json"), "w") as f:
            json.dump({
                "mode": "default_negative_only",
                "base_bits": args.base_bits,
                "upgrade_bits": args.upgrade_bits,
                "total_windows": total_windows,
                "windows_selected": windows_selected,
                "achieved_avg_bits": achieved_avg_bits,
                "final_ppl": final_ppl
            }, f, indent=2)
        print("[OK] Saved mixed-precision model and log.")
        return

    # ------- Mode A: Budget -------
    if args.ppl_goal is None and args.target_avg_bits is not None:
        sens_sorted = sorted(sens_rows, key=lambda x: (x[3], x[0], x[1], x[2]))
        if not (args.base_bits <= args.target_avg_bits <= args.upgrade_bits):
            raise ValueError(f"--target_avg_bits must be within [{args.base_bits}, {args.upgrade_bits}]")
        total_windows = len({(ly, ws, we) for (ly, ws, we, _dp, _cw) in sens_sorted})
        n_upgrade = int(math.ceil(
            total_windows * (args.target_avg_bits - args.base_bits) /
            max(1e-12, (args.upgrade_bits - args.base_bits))
        ))

        if args.downgrade:
            # DOWNGRADE MODE: start from upgrade_bits ckpt, demote least-sensitive windows to base_bits.
            n_downgrade = total_windows - n_upgrade
            need_windows = n_downgrade
            # Pick the LAST n_downgrade unique windows from ascending-delta_ppl sort
            # (smallest PPL benefit from upgrading → safest to demote).
            picked = []
            seen = set()
            for (ly, ws, we, dp, cw) in reversed(sens_sorted):
                if len(picked) >= n_downgrade:
                    break
                key = (ly, ws, we)
                if key in seen: continue
                seen.add(key)
                picked.append((ly, ws, we, cw))
            apply_nsize = args.downgrade_nsize if args.downgrade_nsize is not None else int(args.base_bits)
            # windows kept at upgrade_bits = total - picked
            achieved_avg_bits = args.upgrade_bits - (len(picked) / total_windows) * (args.upgrade_bits - args.base_bits)
        else:
            # UPGRADE MODE (default): start from base_bits ckpt, promote most-sensitive windows to upgrade_bits.
            need_windows = n_upgrade
            picked = []
            seen = set()
            for (ly, ws, we, dp, cw) in sens_sorted:
                if len(picked) >= need_windows:
                    break
                if (dp < 0.0) or args.allow_positive_to_meet_budget:
                    key = (ly, ws, we)
                    if key in seen: continue
                    seen.add(key)
                    picked.append((ly, ws, we, cw))
            apply_nsize = args.override_nsize
            achieved_avg_bits = args.base_bits + (min(len(picked), total_windows) / total_windows) * (args.upgrade_bits - args.base_bits)

        windows_selected = len(picked)
        mode_tag = "downgrade" if args.downgrade else "upgrade"

        print(f"\n[PLAN: Budget Mode / {mode_tag}]")
        print(f"  total_windows={total_windows}, needed_windows={need_windows}")
        print(f"  windows_selected={windows_selected}  (apply_nsize={apply_nsize})")
        print(f"  achieved_avg_bits={achieved_avg_bits:.3f} (target={args.target_avg_bits:.3f})")

        if args.dry_run:
            print("[Dry-run] Exiting without modification.")
            return

        model = AutoModelForCausalLM.from_pretrained(
            args.base_dir, torch_dtype=q_dtype, device_map=device_map, low_cpu_mem_usage=True
        )
        if device_map is None and torch.cuda.is_available():
            model = model.to("cuda")

        total_channels = sum(int(cw) for (_, _, _, cw) in picked) if picked else 0
        done_windows = 0
        done_channels = 0
        touched_by_layer: Dict[str, Dict[str, Any]] = {}

        for (ly, ws, we, cw) in picked:
            _ = apply_windows_to_model(
                model, ref_sd, [(ly, ws, we, cw)], nsize=apply_nsize,
                es_cands=args.es_candidates, sweep_scales=sweep_scales,
                skip_lm_head=args.skip_lm_head, quantize_embeddings=args.quantize_embeddings
            )
            done_windows += 1
            done_channels += int(cw)
            print(f"[Budget] progress: {done_windows}/{windows_selected} windows, "
                  f"{done_channels}/{total_channels} channels")
            if ly not in touched_by_layer:
                key = ly + ".weight"
                shape = list(ref_sd[key].shape) if key in ref_sd else []
                touched_by_layer[ly] = {"n_windows": 0, "windows": [], "nsize": apply_nsize, "shape": shape}
            touched_by_layer[ly]["n_windows"] += 1
            touched_by_layer[ly]["windows"].append((ws, we))

        print("[Eval] Computing final PPL on WikiText-2 (Budget mode)...")
        ids_cpu = encode_wikitext2_cpu(tok)
        final_ppl = eval_ppl_with_ids(model, ids_cpu, seqlen=args.seqlen, use_fp16_fwd=eval_fp16)
        print(f"[Final PPL] {final_ppl:.4f}")

        print(f"[Save] -> {args.out_dir}")
        model.save_pretrained(args.out_dir, safe_serialization=True)
        tok.save_pretrained(args.out_dir)
        with open(os.path.join(args.out_dir, "mixposit_budget_log.json"), "w") as f:
            json.dump({
                "mode": "budget",
                "direction": mode_tag,
                "target_avg_bits": args.target_avg_bits,
                "achieved_avg_bits": achieved_avg_bits,
                "base_bits": args.base_bits,
                "upgrade_bits": args.upgrade_bits,
                "apply_nsize": apply_nsize,
                "total_windows": total_windows,
                "windows_selected": windows_selected,
                "channels_selected": total_channels,
                "final_windows_applied": done_windows,
                "final_channels_applied": done_channels,
                "touched": touched_by_layer,
                "final_ppl": final_ppl
            }, f, indent=2)
        print("[OK] Saved mixed-precision model and log.")
        return

    # ------- Mode B: PPL Goal (unchanged) -------
    if args.ppl_goal is not None:
        sens_sorted = sorted(sens_rows, key=lambda x: (x[3], x[0], x[1], x[2]))
        model = AutoModelForCausalLM.from_pretrained(
            args.base_dir, torch_dtype=q_dtype, device_map=device_map, low_cpu_mem_usage=True
        )
        if device_map is None and torch.cuda.is_available():
            model = model.to("cuda")

        print("[Eval] Tokenizing WikiText-2 test...")
        ids_cpu = encode_wikitext2_cpu(tok)
        print("[Eval] Computing baseline PPL...]")
        ppl = eval_ppl_with_ids(model, ids_cpu, seqlen=args.seqlen, use_fp16_fwd=eval_fp16)
        print(f"[Baseline] PPL = {ppl:.4f}")

        curve_csv_path = os.path.join(args.out_dir, "ppl_vs_windows.csv")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(curve_csv_path, "w", newline="") as fcsv:
            writer = csv.DictWriter(
                fcsv,
                fieldnames=["step", "windows_upgraded", "channels_upgraded", "layer", "win_start", "win_end", "ppl"]
            )
            writer.writeheader()
            writer.writerow({
                "step": 0,
                "windows_upgraded": 0,
                "channels_upgraded": 0,
                "layer": "",
                "win_start": "",
                "win_end": "",
                "ppl": f"{ppl:.6f}"
            })

        upgraded_windows: List[Tuple[str,int,int,int]] = []
        start_t = time.time()
        windows_upgraded = 0
        channels_upgraded = 0
        step = 0

        for (ly, ws, we, dp, cw) in sens_sorted:
            if ppl <= args.ppl_goal:
                break
            if (dp >= 0.0) and (not args.allow_positive_to_meet_budget):
                continue
            upgraded_windows.append((ly, ws, we, cw))
            _ = apply_windows_to_model(
                model, ref_sd, [(ly, ws, we, cw)], nsize=args.override_nsize,
                es_cands=args.es_candidates, sweep_scales=sweep_scales,
                skip_lm_head=args.skip_lm_head, quantize_embeddings=args.quantize_embeddings
            )
            ppl = eval_ppl_with_ids(model, ids_cpu, seqlen=args.seqlen, use_fp16_fwd=eval_fp16)
            print(f"  [{ly} {ws}:{we}) -> PPL {ppl:.4f}")

            windows_upgraded += 1
            channels_upgraded += int(cw)
            step += 1

            with open(curve_csv_path, "a", newline="") as fcsv:
                writer = csv.DictWriter(
                    fcsv,
                    fieldnames=["step", "windows_upgraded", "channels_upgraded", "layer", "win_start", "win_end", "ppl"]
                )
                writer.writerow({
                    "step": step,
                    "windows_upgraded": windows_upgraded,
                    "channels_upgraded": channels_upgraded,
                    "layer": ly,
                    "win_start": ws,
                    "win_end": we,
                    "ppl": f"{ppl:.6f}"
                })

        elapsed = time.time() - start_t
        total_windows = len({(ly, ws, we) for (ly, ws, we, _dp, _cw) in sens_rows})
        achieved_avg_bits = args.base_bits + (min(windows_upgraded, total_windows) / total_windows) * (args.upgrade_bits - args.base_bits)

        plot_path = os.path.join(args.out_dir, "ppl_vs_channels.png")
        try:
            # (optional) could log during loop as well
            pass
        except Exception:
            pass

        print("\n[RESULT: PPL Goal Mode]")
        print(f"  ppl_goal={args.ppl_goal:.4f}, final_ppl={ppl:.4f}, time={elapsed/60:.1f} min")
        print(f"  windows_upgraded={windows_upgraded}/{total_windows}")
        print(f"  achieved_avg_bits={achieved_avg_bits:.3f}  (base={args.base_bits}, upgrade={args.upgrade_bits})")
        print(f"  [Curve] CSV saved to: {curve_csv_path}")

        if args.dry_run:
            print("[Dry-run] Exiting without saving.")
            return

        print(f"[Save] -> {args.out_dir}")
        model.save_pretrained(args.out_dir, safe_serialization=True)
        tok.save_pretrained(args.out_dir)
        with open(os.path.join(args.out_dir, "mixposit_pplgoal_log.json"), "w") as f:
            json.dump({
                "mode": "ppl_goal",
                "ppl_goal": args.ppl_goal,
                "final_ppl": ppl,
                "elapsed_sec": elapsed,
                "base_bits": args.base_bits,
                "upgrade_bits": args.upgrade_bits,
                "total_windows": total_windows,
                "windows_upgraded": windows_upgraded,
                "channels_upgraded": channels_upgraded,
                "upgraded_windows": [{"layer": ly, "start": ws, "end": we, "channel_window": cw}
                                     for (ly, ws, we, cw) in upgraded_windows]
            }, f, indent=2)
        print("[OK] Saved model and PPL-goal log.")
        return

    print("[Error] Invalid mode selection.")

if __name__ == "__main__":
    main()
