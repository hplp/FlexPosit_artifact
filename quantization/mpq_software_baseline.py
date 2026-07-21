#!/usr/bin/env python3
"""Software-only coarse MPQ baseline (FlexPosit rebuttal).

Substrate (fixed): per-channel PoT-scale SQNR-best Posit(nsize, es=1) — same as
FlexPosit. Activations stay FP16/BF16. Hardware-agnostic.

Granularity (configurable):
  --granularity layer    candidate region = one transformer block
  --granularity module   candidate region = attention or MLP sub-block within a block

Sensitivity (configurable):
  --sensitivity fisher
    SqueezeLLM-style empirical Fisher-diagonal × quantization-perturbation²:
      score(R) = Σ_{i ∈ R} F_ii · ((w_i - Q_{b_low}(w_i))^2 - (w_i - Q_{b_high}(w_i))^2)
    F_ii = (1/N) Σ_n (∂L_n/∂w_i)^2 estimated on the FP reference model with
    WikiText-2 train calibration (next-token CE loss).

  --sensitivity ppl_probe
    FlexPosit Algorithm 2 style PPL-probe: with all weights at b_low, temporarily
    upgrade region R from b_low to b_high, measure WikiText-2 test PPL, restore.
      delta_ppl(R) = base_ppl - upgraded_ppl  (positive = beneficial)

Score-per-bit:
  Greedy ranking uses score(R) / region_weights, which is identical to score per
  added stored bit since (b_high - b_low) is constant within a sweep.

Sweep:
  Rank regions DESCENDING by score-per-weight, commit until upgraded_weights /
  total >= target_fraction. Average bits is weighted by weight count. Evaluate
  WikiText-2 PPL at each target ∈ {4.0..5.0 step 0.1}.

Output files in --out_dir:
  sensitivity.csv         (region, weights, raw_score, score_per_weight)
  ppl_vs_avg_bits.csv     (step, target, achieved, regions_upgraded, weights_upgraded, ppl)
  run_log.json            (config + base_ppl + ranked region order + selected regions per target)
"""
import argparse, csv, gc, json, math, os, re, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import transformers.modeling_utils as modeling_utils
from transformers import AutoModelForCausalLM, AutoTokenizer
from qtorch_plus.quant import posit_quantize
from datasets import load_dataset

transformers.logging.set_verbosity_error()
EPS = 1e-8
LAYER_RE = re.compile(r"^(.*?\.(?:h|layers)\.\d+)\.")


def is_quant_linear(mod):
    return isinstance(mod, nn.Linear) or isinstance(mod, modeling_utils.Conv1D)


def layer_of(name):
    m = LAYER_RE.match(name)
    return m.group(1) if m else None


def module_kind(name):
    """attn | mlp heuristic (OPT/Phi-2/LLaMA/Mistral/DeepSeek)."""
    n = name.lower()
    if "attn" in n or "attention" in n:
        return "attn"
    return "mlp"


@torch.no_grad()
def quantize_pc_posit(W_fp, nsize, es=1, log2_min=-8, log2_max=9, device="cuda"):
    """Per-channel PoT-scale SQNR-best Posit(nsize, es). CPU FP32 in/out."""
    W = W_fp.to(device).float()
    Cout, K = W.shape
    scales = torch.tensor(
        [2.0 ** i for i in range(log2_min, log2_max + 1)],
        device=device, dtype=torch.float32,
    )
    S = scales.numel()
    W_scaled = W[:, None, :] * scales[None, :, None]
    q_2d = posit_quantize(
        W_scaled.reshape(Cout * S, K), nsize=nsize, es=es, scale=1.0
    ).reshape(Cout, S, K)
    Q = q_2d / scales[None, :, None]
    sig = (W[:, None, :] ** 2).sum(dim=-1) + EPS
    noise = ((W[:, None, :] - Q) ** 2).sum(dim=-1) + EPS
    sqnr = 10.0 * torch.log10(sig / noise)
    best_s = sqnr.argmax(dim=-1)
    out = torch.empty_like(W)
    for c in range(Cout):
        s = float(scales[best_s[c]].item())
        out[c] = posit_quantize(W[c] * s, nsize=nsize, es=es, scale=1.0) / s
    return out.detach().cpu()


@torch.no_grad()
def eval_wikitext_ppl(model, tok, seqlen, use_fp16=True):
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(test["text"]), return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids
    n = ids.numel() // seqlen
    dev = next(p.device for p in model.parameters() if p.device.type != "meta")
    model.eval()
    nll = 0.0
    ctx = (
        torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
        if use_fp16 and dev.type == "cuda"
        else torch.cuda.amp.autocast(enabled=False)
    )
    with ctx:
        for i in range(n):
            batch = ids[:, i * seqlen : (i + 1) * seqlen].to(dev)
            logits = model(batch).logits
            sl = logits[:, :-1, :].contiguous().float()
            lab = batch[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lab.view(-1))
            nll += loss.item() * seqlen
            del batch, logits, sl, lab, loss
    return math.exp(nll / (n * seqlen))


def collect_region_layout(model, granularity):
    layer_groups = {}
    for name, mod in model.named_modules():
        if not is_quant_linear(mod):
            continue
        if name == "lm_head" or name.endswith(".lm_head"):
            continue
        if not hasattr(mod, "weight") or mod.weight is None or mod.weight.dim() != 2:
            continue
        pre = layer_of(name)
        if pre is None:
            continue
        layer_groups.setdefault(pre, []).append(name)

    def lidx(p):
        m = re.search(r"\.(\d+)$", p)
        return int(m.group(1)) if m else -1
    layers = sorted(layer_groups.keys(), key=lidx)

    regions = []
    if granularity == "layer":
        for p in layers:
            regions.append((p, layer_groups[p]))
    elif granularity == "module":
        for p in layers:
            attn = [n for n in layer_groups[p] if module_kind(n) == "attn"]
            mlp = [n for n in layer_groups[p] if module_kind(n) == "mlp"]
            if attn:
                regions.append((f"{p}.attn", attn))
            if mlp:
                regions.append((f"{p}.mlp", mlp))
    else:
        raise ValueError(granularity)
    return regions


def calib_chunks(tok, seqlen, n_samples, split="train"):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids
    take = min(n_samples, ids.numel() // seqlen)
    for i in range(take):
        yield ids[:, i * seqlen : (i + 1) * seqlen]


def compute_fisher_diagonal(model_fp, tok, seqlen, n_calib):
    """F_ii = (1/N) Σ_n (∂L_n/∂w_i)^2; returns dict {param_name: CPU FP32 tensor}."""
    model_fp.train()
    try:
        model_fp.gradient_checkpointing_enable()
        model_fp.config.use_cache = False
    except Exception as e:
        print(f"  [warn] gradient_checkpointing_enable failed: {e}", flush=True)

    want = set()
    for name, mod in model_fp.named_modules():
        if is_quant_linear(mod) and mod.weight is not None and mod.weight.dim() == 2:
            if name == "lm_head" or name.endswith(".lm_head"):
                continue
            if layer_of(name) is None:
                continue
            want.add(name + ".weight")

    grad_sq = {n: torch.zeros(p.shape, dtype=torch.float32, device="cpu")
               for n, p in model_fp.named_parameters() if n in want}
    dev = next(p.device for p in model_fp.parameters() if p.device.type != "meta")

    n_done = 0
    for i, batch in enumerate(calib_chunks(tok, seqlen, n_calib)):
        batch = batch.to(dev)
        model_fp.zero_grad(set_to_none=True)
        out = model_fp(batch)
        logits = out.logits
        sl = logits[:, :-1, :].contiguous().float()
        lab = batch[:, 1:].contiguous()
        loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lab.view(-1))
        loss.backward()
        with torch.no_grad():
            for n, p in model_fp.named_parameters():
                if n in grad_sq and p.grad is not None:
                    grad_sq[n] += (p.grad.detach().float() ** 2).cpu()
        n_done += 1
        if (i + 1) % 16 == 0 or i + 1 == n_calib:
            print(f"  [Fisher] {i+1}/{n_calib} batches", flush=True)
    for n in grad_sq:
        grad_sq[n] /= max(n_done, 1)
    return grad_sq


def fisher_sensitivity_per_region(layout, ref_sd, fisher_diag, b_low, b_high, es, device):
    sens = {}
    for key, names in layout:
        s = 0.0
        for n in names:
            wkey = n + ".weight"
            if wkey not in ref_sd or wkey not in fisher_diag:
                continue
            w_fp = ref_sd[wkey].float()
            w_low = quantize_pc_posit(w_fp, nsize=b_low, es=es, device=device)
            w_high = quantize_pc_posit(w_fp, nsize=b_high, es=es, device=device)
            d_low_sq = (w_fp - w_low) ** 2
            d_high_sq = (w_fp - w_high) ** 2
            s += float((fisher_diag[wkey] * (d_low_sq - d_high_sq)).sum().item())
        sens[key] = s
    return sens


def ppl_probe_sensitivity_per_region(
    model_q, tok, layout, ref_sd, key_to_mods, b_high, es, seqlen, dtype, base_ppl
):
    """FlexPosit Algorithm-2 style: temporarily upgrade each region b_low->b_high,
    measure PPL, restore. Returns dict {region: delta_ppl = base - new}.
    Also writes incremental rows to a CSV-like list (caller will flush)."""
    sens = {}
    use_fp16 = (dtype == torch.float16)
    for key, names in layout:
        mods = key_to_mods[key]
        saved = {n: m.weight.data.clone() for n, m in mods}
        t0 = time.time()
        for n, m in mods:
            wkey = n + ".weight"
            if wkey not in ref_sd:
                print(f"  [warn] missing ref weight {wkey}", flush=True)
                continue
            wq = quantize_pc_posit(
                ref_sd[wkey].float(), nsize=b_high, es=es, device=m.weight.device
            )
            m.weight.data = wq.to(m.weight.dtype).to(m.weight.device)
        ppl = eval_wikitext_ppl(model_q, tok, seqlen, use_fp16=use_fp16)
        for n, m in mods:
            m.weight.data = saved[n]
        del saved
        delta = base_ppl - ppl  # positive = improvement
        sens[key] = delta
        dt = time.time() - t0
        print(
            f"  [probe {key}] PPL={ppl:.4f} delta=+{delta:.4f}  ({dt:.1f}s)",
            flush=True,
        )
    return sens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True)
    p.add_argument("--ref_id", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--granularity", choices=["layer", "module"], default="layer")
    p.add_argument("--sensitivity", choices=["fisher", "ppl_probe"], default="fisher")
    p.add_argument("--b_low", type=int, default=4)
    p.add_argument("--b_high", type=int, default=8)
    p.add_argument("--es", type=int, default=1)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--fisher_seqlen", type=int, default=512)
    p.add_argument("--n_calib", type=int, default=128)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[args] {vars(args)}", flush=True)
    log = {"args": vars(args)}

    tok = AutoTokenizer.from_pretrained(args.base_dir, use_fast=True)
    td = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    # =====================================================================
    # Compute sensitivity → dict {region: raw_score}
    # =====================================================================
    if args.sensitivity == "fisher":
        # Phase 1a: FP reference + Fisher pass
        print(f"[Phase 1a Fisher] Load FP {args.ref_id}", flush=True)
        model_fp = AutoModelForCausalLM.from_pretrained(
            args.ref_id, torch_dtype=td, device_map="auto", low_cpu_mem_usage=True,
        )
        layout = collect_region_layout(model_fp, args.granularity)
        name_to_numel = {n: p.numel() for n, p in model_fp.named_parameters()}
        region_weights = {
            k: sum(name_to_numel.get(n + ".weight", 0) for n in names)
            for k, names in layout
        }
        total_w = sum(region_weights.values())
        print(f"[Regions] {args.granularity} count={len(layout)} total_w={total_w}", flush=True)

        print(f"[Fisher] n_calib={args.n_calib} fisher_seqlen={args.fisher_seqlen}", flush=True)
        t0 = time.time()
        fisher_diag = compute_fisher_diagonal(model_fp, tok, args.fisher_seqlen, args.n_calib)
        print(f"[Fisher] done ({time.time()-t0:.1f}s)", flush=True)

        print("[Snapshot ref weights -> CPU FP16]", flush=True)
        ref_sd = {n: p.detach().cpu().to(torch.float16)
                  for n, p in model_fp.named_parameters() if n.endswith(".weight")}

        # Free FP model BEFORE the SQNR-search inner buffer is allocated, otherwise
        # the residual activations + grad-checkpointing buffers OOM on 7B + A100-40GB.
        del model_fp
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[Sensitivity] Fisher  {args.b_low}b -> {args.b_high}b ...", flush=True)
        t0 = time.time()
        sens = fisher_sensitivity_per_region(
            layout, ref_sd, fisher_diag, args.b_low, args.b_high, args.es, device="cuda"
        )
        print(f"[Sensitivity] done ({time.time()-t0:.1f}s)", flush=True)

        del fisher_diag
        gc.collect()
        torch.cuda.empty_cache()

        # Load quant model for sweep
        print(f"[Phase 2] Load base posit-{args.b_low} model {args.base_dir}", flush=True)
        model_q = AutoModelForCausalLM.from_pretrained(
            args.base_dir, torch_dtype=td, device_map="auto", low_cpu_mem_usage=True,
        )
        q_modules = {n: m for n, m in model_q.named_modules() if is_quant_linear(m)}
        key_to_mods = {k: [(n, q_modules[n]) for n in names if n in q_modules]
                       for k, names in layout}

        print(f"[Baseline b={args.b_low}] eval PPL...", flush=True)
        t0 = time.time()
        base_ppl = eval_wikitext_ppl(model_q, tok, args.seqlen, use_fp16=(args.dtype == "fp16"))
        print(f"[Baseline] PPL = {base_ppl:.4f}  ({time.time()-t0:.1f}s)", flush=True)

        score_label = "fisher_score"

    else:  # ppl_probe
        # Phase 1: load FP-ref for sd, then load quant for probe-eval
        print(f"[Phase 1a PPL-probe] Load FP {args.ref_id} for ref_sd", flush=True)
        ref = AutoModelForCausalLM.from_pretrained(
            args.ref_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).cpu()
        ref_sd = {n: p.detach().cpu().to(torch.float16)
                  for n, p in ref.named_parameters() if n.endswith(".weight")}
        del ref
        gc.collect()

        print(f"[Phase 1b] Load base posit-{args.b_low} model {args.base_dir}", flush=True)
        model_q = AutoModelForCausalLM.from_pretrained(
            args.base_dir, torch_dtype=td, device_map="auto", low_cpu_mem_usage=True,
        )
        layout = collect_region_layout(model_q, args.granularity)
        name_to_numel = {n: p.numel() for n, p in model_q.named_parameters()}
        region_weights = {
            k: sum(name_to_numel.get(n + ".weight", 0) for n in names)
            for k, names in layout
        }
        total_w = sum(region_weights.values())
        print(f"[Regions] {args.granularity} count={len(layout)} total_w={total_w}", flush=True)

        q_modules = {n: m for n, m in model_q.named_modules() if is_quant_linear(m)}
        key_to_mods = {k: [(n, q_modules[n]) for n in names if n in q_modules]
                       for k, names in layout}

        print(f"[Baseline b={args.b_low}] eval PPL...", flush=True)
        t0 = time.time()
        base_ppl = eval_wikitext_ppl(model_q, tok, args.seqlen, use_fp16=(args.dtype == "fp16"))
        print(f"[Baseline] PPL = {base_ppl:.4f}  ({time.time()-t0:.1f}s)", flush=True)

        print(f"[Sensitivity] PPL-probe  upgrading each region {args.b_low}b -> {args.b_high}b", flush=True)
        t0 = time.time()
        sens = ppl_probe_sensitivity_per_region(
            model_q, tok, layout, ref_sd, key_to_mods,
            args.b_high, args.es, args.seqlen, td, base_ppl,
        )
        print(f"[Sensitivity] done ({time.time()-t0:.1f}s)", flush=True)

        score_label = "delta_ppl"

    log["base_ppl"] = base_ppl
    log["total_weights"] = total_w

    # =====================================================================
    # Write sensitivity CSV
    # =====================================================================
    sens_csv = os.path.join(args.out_dir, "sensitivity.csv")
    with open(sens_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "weights", score_label, "score_per_weight"])
        for k in sorted(sens.keys(), key=lambda x: -sens[x]):
            w.writerow([
                k, region_weights[k],
                f"{sens[k]:.6e}",
                f"{sens[k]/max(region_weights[k],1):.6e}",
            ])
    print(f"[Saved] {sens_csv}", flush=True)

    # =====================================================================
    # Greedy sweep
    # =====================================================================
    # Rank by score-per-weight DESCENDING.  For Fisher: positive = beneficial.
    # For PPL-probe delta = base-new: positive = beneficial.
    ranked = sorted(
        sens.keys(),
        key=lambda k: -(sens[k] / max(region_weights[k], 1))
    )
    print(f"[Ranked top 5] {ranked[:5]}", flush=True)
    log["ranked_order"] = ranked

    sweep_csv = os.path.join(args.out_dir, "ppl_vs_avg_bits.csv")
    targets = [round(4.0 + 0.1 * i, 1) for i in range(11)]
    upgraded = []
    upgraded_w = 0
    selected_at_target = {}
    with open(sweep_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["step", "target_avg_bits", "achieved_avg_bits",
                     "regions_upgraded", "weights_upgraded", "ppl"])
        wr.writerow([0, 4.0, 4.0, 0, 0, f"{base_ppl:.6f}"])
        f.flush()
        for step_i, tgt in enumerate(targets, start=1):
            need_w = (tgt - args.b_low) * total_w / (args.b_high - args.b_low)
            while upgraded_w < need_w and len(upgraded) < len(ranked):
                nxt = ranked[len(upgraded)]
                for n, m in key_to_mods[nxt]:
                    wkey = n + ".weight"
                    if wkey not in ref_sd:
                        continue
                    wq = quantize_pc_posit(
                        ref_sd[wkey].float(), nsize=args.b_high, es=args.es,
                        device=m.weight.device,
                    )
                    m.weight.data = wq.to(m.weight.dtype).to(m.weight.device)
                upgraded.append(nxt)
                upgraded_w += region_weights[nxt]
            ach = args.b_low + (upgraded_w / total_w) * (args.b_high - args.b_low)
            t0 = time.time()
            ppl = eval_wikitext_ppl(model_q, tok, args.seqlen, use_fp16=(args.dtype == "fp16"))
            dt = time.time() - t0
            wr.writerow([
                step_i, f"{tgt:.1f}", f"{ach:.6f}",
                len(upgraded), upgraded_w, f"{ppl:.6f}",
            ])
            f.flush()
            selected_at_target[f"{tgt:.1f}"] = list(upgraded)
            print(
                f"  [sweep {step_i}] tgt={tgt:.1f} regions={len(upgraded)} "
                f"achieved={ach:.3f} PPL={ppl:.4f}  ({dt:.1f}s)",
                flush=True,
            )

    log["selected_regions_per_target"] = selected_at_target
    with open(os.path.join(args.out_dir, "run_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"[Done] sensitivity: {sens_csv}")
    print(f"[Done] sweep:       {sweep_csv}")


if __name__ == "__main__":
    main()
