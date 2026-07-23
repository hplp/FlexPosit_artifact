# FlexPosit MICRO 2026 Artifact

See the paper's Artifact Appendix for the full description. This README is a quick reference.

## Reproduce

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # required for gated Llama-2

bash 01_install.sh              # ~20 min
bash 02_headline_ppl.sh         # Table 2       13-17 h
bash 03_ablation.sh             # Table 3       1 h
bash 04_mpq_granularity.sh      # Table 4       1 min
bash 05_hardware.sh             # Figures 11-12, Table 10  ~2 h
bash 06_act_quant.sh            # Table 6       0.5-1 h
```

Each script writes its CSV to `results/` and ends with a `verify.py` diff against `expected/` (±0.1 PPL tolerance).

## Requirements

- 1x NVIDIA GPU with ≥40 GB VRAM and compute capability 8.0–9.0 (A40, A6000, L40S, A100, RTX 6000 Ada, H100, H200). Blackwell (compute 10.0+) requires PyTorch 2.5+ with CUDA 12.4+; swap `requirements.txt` accordingly.
- ~250 GB free disk
- Miniforge/Miniconda; NVIDIA driver ≥520
- HuggingFace token with access to `meta-llama/Llama-2-7b-hf`

If Meta approval is pending, pass `SKIP_LLAMA2=1` to reproduce the other 8 models.
On a 24 GB card, pass `SKIP_QWEN14B=1` to skip the 14B model.

## Contact

Yimin Gao <yg9bq@virginia.edu>
