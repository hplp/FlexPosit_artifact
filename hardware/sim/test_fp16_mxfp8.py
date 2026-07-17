# python test_fp16_mxfp8.py --is_generation > ./log/test_fp16_mxfp8.log
#
# FP16-MXFP8 accelerator: 16-bit activations x MXFP8 (8-bit) weights, parallel
# (non-bit-serial) float datapath. Hardware metrics taken from "Table 5: Iso-area
# configurations": Per-PE area Comp=723.2 / Norm=0.73x / Real=768.0 um^2, iso-area
# config 9x16 -> 0.1106 mm^2 (768.0 * 9 * 16 = 0.1106 mm^2, matches the table).
#
# pe_energy: per-PE power for FP16-MXFP8 is 378.6 uW. In this model the per-PE energy
# values are pJ/op, which at the implied 1 GHz clock equal per-PE power in mW
# (e.g. FP16 0.475 -> 475 uW, Olive 0.33406 -> 334 uW). So 378.6 uW -> 0.3786 pJ/op.
import argparse
from accelerator import Accelerator
from ramulator_dram_sim import get_cache_stats

model_list = [
    "gpt2-large", "gpt2-xl", "microsoft/phi-2", "facebook/opt-2.7b", "meta-llama/Llama-2-7b-hf",
    "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.1", "deepseek-ai/deepseek-llm-7b-base", "Qwen/Qwen2.5-14B",
]

# Hardware metrics from Table 5 (FP16-MXFP8 row)
PE_ENERGY = 0.3786  # per-PE power 378.6 uW = 0.3786 pJ/op at 1 GHz (provided)
PE_AREA   = 768.0   # Real per-PE area (um^2), includes local pipeline registers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_generation", action="store_true", help="If enabled, then evaluate")
    parser.add_argument("--pe_x", type=int, default=None, help="PE array dimension X")
    parser.add_argument("--pe_y", type=int, default=None, help="PE array dimension Y")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of sequences decoded in parallel (generation mode)")
    parser.add_argument("--context_length", type=int, default=256, help="Sequence length (in tokens) for the workload")
    args = parser.parse_args()
    is_generation = args.is_generation
    pe_x = args.pe_x
    pe_y = args.pe_y

    if pe_x is not None and pe_y is not None:
        pe_array_dim = [pe_x, pe_y]
    elif is_generation:
        pe_array_dim = [9, 16]   # iso-area config from Table 5
    else:
        pe_array_dim = [16, 16]

    total_energy_list = [[0, 0] for _ in model_list]
    total_latency_list = [0 for _ in model_list]

    # Print accelerator configuration
    print("Accelerator: FP16-MXFP8 (FP16 act x MXFP8 weight)")
    print(f"PE Array Dimension: {pe_array_dim}")
    print(f"Input Precision: 16-bit, Weight Precision: 8-bit (MXFP8)")
    print(f"Context Length: 256, Generation Mode: {is_generation}")
    print(f"Models to test: {len(model_list)}")
    print(f"pe_energy: {PE_ENERGY}  (estimated 0.73x FP16; table gives area only)")
    print(f"pe_area: {PE_AREA}")
    print()

    total_ops_list = {
        'gpt2-large': 65434880*2,
        'gpt2-xl': 82638400*2,
        'microsoft/phi-2': 2650537984*2,
        'facebook/opt-2.7b': 2648162304*2,
        'meta-llama/Llama-2-7b-hf': 6611533824*2,
        'Qwen/Qwen2.5-7B': 6611533824*2,
        'mistralai/Mistral-7B-v0.1': 6611533824*2,
        'deepseek-ai/deepseek-llm-7b-base': 6611533824*2,
        'Qwen/Qwen2.5-14B': 13223067648*2,
    }

    for idx, model_name in enumerate(model_list):
        acc = Accelerator(
            model_name=model_name,
            i_prec=16,
            w_prec=8,
            is_bit_serial=False,
            pe_dp_size=1,
            pe_energy=PE_ENERGY,
            pe_area=PE_AREA,
            pe_array_dim=pe_array_dim,
            context_length=args.context_length,
            batch_size=args.batch_size,
            is_generation=is_generation,
            use_scale_overhead_lat=False,
            # MXFP8 uses E8M0 block scale per 32 weights (~0.25 bit/weight overhead);
            # left out of the latency model for parity with the other accelerators.
            # scale_bits=8,
            # group_size=32,
        )

        total_cycle    = acc.calc_cycle()
        compute_energy = acc.calc_compute_energy() / 1e6
        sram_rd_energy = acc.calc_sram_rd_energy() / 1e6
        sram_wr_energy = acc.calc_sram_wr_energy() / 1e6
        dram_energy    = acc.calc_dram_energy() / 1e6
        onchip_energy  = compute_energy + sram_rd_energy + sram_wr_energy
        total_energy   = compute_energy + sram_rd_energy + sram_wr_energy + dram_energy

        print(f'[{idx+1}/{len(model_list)}] Model: {model_name}')
        print(f'  Total Cycle:        {total_cycle[1]:,}')
        print(f'  PE Array Area:      {acc.pe_array_area / 1e6:.6f} mm²')
        print(f'  Weight Buffer:      {acc.w_sram.area:.6f} mm²')
        print(f'  Input Buffer:       {acc.i_sram.area:.6f} mm²')
        print(f'  Total Area:         {(acc.pe_array_area / 1e6 + acc.w_sram.area + acc.i_sram.area):.6f} mm²')
        print(f'  DRAM Energy:        {dram_energy:.2f} uJ')
        print(f'  On-chip Energy:     {onchip_energy:.2f} uJ')
        print(f'  Total Energy:       {total_energy:.2f} uJ')

        op_model = total_ops_list[model_name]
        total_gops = op_model / total_cycle[1]
        total_power = total_energy / total_cycle[1] * 1000000
        total_gops_per_power = total_gops / total_power * 1000
        print(f'  Total Ops:          {total_gops:.2f} GOPS')
        print(f'  Total Power:        {total_power:.2f} mW')
        print(f'  Total GOps per Power: {total_gops_per_power:.2f} GOPS/W')

        print(f'  Energy Delay Product: {total_energy * total_cycle[1]:.2f}')

        # Bottleneck analysis
        acc.print_bottleneck_analysis(show_details=False)

        total_latency_list[idx] = total_cycle[1]
        total_energy_list[idx][0] = round(onchip_energy)
        total_energy_list[idx][1] = round(total_energy)
        print()

    print("\nSummary:")
    print(f'Latency (cycles): {total_latency_list}')
    print(f'Energy [On-chip, Total] (uJ): {total_energy_list}')
