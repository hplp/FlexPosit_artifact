# python test_baseline.py --is_generation > ./log/test_baseline.log
import argparse
from accelerator import Accelerator
from ramulator_dram_sim import get_cache_stats 

model_list = [
    "gpt2-large", "gpt2-xl", "microsoft/phi-2", "facebook/opt-2.7b", "meta-llama/Llama-2-7b-hf",
    "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.1", "deepseek-ai/deepseek-llm-7b-base", "Qwen/Qwen2.5-14B",
]


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
        pe_array_dim = [7, 16]
        # pe_array_dim = [16, 16]
    else:
        pe_array_dim = [16, 16]
    
    total_energy_list = [[0, 0] for _ in model_list]
    total_latency_list = [0 for _ in model_list]

    # Print accelerator configuration
    print("Accelerator: Baseline (FP16)")
    print(f"PE Array Dimension: {pe_array_dim}")
    print(f"Input Precision: 16-bit, Weight Precision: 16-bit")
    print(f"Context Length: 256, Generation Mode: {is_generation}")
    print(f"Models to test: {len(model_list)}")
    print(f"pe_energy: {0.475}")
    print(f"pe_area: {1039.559}")
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
            w_prec=16,
            is_bit_serial=False,
            pe_dp_size=1,
            pe_energy=0.475,
            pe_area=1039.559,
            pe_array_dim=pe_array_dim,
            context_length=args.context_length,
            batch_size=args.batch_size,
            is_generation=is_generation,
            use_scale_overhead_lat=False,
            # scale_bits=8,
            # meta_bits=2,
            # group_size=128,
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

        # Compute total MACs for this model across layers
        total_macs = 0
        for lname in acc.layer_name_list:
            w_dim = acc.weight_dim[lname]
            o_dim = acc.output_dim[lname]
            if w_dim is None or o_dim is None:
                continue
            cout, cin = w_dim
            num_token, _ = o_dim
            total_macs += int(cout) * int(cin) * int(num_token)
        print(f'  Total MACs:         {total_macs:,}')
        
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
    