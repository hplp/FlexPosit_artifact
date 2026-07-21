# python test_flexposit.py --is_generation > ./log/v2_sweep_1108_ddr4_3200ac/test_flexposit.log
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
    parser.add_argument("--is_lossless", action="store_true", help="If enabled, then evaluate")
    parser.add_argument("--pe_x", type=int, default=None, help="PE array dimension X")
    parser.add_argument("--pe_y", type=int, default=None, help="PE array dimension Y")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of sequences decoded in parallel (generation mode)")
    parser.add_argument("--context_length", type=int, default=256, help="Sequence length (in tokens) for the workload")
    args = parser.parse_args()
    is_generation = args.is_generation
    is_lossless = args.is_lossless
    pe_x = args.pe_x
    pe_y = args.pe_y

    # Per-model precision configuration (MixPosit)
    w_prec_list = {
        'gpt2-large': 4.2,
        'gpt2-xl': 4.1,
        # 'facebook/opt-1.3b': 4.4,
        'facebook/opt-2.7b': 4.1,
        'microsoft/phi-2': 4.5,
        '01-ai/Yi-6B': 5.0,
        'meta-llama/Llama-2-7b-hf': 4.6,
        # 'meta-llama/Llama-2-13b-hf': 5.0,
        # 'meta-llama/Meta-Llama-3-8B': 5.0,
        'Qwen/Qwen2.5-7B': 4.4,
        'deepseek-ai/deepseek-llm-7b-base': 4.3,
        'mistralai/Mistral-7B-v0.1': 4.6,
        'Qwen/Qwen2.5-14B': 5.0,
    }

    total_ops_list = {
        'gpt2-large': 65434880*2,
        'gpt2-xl': 82638400*2,
        'microsoft/phi-2': 2650537984*2,
        'facebook/opt-2.7b': 2648162304*2,
        'meta-llama/Llama-2-7b-hf': 6611533824*2,
        # 7B models: ~6.6e9 MACs * 2; 13B: ~1.3e10 * 2 (run roofline/compute_model_macs.py for exact)
        'Qwen/Qwen2.5-7B': 6611533824*2,
        'mistralai/Mistral-7B-v0.1': 6611533824*2,
        'deepseek-ai/deepseek-llm-7b-base': 6611533824*2,
        'Qwen/Qwen2.5-14B': 13223067648*2,
    }

    # w_prec_list = {
    #     'gpt2-large': 5.0,
    #     'gpt2-xl': 5.0,
    #     'facebook/opt-1.3b': 5.0,
    #     'facebook/opt-2.7b': 5.0,
    #     'microsoft/phi-2': 5.0, 
    #     '01-ai/Yi-6B': 5.0, 
    #     'meta-llama/Llama-2-7b-hf': 5.0, 
    #     'meta-llama/Llama-2-13b-hf': 5.0, 
    #     'meta-llama/Meta-Llama-3-8B': 5.0, 
    #     'Qwen/Qwen2.5-7B': 5.0,
    #     'mistralai/Mistral-7B-v0.1': 5.0,
    #     'deepseek-ai/deepseek-llm-7b-base': 5.0,
    # }
    
    print(f"w_prec_list: {w_prec_list}")
    
    if pe_x is not None and pe_y is not None:
        pe_array_dim = [pe_x, pe_y]
    elif is_generation:
        pe_array_dim = [32, 16]
        # pe_array_dim = [64, 16]
    else:
        pe_array_dim = [64, 16]
    
    total_energy_list = [[0, 0] for _ in model_list]
    total_latency_list = [0 for _ in model_list]

    # 打印加速器配置信息
    print("Accelerator: MixPosit (Bit-Serial + Mixed-Precision)")
    print(f"PE Array Dimension: {pe_array_dim}")
    print(f"Input Precision: 16-bit, Weight Precision: Variable (per-model)")
    print(f"PE DP Size: 4, Is MixPosit: True")
    print(f"Context Length: 256, Generation Mode: {is_generation}")
    print(f"Models to test: {len(model_list)}")
    print(f"pe_energy: {0.125}")
    # print(f"pe_energy: {0.071}")
    print(f"pe_area: {214.3}")
    print(f"pe_dp_size: 4")
    print()

    for idx, model_name in enumerate(model_list):
        # 获取该模型的精度
        w_prec = w_prec_list[model_name]
        acc = Accelerator(
            model_name=model_name, 
            i_prec=16,
            w_prec=w_prec,
            is_bit_serial=True,
            pe_dp_size=4,
            pe_energy=0.125,
            # pe_energy=0.071,
            pe_area=214.3,
            pe_array_dim=pe_array_dim,
            context_length=args.context_length,
            batch_size=args.batch_size,
            is_generation=is_generation,
            is_flexposit=True,
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

        w_prec_display = f"{w_prec:.4f}-bit" if isinstance(w_prec, float) else f"{w_prec}-bit"
        print(f'[{idx+1}/{len(model_list)}] Model: {model_name} (W_prec: {w_prec_display})')
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
        
        print(f'  --- Energy Breakdown ---')
        print(f'  PE Compute Energy:  {compute_energy:.2f} uJ')
        print(f'  SRAM Read Energy:   {sram_rd_energy:.2f} uJ')
        print(f'  SRAM Write Energy:  {sram_wr_energy:.2f} uJ')

        # Bottleneck analysis
        acc.print_bottleneck_analysis(show_details=False)
        
        total_latency_list[idx] = total_cycle[1]
        total_energy_list[idx][0] = round(onchip_energy)
        total_energy_list[idx][1] = round(total_energy)
        print()
    
    print("\nSummary:")
    print(f'Latency (cycles): {total_latency_list}')
    print(f'Energy [On-chip, Total] (uJ): {total_energy_list}')
    