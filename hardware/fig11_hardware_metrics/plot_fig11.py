import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

import os
import argparse
import math
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects
from matplotlib import gridspec
from matplotlib.patches import Patch

# ==============================
#  Advanced style settings - consistent with hardware block diagrams
# ==============================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Roboto', 'Source Sans Pro', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.facecolor'] = 'white'  # clean white background for subplots
plt.rcParams['axes.labelcolor'] = '#444444'  # dark gray axis labels
plt.rcParams['xtick.color'] = '#444444'
plt.rcParams['ytick.color'] = '#444444'
plt.rcParams['text.color'] = '#444444'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.color'] = '#CCCCCC'

# ==============================
#  Parse log files
# ==============================
def parse_log_file(log_path):
    """
    Parse one log file and extract latency and energy arrays.
    Returns: (latency_list, on_chip_energy_list, off_chip_energy_list, accelerator_name, energy_unit)
    """
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract accelerator name (first line)
    first_line = content.split('\n')[0]
    acc_name_match = re.search(r'Accelerator:\s+(.+)', first_line)
    accelerator_name = acc_name_match.group(1) if acc_name_match else "Unknown"
    
    # Extract latency array in Summary
    latency_match = re.search(r'Latency \(cycles\):\s*\[([^\]]+)\]', content)
    if not latency_match:
        print(f"Warning: Latency not found in {log_path}")
        return None, None, None, accelerator_name, None
    
    latency_str = latency_match.group(1)
    latency_list = [float(x.strip()) for x in latency_str.split(',')]
    
    # Extract Energy arrays in Summary
    # Format: Energy [On-chip, Total] (mJ/uJ): [[on1, total1], [on2, total2], ...]
    energy_match = re.search(r'Energy \[On-chip, Total\] \((mJ|uJ)\):\s*(\[\[.+?\]\])', content)
    if not energy_match:
        print(f"Warning: Energy not found in {log_path}")
        return latency_list, None, None, accelerator_name, None
    
    energy_unit = energy_match.group(1)
    energy_str = energy_match.group(2)
    # Parse nested list [[on1, total1], [on2, total2], ...]
    # Use precise regex to match each [num, num] pair
    energy_pairs = re.findall(r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]', energy_str)
    on_chip_energy_raw = []
    off_chip_energy_raw = []
    total_energy_raw = []
    
    for on_chip_str, total_str in energy_pairs:
        on_chip = float(on_chip_str)
        total = float(total_str)
        off_chip = total - on_chip
        on_chip_energy_raw.append(on_chip)
        off_chip_energy_raw.append(off_chip)
        total_energy_raw.append(total)
    
    # Convert all energies to Joules for internal calculations
    unit_scale = {'mJ': 1e-3, 'uJ': 1e-6}.get(energy_unit)
    if unit_scale is None:
        raise ValueError(f"Unsupported energy unit '{energy_unit}' in {log_path}")
    
    on_chip_energy = [value * unit_scale for value in on_chip_energy_raw]
    off_chip_energy = [value * unit_scale for value in off_chip_energy_raw]
    total_energy = [value * unit_scale for value in total_energy_raw]
    
    # Print parsing result
    print(f"\n{'='*70}")
    print(f"📄 File: {os.path.basename(log_path)}")
    print(f"🏷️  Accelerator: {accelerator_name}")
    print(f"⏱️  Latency (cycles): {latency_list}")
    print(f"⚡ On-chip Energy ({energy_unit}): {on_chip_energy_raw}")
    print(f"💾 Off-chip Energy ({energy_unit}): {off_chip_energy_raw}")
    print(f"📊 Total Energy ({energy_unit}): {total_energy_raw}")
    print(f"{'='*70}")
    
    # Extract Total GOps per Power for each model (throughput efficiency)
    gops_per_power_matches = re.findall(r'Total GOps per Power:\s*([\d\.]+)', content)
    if not gops_per_power_matches:
        print(f"Warning: GOps per Power not found in {log_path}")
        gops_per_power = None
    else:
        gops_per_power = [float(value) for value in gops_per_power_matches]
        if len(gops_per_power) != len(latency_list):
            print(f"Warning: GOps per Power count mismatch in {log_path} (expected {len(latency_list)}, got {len(gops_per_power)})")
            gops_per_power = gops_per_power[:len(latency_list)]

    return latency_list, on_chip_energy, off_chip_energy, gops_per_power, accelerator_name, energy_unit


def load_all_logs(log_dir):
    """
    Load all log files
    Return: dict with keys as accelerator names
    """
    log_files = {
        'Baseline': 'test_baseline.log',
        'FlexPosit': 'test_flexposit.log',
        'BitMod': 'test_bitmod.log',
        'Olive': 'test_olive.log',
        'FP16-MXFP8': 'test_fp16_mxfp8.log',  # optional 5th accelerator (loaded only if log present)
    }
    
    data = {}
    
    for acc_key, log_file in log_files.items():
        log_path = os.path.join(log_dir, log_file)
        if os.path.exists(log_path):
            latency, on_chip, off_chip, gops_per_power, acc_name, energy_unit = parse_log_file(log_path)
            if latency is not None and on_chip is not None and off_chip is not None and gops_per_power is not None:
                if len(gops_per_power) != len(latency):
                    print(f"⚠️  Skip loading GOps/W for {log_file}: length mismatch")
                    continue
                data[acc_key] = {
                    'latency': np.array(latency),
                    'on_chip_energy': np.array(on_chip),
                    'off_chip_energy': np.array(off_chip),
                    'gops_per_power': np.array(gops_per_power),
                    'full_name': acc_name,
                    'energy_unit': energy_unit,
                }
                unit_display = energy_unit if energy_unit is not None else "unknown unit"
                print(f"✅ Loaded: {log_file} ({acc_name}) – energy parsed as {unit_display}")
            else:
                print(f"⚠️  Skip loading energy data for {log_file} (missing entries)")
        else:
            print(f"⚠️  File not found: {log_path}")
    
    return data


def normalize_data(data, baseline_key='Baseline'):
    """
    Normalize all data to baseline (per-model normalization)
    Each model independently normalized such that baseline total energy = 1.0
    """
    if baseline_key not in data:
        print(f"Warning: baseline missing, use the first accelerator as baseline")
        baseline_key = list(data.keys())[0]
    
    baseline_latency = np.array(data[baseline_key]['latency'])
    baseline_on_chip = np.array(data[baseline_key]['on_chip_energy'])
    baseline_off_chip = np.array(data[baseline_key]['off_chip_energy'])
    baseline_total_energy = baseline_on_chip + baseline_off_chip  # 每个模型的baseline总能量
    baseline_gops_per_power = np.array(data[baseline_key]['gops_per_power'])

    if baseline_gops_per_power.size == 0:
        raise ValueError(f"Baseline accelerator '{baseline_key}' has no GOps/W data for normalization")
    if np.any(baseline_gops_per_power == 0):
        raise ValueError(f"Baseline accelerator '{baseline_key}' contains zero GOps/W value, cannot normalize")
    
    # Compute baseline EDP
    baseline_edp = baseline_latency * baseline_total_energy
    
    normalized_data = {}
    
    print("\n" + "="*70)
    print("🔄 Start normalizing data (relative to Baseline, per-model)")
    print("="*70)
    
    model_names = [
        "GPT2-L", "GPT2-XL", "Phi-2", "OPT-2.7B", "Llama2-7B",
        "Qwen2.5-7B", "Mistral-7B", "DeepSeek-7B", "Qwen2.5-14B"
    ]
    
    for acc_key, acc_data in data.items():
        acc_on_chip = np.array(acc_data['on_chip_energy'])
        acc_off_chip = np.array(acc_data['off_chip_energy'])
        acc_latency = np.array(acc_data['latency'])
        acc_total_energy = acc_on_chip + acc_off_chip
        acc_gops_per_power = np.array(acc_data['gops_per_power'])
        
        # Latency normalization: per-model
        norm_latency = acc_latency / baseline_latency
        
        # Energy normalization: per-model (element-wise division)
        norm_on_chip = acc_on_chip / baseline_total_energy
        norm_off_chip = acc_off_chip / baseline_total_energy
        
        # EDP normalization: EDP = Latency × Total_Energy
        acc_edp = acc_latency * acc_total_energy
        norm_edp = acc_edp / baseline_edp
        if np.any(acc_gops_per_power == 0):
            print(f"Warning: {acc_key} contains zero GOps/W value, normalization may be skewed")
        norm_gops_per_power = acc_gops_per_power / baseline_gops_per_power
        
        normalized_data[acc_key] = {
            'norm_latency': norm_latency,
            'norm_on_chip': norm_on_chip,
            'norm_off_chip': norm_off_chip,
            'norm_edp': norm_edp,
            'norm_gops_per_power': norm_gops_per_power,
            'full_name': acc_data['full_name']
        }
        
        # Print normalized results (per-model)
        print(f"\n📊 {acc_key} ({acc_data['full_name']})")
        for i, model_name in enumerate(model_names):
            total_norm = norm_on_chip[i] + norm_off_chip[i]
            print(f"   {model_name:12s}: Latency={norm_latency[i]:.3f}, "
                  f"Energy={total_norm:.3f}, EDP={norm_edp[i]:.3f}, GOP/W={norm_gops_per_power[i]:.3f}")
    
    print("\n" + "="*70)
    
    return normalized_data


# ==============================
#  Plotting (with EDP)
# ==============================
def plot_metrics_with_edp(normalized_data, output_prefix='auto_hw_metrics_edp'):
    """
    Plot latency, energy, EDP, and normalized GOps/W (four subplots)
    """
    # 模型名称（9个模型）+ 平均值
    models = [
        "GPT2-L", "GPT2-XL", "Phi-2",
        "OPT-2.7B", "Llama2-7B", "Qwen2.5-7B",
        "Mistral-7B", "DeepSeek-7B", "Qwen2.5-14B", "Average"
    ]
    
    # Accelerator order (can be adjusted)
    accelerator_order = ['FlexPosit', 'BitMod', 'Olive', 'FP16-MXFP8', 'Baseline']
    accelerators = [acc for acc in accelerator_order if acc in normalized_data]
    
    # Color scheme consistent with hardware figures (teal, orange, gray)
    acc_colors = {
        # FlexPosit：颜色与 ppl_vs_edp 图例中的圆圈完全一致
        # ppl_vs_edp 中使用的是 RGB ≈ (0.45, 0.72, 0.45)，对应十六进制约为 #73B873
        'FlexPosit': '#73B873',
        'BitMod': '#F4A261',      # 暖橙色（与Bit-serial路径呼应）
        'Olive': '#457B9D',       # 深青蓝（稳重对比）
        'FP16-MXFP8': '#8E7CC3',  # 紫色（FP16激活+FP8权重，区别于其他）
        'Baseline': '#BDBDBD'     # 浅灰（代表参考基线）
    }
    
    # Prepare matrices (including averages)
    n_models = len(models)  # 包含Average
    n_accs = len(accelerators)
    n_actual_models = n_models - 1  # actual models (exclude Average)
    
    norm_cycle = np.zeros((n_models, n_accs))
    norm_energy_on = np.zeros((n_models, n_accs))
    norm_energy_off = np.zeros((n_models, n_accs))
    norm_edp = np.zeros((n_models, n_accs))
    norm_gops_per_power = np.zeros((n_models, n_accs))
    
    for j, acc in enumerate(accelerators):
        # 前n_actual_models行是实际数据
        norm_cycle[:n_actual_models, j] = normalized_data[acc]['norm_latency']
        norm_energy_on[:n_actual_models, j] = normalized_data[acc]['norm_on_chip']
        norm_energy_off[:n_actual_models, j] = normalized_data[acc]['norm_off_chip']
        norm_edp[:n_actual_models, j] = normalized_data[acc]['norm_edp']
        norm_gops_per_power[:n_actual_models, j] = normalized_data[acc]['norm_gops_per_power']
        
        # 最后一行是平均值
        norm_cycle[n_actual_models, j] = np.mean(normalized_data[acc]['norm_latency'])
        norm_energy_on[n_actual_models, j] = np.mean(normalized_data[acc]['norm_on_chip'])
        norm_energy_off[n_actual_models, j] = np.mean(normalized_data[acc]['norm_off_chip'])
        norm_edp[n_actual_models, j] = np.mean(normalized_data[acc]['norm_edp'])
        norm_gops_per_power[n_actual_models, j] = np.mean(normalized_data[acc]['norm_gops_per_power'])
    
    # ==============================
    #  Plotting
    # ==============================
    # Use hatch to distinguish On-chip/Off-chip (instead of color)
    energy_hatches = ['///', '\\\\']  # On-Chip用斜线，Off-Chip用反斜线
    bar_spacing = 1.33  # 控制同组柱子间距，保持紧凑
    # 柱宽随加速器数量自适应：4个用0.34，5个收窄以免组间重叠
    bar_width = 0.34 if len(accelerators) <= 4 else 0.27
    x_base = np.arange(len(models)) * 2.1  # 增大不同模型之间的间距
    
    # Create 4 subplots in 2x2 layout
    fig = plt.figure(figsize=(24, 7.0))  # 增加宽度以适应9个模型
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.48, wspace=0.18)
    axes = [
        plt.subplot(gs[0, 0]),  # (a)
        plt.subplot(gs[0, 1]),  # (b)
        plt.subplot(gs[1, 0]),  # (c)
        plt.subplot(gs[1, 1])   # (d)
    ]

    subplot_titles = [
        "(a) Normalized Latency",
        "(b) Normalized Energy",
        "(c) Normalized EDP",
        "(d) Normalized Energy Efficiency"
    ]

    offset_center = (len(accelerators) - 1) / 2 if accelerators else 0
    
    # -------------------------------------------------------
    # (1) Latency plot
    for j, acc in enumerate(accelerators):
        x = x_base + (j - offset_center) * bar_width * bar_spacing
        bars = axes[0].bar(x, norm_cycle[:, j], bar_width, label=acc, 
                           color=acc_colors.get(acc, '#999999'), 
                           edgecolor='black', linewidth=0.8,
                           alpha=0.90, zorder=3)
        
        # Add labels
        for i, (bar, val) in enumerate(zip(bars, norm_cycle[:, j])):
            if val > 0.05:  # 只显示足够大的值
                lbl = f"{val:.2f}" if math.isclose(val, 1.0, rel_tol=1e-9, abs_tol=1e-9) else f"{val:.3f}"
                axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.05, 
                            lbl, ha='center', va='bottom', 
                            fontsize=8, fontweight='bold', 
                            color=acc_colors.get(acc, '#999999'),
                            rotation=90)
    
    axes[0].set_ylabel("Normalized Latency", fontweight='bold')
    max_latency = norm_cycle.max()
    axes[0].set_ylim(0, max(1.1, max_latency * 1.15))
    axes[0].set_xticks(x_base)
    axes[0].set_xticklabels(models, rotation=25, ha='center', fontweight='bold')
    
    # Highlight Average tick (teal)
    labels = axes[0].get_xticklabels()
    labels[-1].set_color('#4FB0A9')  # 青绿主色强调
    labels[-1].set_weight('extra bold')
    
    # X label
    
    # Legend: transparent background
    axes[0].legend(accelerators, ncol=len(accelerators), bbox_to_anchor=(0.5, 1.24),
                   loc='upper center', frameon=True, fancybox=False, shadow=False,
                   framealpha=0.9, edgecolor='#CCCCCC',
                   handletextpad=0.25, columnspacing=0.65)
    
    # -------------------------------------------------------
    # (2) Energy (stacked)
    for j, acc in enumerate(accelerators):
        x = x_base + (j - offset_center) * bar_width * bar_spacing
        
        # On-chip部分（底部）- 使用加速器颜色 + 黑色斜线图案（黑色边框）
        bars_on = axes[1].bar(x, norm_energy_on[:, j], bar_width,
                              color=acc_colors.get(acc, '#999999'), 
                              edgecolor='black', 
                              linewidth=0.8, alpha=0.75, 
                              hatch=energy_hatches[0], zorder=3)
        
        # Off-chip部分（堆叠在上面）- 使用加速器颜色 + 黑色反斜线图案（黑色边框）
        bars_off = axes[1].bar(x, norm_energy_off[:, j], bar_width,
                               bottom=norm_energy_on[:, j],
                               color=acc_colors.get(acc, '#999999'), 
                               edgecolor='black', 
                               linewidth=0.8, alpha=0.45,
                               hatch=energy_hatches[1], zorder=3)
        
        # Add labels - show total energy only
        for i, (bar_on, bar_off, val_on, val_off) in enumerate(zip(
            bars_on, bars_off, norm_energy_on[:, j], norm_energy_off[:, j])):
            total = val_on + val_off
            
            # Total energy label
            total_energy = val_on + val_off
            lbl_total = f"{total_energy:.2f}" if math.isclose(total_energy, 1.0, rel_tol=1e-9, abs_tol=1e-9) else f"{total_energy:.3f}"
            txt_total = axes[1].text(bar_off.get_x() + bar_off.get_width()/2, 
                       total_energy + 0.05, 
                       lbl_total, ha='center', va='bottom',
                       fontsize=7.5, fontweight='bold', 
                       color='#000000',
                       rotation=90)
    
    axes[1].set_ylabel("Normalized Energy", fontweight='bold')
    max_energy = (norm_energy_on + norm_energy_off).max()
    axes[1].set_ylim(0, max(1.15, max_energy * 1.15))
    axes[1].set_xticks(x_base)
    axes[1].set_xticklabels(models, rotation=25, ha='center', fontweight='bold')
    
    # Highlight Average tick (teal)
    labels = axes[1].get_xticklabels()
    labels[-1].set_color('#4FB0A9')  # 青绿主色强调
    labels[-1].set_weight('extra bold')
    
    # Title
    
    # Custom legend - show hatch patterns
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', hatch=energy_hatches[0], 
              alpha=0.75, label='On-Chip Energy'),
        Patch(facecolor='gray', edgecolor='black', hatch=energy_hatches[1], 
              alpha=0.45, label='Off-Chip Energy')
    ]
    axes[1].legend(handles=legend_elements, ncol=2, bbox_to_anchor=(0.5, 1.24),
                   loc='upper center', frameon=True, fancybox=False, shadow=False,
                   framealpha=0.9, edgecolor='#CCCCCC',
                   handletextpad=0.25, columnspacing=0.65)
    
    # -------------------------------------------------------
    # (3) EDP plot
    for j, acc in enumerate(accelerators):
        x = x_base + (j - offset_center) * bar_width * bar_spacing
        bars = axes[2].bar(x, norm_edp[:, j], bar_width, label=acc,
                           color=acc_colors.get(acc, '#999999'), 
                           edgecolor='black', linewidth=0.8,
                           alpha=0.90, zorder=3)
        
        # Add labels
        for i, (bar, val) in enumerate(zip(bars, norm_edp[:, j])):
            if val > 0.05:  # 只显示足够大的值
                lbl = f"{val:.2f}" if math.isclose(val, 1.0, rel_tol=1e-9, abs_tol=1e-9) else f"{val:.3f}"
                axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.05, 
                            lbl, ha='center', va='bottom', 
                            fontsize=8, fontweight='bold', 
                            color=acc_colors.get(acc, '#999999'),
                            rotation=90)
    
    axes[2].set_ylabel("Normalized EDP", fontweight='bold')
    max_edp = norm_edp.max()
    axes[2].set_ylim(0, max(1.1, max_edp * 1.15))
    axes[2].set_xticks(x_base)
    axes[2].set_xticklabels(models, rotation=25, ha='center', fontweight='bold')
    
    # Highlight Average tick (teal)
    labels = axes[2].get_xticklabels()
    labels[-1].set_color('#4FB0A9')  # 青绿主色强调
    labels[-1].set_weight('extra bold')
    
    # X label
    
    # Legend - removed for subplot (c)
    # axes[2].legend(accelerators, ncol=len(accelerators), bbox_to_anchor=(0.5, 1.24),
    #                loc='upper center', frameon=True, fancybox=False, shadow=False,
    #                framealpha=0.9, edgecolor='#CCCCCC',
    #                handletextpad=0.25, columnspacing=0.65)

    # -------------------------------------------------------
    # (4) Normalized GOps/W plot
    for j, acc in enumerate(accelerators):
        x = x_base + (j - offset_center) * bar_width * bar_spacing
        bars = axes[3].bar(x, norm_gops_per_power[:, j], bar_width, label=acc,
                           color=acc_colors.get(acc, '#999999'),
                           edgecolor='black', linewidth=0.8,
                           alpha=0.90, zorder=3)

        # Add labels
        for bar, val in zip(bars, norm_gops_per_power[:, j]):
            if val > 0.05:
                lbl = f"{val:.2f}" if math.isclose(val, 1.0, rel_tol=1e-9, abs_tol=1e-9) else f"{val:.3f}"
                axes[3].text(bar.get_x() + bar.get_width()/2, val + 0.05,
                             lbl, ha='center', va='bottom',
                             fontsize=8, fontweight='bold',
                             color=acc_colors.get(acc, '#999999'),
                             rotation=90)

    axes[3].set_ylabel("Normalized Energy Efficiency", fontweight='bold')
    max_gops_per_power = norm_gops_per_power.max()
    axes[3].set_ylim(0, max(1.1, max_gops_per_power * 1.15))
    axes[3].set_xticks(x_base)
    axes[3].set_xticklabels(models, rotation=25, ha='center', fontweight='bold')

    # Highlight Average tick (teal)
    labels = axes[3].get_xticklabels()
    labels[-1].set_color('#4FB0A9')
    labels[-1].set_weight('extra bold')


    # Legend - removed for subplot (d)
    # axes[3].legend(accelerators, ncol=len(accelerators), bbox_to_anchor=(0.5, 1.24),
    #                loc='upper center', frameon=True, fancybox=False, shadow=False,
    #                framealpha=0.9, edgecolor='#CCCCCC',
    #                handletextpad=0.25, columnspacing=0.65)
    
    # -------------------------------------------------------
    # Styling
    for idx, ax in enumerate(axes):
        # Add subplot title below the plot (keep horizontally centered)
        ax.text(0.5, -0.32, subplot_titles[idx], 
                transform=ax.transAxes, ha='center', va='top', 
                fontsize=13, fontweight='bold', rotation=0)
        
        # Add background span for Average section
        avg_pos = x_base[-1]  # Average的位置
        half_group_width = (len(accelerators) * bar_width * bar_spacing) / 2
        ax.axvspan(avg_pos - half_group_width - 0.1, avg_pos + half_group_width + 0.1, 
                   color='#E8E8E8', alpha=0.32, zorder=0)
        
        # Grid
        ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.6, zorder=0, color='#CCCCCC')
        ax.set_axisbelow(True)
        
        # Vertical separators between models (exclude before Average)
        for i in range(1, len(models) - 1):  # 只到倒数第二个位置
            ax.axvline(x_base[i] - x_base[1]/2, color='#DDDDDD', linestyle='-', linewidth=1.2, alpha=0.5, zorder=1)
        
        # Axes spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_color('#444444')
        ax.spines['bottom'].set_color('#444444')
    
    # No suptitle, keep compact
    plt.tight_layout()
    
    # Save figures
    png_file = f'{output_prefix}.png'
    pdf_file = f'{output_prefix}.pdf'
    plt.savefig(png_file, dpi=400, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print("\n" + "="*60)
    print("✅ Figures saved:")
    print(f"   📊 PNG格式 (400 DPI): {png_file}")
    print(f"   📄 PDF矢量图: {pdf_file}")
    print("="*60)


# ==============================
#  Main
# ==============================
if __name__ == '__main__':
    # Self-contained artifact: read the 5 accelerator logs from ./data and write
    # fig11_hardware_metrics.{png,pdf} next to this script. Only needs matplotlib + numpy.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Reproduce Figure 11 (normalized HW metrics) from bundled logs')
    parser.add_argument('--data-dir', default=os.path.join(script_dir, 'data'),
                        help='Directory holding the 5 test_*.log files (default: ./data)')
    args = parser.parse_args()
    log_dir = args.data_dir

    print("="*60)
    print("Reproducing Figure 11: normalized latency / energy / EDP / energy-efficiency")
    print("="*60)
    print(f"Data dir: {log_dir}\n")

    data = load_all_logs(log_dir)
    if not data:
        print("Error: no data loaded")
        exit(1)
    print(f"\nLoaded {len(data)} accelerators\n")

    normalized_data = normalize_data(data)
    output_prefix = os.path.join(script_dir, 'fig11_hardware_metrics')
    plot_metrics_with_edp(normalized_data, output_prefix)
    
    print("\n✨ Done!")

