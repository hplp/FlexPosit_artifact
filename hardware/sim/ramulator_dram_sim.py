import math
import os
import subprocess
import tempfile
import textwrap

# =========================
# Configuration constants
# =========================
ELEM_SIZE_B   = 1           # Size of one element in bytes (keep 1 if weights are byte-packed)
SCALE_SIZE_B  = 1          # Typical scale size (e.g., FP32). If you really use 1B scales, set to 1.
LINE_SIZE_B   = 64          # Access granularity / cache line (64B is natural for DDR4 BL8 at system level)
SCALE_BASE    = 0x1000_0000 # Base address for scales
DATA_BASE     = 0x2000_0000 # Base address for weights
INPUT_BASE    = 0x2100_0000 # Base address for inputs (separate from weights to avoid accidental co-location)
OUTPUT_BASE   = 0x3000_0000 # Base address for outputs
ROW_SPREAD_EXTRA = 0x0      # Optional extra offset to emulate row/bank spreading for scales

# =========================
# Caching mechanism
# =========================
ENABLE_RAMULATOR_CACHE = True  # Set to False to disable caching (for debugging)
_ramulator_cache = {}
_cache_stats = {'hits': 0, 'misses': 0}

def get_cache_stats():
    """Return cache hit/miss statistics."""
    total = _cache_stats['hits'] + _cache_stats['misses']
    hit_rate = _cache_stats['hits'] / total * 100 if total > 0 else 0
    return {
        'hits': _cache_stats['hits'],
        'misses': _cache_stats['misses'],
        'total': total,
        'hit_rate': hit_rate
    }

def clear_cache():
    """Clear the Ramulator result cache."""
    global _ramulator_cache, _cache_stats
    _ramulator_cache = {}
    _cache_stats = {'hits': 0, 'misses': 0}

# =========================
# Utilities
# =========================
def ceil_div(a: int, b: int) -> int:
    """Ceiling division for positive integers."""
    return (a + b - 1) // b

# ===========================================================
# Trace generators + runners (produce total cycles, not avg)
# ===========================================================
def ramulator_weight_read_group_wise(workload_bytes: int, group_size: int = 128, col: int = 1) -> tuple[int, float]:
    """
    Group-wise weight read with per-column scale factors.
    For each 'big group' (col * group_size elements):
      1) Read contiguous weights of that group.
      2) Read 'col' scale factors (one per column).
    Returns: (total_cycles, total_energy_pJ)
    - total_cycles: TOTAL cycles of this trace (not average latency)
    - total_energy_pJ: Total energy in picojoules (pJ)
    """
    # Check cache first
    if ENABLE_RAMULATOR_CACHE:
        cache_key = ('weight_group_wise', workload_bytes, group_size, col)
        if cache_key in _ramulator_cache:
            _cache_stats['hits'] += 1
            return _ramulator_cache[cache_key]
        _cache_stats['misses'] += 1
    
    # Original implementation
    total_elems = ceil_div(workload_bytes, ELEM_SIZE_B)
    elems_per_big_group = group_size * col
    num_big_groups = ceil_div(total_elems, elems_per_big_group)

    def gen_lines():
        for g in range(num_big_groups):
            # Actual remaining elements for this group (last group may be partial)
            remaining = total_elems - g * elems_per_big_group
            this_elems = min(remaining, elems_per_big_group)
            this_bytes = this_elems * ELEM_SIZE_B

            # 1) Read contiguous weights for this group
            base = DATA_BASE + g * (elems_per_big_group * ELEM_SIZE_B)  # stride by full-group span
            n_lines = ceil_div(this_bytes, LINE_SIZE_B)
            for i in range(n_lines):
                yield f"R 0x{base + i * LINE_SIZE_B:x}"

            # 2) Read 'col' scale factors (one per column)
            for c in range(col):
                scale_addr = SCALE_BASE + g * col * SCALE_SIZE_B + c * SCALE_SIZE_B + ROW_SPREAD_EXTRA
                yield f"R 0x{scale_addr:x}"

            # #### Worse Group-wise
            # SCALE_ROW_STRIDE = 0x2000  # 8KB = typical DRAM row size
            # for c in range(col):
            #     scale_addr = SCALE_BASE + g * col * SCALE_ROW_STRIDE + c * SCALE_ROW_STRIDE
            #     yield f"R 0x{scale_addr:x}"
            # ####

    result = run_ramulator_with_trace(gen_lines())
    
    # Store in cache
    if ENABLE_RAMULATOR_CACHE:
        _ramulator_cache[cache_key] = result
    
    return result

def ramulator_weight_read_not_group_wise(workload_bytes: int) -> tuple[int, float]:
    """
    Non-group-wise weight read (no scales). Reads are contiguous.
    Returns: (total_cycles, total_energy_pJ)
    - total_cycles: TOTAL cycles of this trace
    - total_energy_pJ: Total energy in picojoules (pJ)
    """
    # Check cache first
    if ENABLE_RAMULATOR_CACHE:
        cache_key = ('weight_not_group_wise', workload_bytes)
        if cache_key in _ramulator_cache:
            _cache_stats['hits'] += 1
            return _ramulator_cache[cache_key]
        _cache_stats['misses'] += 1
    
    # Original implementation
    n_lines = ceil_div(workload_bytes, LINE_SIZE_B)
    result = run_ramulator_with_trace(f"R 0x{DATA_BASE + i * LINE_SIZE_B:x}" for i in range(n_lines))
    
    # Store in cache
    if ENABLE_RAMULATOR_CACHE:
        _ramulator_cache[cache_key] = result
    
    return result

def ramulator_input_read(workload_bytes: int, group_size: int = 128) -> tuple[int, float]:
    """
    Input read (contiguous). 'group_size' kept for signature compatibility (unused).
    Returns: (total_cycles, total_energy_pJ)
    - total_cycles: TOTAL cycles of this trace
    - total_energy_pJ: Total energy in picojoules (pJ)
    """
    # Check cache first
    if ENABLE_RAMULATOR_CACHE:
        cache_key = ('input_read', workload_bytes)
        if cache_key in _ramulator_cache:
            _cache_stats['hits'] += 1
            return _ramulator_cache[cache_key]
        _cache_stats['misses'] += 1
    
    # Original implementation
    n_lines = ceil_div(workload_bytes, LINE_SIZE_B)
    result = run_ramulator_with_trace(f"R 0x{INPUT_BASE + i * LINE_SIZE_B:x}" for i in range(n_lines))
    
    # Store in cache
    if ENABLE_RAMULATOR_CACHE:
        _ramulator_cache[cache_key] = result
    
    return result

def ramulator_output_write(workload_bytes: int, group_size: int = 128) -> tuple[int, float]:
    """
    Output write (contiguous). 'group_size' kept for signature compatibility (unused).
    Returns: (total_cycles, total_energy_pJ)
    - total_cycles: TOTAL cycles of this trace
    - total_energy_pJ: Total energy in picojoules (pJ)
    """
    # Check cache first
    if ENABLE_RAMULATOR_CACHE:
        cache_key = ('output_write', workload_bytes)
        if cache_key in _ramulator_cache:
            _cache_stats['hits'] += 1
            return _ramulator_cache[cache_key]
        _cache_stats['misses'] += 1
    
    # Original implementation
    n_lines = ceil_div(workload_bytes, LINE_SIZE_B)
    result = run_ramulator_with_trace(f"W 0x{OUTPUT_BASE + i * LINE_SIZE_B:x}" for i in range(n_lines))
    
    # Store in cache
    if ENABLE_RAMULATOR_CACHE:
        _ramulator_cache[cache_key] = result
    
    return result

# Backward compatibility alias
def ramulator_weight_read(workload_bytes: int, group_size: int = 128) -> tuple[int, float]:
    """Backward-compatible wrapper: defaults to group-wise reading.
    Returns: (total_cycles, total_energy_pJ)"""
    return ramulator_weight_read_group_wise(workload_bytes, group_size)

# ===========================================
# File I/O + LD/ST conversion + Ramulator run
# ===========================================
def run_ramulator_with_trace(rw_trace_lines) -> tuple[int, float]:
    """
    Accepts an iterator/generator (or list) of 'R 0x...' / 'W 0x...' lines (no trailing newline).
    Writes an RW trace file, converts to LD/ST, and runs Ramulator.
    Returns: (total_cycles, total_energy_pJ)
    - total_cycles: TOTAL cycles of the whole trace (prefers `memory_system_cycles`)
    - total_energy_pJ: Total energy in picojoules (pJ) from Ramulator, 0 if not available
    """
    total_reqs = 0
    with tempfile.NamedTemporaryFile(mode='w', suffix='.trace', delete=False) as f:
        for line in rw_trace_lines:
            f.write(line + '\n')
            if line and line[0] in ('R', 'W'):
                total_reqs += 1
        rw_trace_file = f.name

    try:
        ldst_trace_file = convert_rw_to_ldst(rw_trace_file)
        return run_ramulator_simulation(ldst_trace_file, total_requests_hint=total_reqs)
    finally:
        try:
            os.unlink(rw_trace_file)
        except FileNotFoundError:
            pass
        try:
            if 'ldst_trace_file' in locals():
                os.unlink(ldst_trace_file)
        except FileNotFoundError:
            pass

def convert_rw_to_ldst(rw_trace_file: str) -> str:
    """
    Convert RW format to Ramulator Load/Store (LD/ST) format.
    - 'R' → 'LD'
    - 'W' → 'ST'
    Keeps hex addresses as-is; supports decimal as fallback.
    """
    def convert_line(line: str, decimal: bool = False, strict: bool = False, lineno: int = 0, src: str = ""):
        s = line.strip()
        if not s or s.startswith(("#", ";", "//")):
            return None
        parts = s.split()
        if len(parts) < 2:
            if strict:
                raise ValueError(f"[{src}:{lineno}] invalid line: {line.strip()}")
            return None
        op, addr = parts[0].upper(), parts[1]
        if op in ("R", "LD", "L", "LOAD"):
            op2 = "LD"
        elif op in ("W", "ST", "S", "STORE"):
            op2 = "ST"
        else:
            if strict:
                raise ValueError(f"[{src}:{lineno}] unknown op '{op}'")
            return None
        try:
            _ = int(addr, 0)  # parse hex (0x...) or decimal
        except Exception:
            if strict:
                raise ValueError(f"[{src}:{lineno}] bad addr '{addr}'")
            return None
        addr_out = addr if addr.startswith(("0x", "0X")) else addr  # keep existing representation
        return f"{op2} {addr_out}\n"

    ldst_trace_file = rw_trace_file.replace('.trace', '.ldst.trace')
    with open(rw_trace_file) as fin, open(ldst_trace_file, "w") as fout:
        for i, line in enumerate(fin, start=1):
            out = convert_line(line, decimal=False, strict=False, lineno=i, src=os.path.basename(rw_trace_file))
            if out is not None:
                fout.write(out)
    return ldst_trace_file

def run_ramulator_simulation(ldst_trace_file: str, total_requests_hint: int | None = None) -> tuple[int, float]:
    """
    Run Ramulator with a temporary YAML config pointing to the LD/ST trace.
    Returns: (total_cycles, total_energy_pJ)
    
    Priority of cycle return value:
      1) memory_system_cycles (TOTAL cycles for the entire trace)
      2) (average latency) * (number of requests)  [uses Ramulator counters if present, else falls back to total_requests_hint]
      3) 1 (hard fallback)
    
    Energy is returned in picojoules (pJ). If energy data is not available from Ramulator, returns 0.
    
    DRAM timing preset is fixed to DDR4_3200AC.
    """

    
#     config_content = f"""Frontend:
#   impl: LoadStoreTrace
#   path: {ldst_trace_file}
#   clock_ratio: 1

# Translation:
#   impl: IdentityTranslation
#   max_addr: 1000000000

# MemorySystem:
#   impl: GenericDRAM
#   clock_ratio: 1

#   DRAM:
#     impl: DDR5
#     org:
#       preset: DDR5_8Gb_x8   
#       channel: 2
#       rank: 1
#     timing:
#       preset: DDR5_3200C    
#     drampower_enable: true
#     voltage:
#       preset: Default
#     current:
#       preset: Default
#     RFM:                        
#       BRC: 2 

#   Controller:
#     impl: Generic
#     Scheduler:
#       impl: FRFCFS
#     RefreshManager:
#       impl: AllBank
#     RowPolicy:
#       impl: OpenRowPolicy
#       cap: 4
#     plugins:

#   AddrMapper:
#     impl: RoBaRaCoCh
# """


    config_content = f"""Frontend:
  impl: LoadStoreTrace
  path: {ldst_trace_file}
  clock_ratio: 1

Translation:
  impl: IdentityTranslation
  max_addr: 1000000000

MemorySystem:
  impl: GenericDRAM
  clock_ratio: 1

  DRAM:
    impl: DDR4
    org:
      preset: DDR4_8Gb_x8
      channel: 2
      rank: 1
    timing:
      preset: DDR4_3200AC
    drampower_enable: true
    voltage:
      preset: Default
    current:
      preset: Default

  Controller:
    impl: Generic
    Scheduler:
      impl: FRFCFS
    RefreshManager:
      impl: AllBank
    RowPolicy:
      impl: OpenRowPolicy
      cap: 4
    plugins:

  AddrMapper:
    impl: RoBaRaCoCh
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_config_file = f.name

    try:
        # Resolve Ramulator binary path (self-contained artifact layout):
        # - Prefer environment variable RAMULATOR_BIN
        # - Else use the bundled binary at ../ramulator/ramulator2 relative to this file (sim/)
        here = os.path.dirname(os.path.abspath(__file__))
        default_bin = os.path.normpath(os.path.join(here, os.pardir, 'ramulator', 'ramulator2'))
        ramulator_path = os.environ.get("RAMULATOR_BIN", default_bin)

        if not os.path.exists(ramulator_path):
            raise RuntimeError(f"Ramulator 2.0 executable not found: {ramulator_path}")

        # Ensure the co-located libramulator.so is found regardless of the ELF's baked-in
        # RUNPATH (LD_LIBRARY_PATH is searched before DT_RUNPATH).
        env = os.environ.copy()
        bin_dir = os.path.dirname(os.path.abspath(ramulator_path))
        env["LD_LIBRARY_PATH"] = bin_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")

        result = subprocess.run(
            [ramulator_path, '-f', temp_config_file],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Ramulator execution failed: {result.stderr}")

        # ---- Parse stdout ----
        total_cycles = None
        avg_rd_lat = None
        avg_wr_lat = None
        rd_reqs = None
        wr_reqs = None
        total_energy = None  # parse total energy

        for line in result.stdout.splitlines():
            if 'memory_system_cycles' in line and ':' in line:
                try:
                    total_cycles = int(line.split(':', 1)[1].strip())
                except:
                    pass
            elif 'avg_read_latency_0' in line and ':' in line:
                try:
                    v = float(line.split(':', 1)[1].strip())
                    if v == v and v > 0:  # filter NaN
                        avg_rd_lat = v
                except:
                    pass
            elif 'avg_write_latency_0' in line and ':' in line:
                try:
                    v = float(line.split(':', 1)[1].strip())
                    if v == v and v > 0:
                        avg_wr_lat = v
                except:
                    pass
            elif 'total_num_read_requests' in line and ':' in line:
                try:
                    rd_reqs = int(line.split(':', 1)[1].strip())
                except:
                    pass
            elif 'total_num_write_requests' in line and ':' in line:
                try:
                    wr_reqs = int(line.split(':', 1)[1].strip())
                except:
                    pass
            # Parse energy stats from Ramulator2
            elif 'total_energy' in line and ':' in line and 'rank' not in line:
                try:
                    # Energy unit is nJ; convert to pJ for the rest of the pipeline
                    energy_nJ = float(line.split(':', 1)[1].strip())
                    total_energy = energy_nJ * 1e3  # 1 nJ = 1000 pJ
                except:
                    pass

        # 1) Prefer total cycles (this is the total completion time for the trace)
        final_cycles = None
        if isinstance(total_cycles, int) and total_cycles > 0:
            final_cycles = total_cycles
        else:
            # 2) Fallback: (average latency) * (#requests)
            total_reqs = None
            if (rd_reqs is not None) or (wr_reqs is not None):
                total_reqs = (rd_reqs or 0) + (wr_reqs or 0)
            elif total_requests_hint:
                total_reqs = total_requests_hint

            avg_lat = None
            # If both read+write averages exist but not their separate counts, use simple mean.
            if (avg_rd_lat is not None) and (avg_wr_lat is not None):
                avg_lat = 0.5 * (avg_rd_lat + avg_wr_lat)
            elif avg_rd_lat is not None:
                avg_lat = avg_rd_lat
            elif avg_wr_lat is not None:
                avg_lat = avg_wr_lat

            if (avg_lat is not None) and total_reqs:
                final_cycles = max(1, math.ceil(avg_lat * total_reqs))
            else:
                # 3) Last resort
                final_cycles = 1
        
        # Return (cycles, energy_pJ); 0 if energy is unavailable
        final_energy = total_energy if total_energy is not None else 0.0
        return (final_cycles, final_energy)

    except subprocess.TimeoutExpired:
        raise RuntimeError("Ramulator execution timeout")
    finally:
        try:
            os.unlink(temp_config_file)
        except FileNotFoundError:
            pass
