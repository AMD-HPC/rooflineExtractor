# Copyright (C) 2026 Advanced Micro Devices, Inc.
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file # or at https://opensource.org/licenses/MIT.

import pandas as pd
import numpy as np
import pdb
import shutil
import argparse
import json
import requests

from pathlib import Path

_config_dir = Path(__file__).parent / "config"

# Qualitative palette (matches Plotly default used previously for kernel colors)
_QUAL_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
    "#FF97FF", "#FECB52",
]

sigRuntime = 10  # Default value, can be changed from command line

# MFMA *instruction* counters (count the number of MFMA instructions issued, which
# is distinct from the SQ_INSTS_VALU_MFMA_MOPS_* counters that count total
# math operations. MFMA instructions are included in SQ_INSTS_VALU, so they are
# subtracted from the SQ_INSTS_VALU leftover to avoid being attributed to the
# "Other VALU" bucket. Collected on all platforms; guarded by column-existence
# checks to stay compatible with counter files captured by earlier versions.
MFMA_INST_COUNTERS = [
    'SQ_INSTS_VALU_MFMA_I8',
    'SQ_INSTS_VALU_MFMA_F6F4',
    'SQ_INSTS_VALU_MFMA_F8',
    'SQ_INSTS_VALU_MFMA_BF16',
    'SQ_INSTS_VALU_MFMA_F16',
    'SQ_INSTS_VALU_MFMA_F32',
    'SQ_INSTS_VALU_MFMA_F64',
]


def _mfma_inst_sum(df):
    """Sum of all MFMA instruction counters present in df (0 if none present)."""
    cols = [c for c in MFMA_INST_COUNTERS if c in df.columns]
    if not cols:
        return 0
    return df[cols].sum(axis=1)

# Bandwidths — loaded from config/cache_bandwidths.csv relative to this script's location
_caches_path = _config_dir / "cache_bandwidths.csv"
_caches_df = pd.read_csv(_caches_path, index_col="Architecture")
caches = _caches_df.to_dict(orient="index")


# Function to convert columns with type mismatches to integers
def convert_columns_to_int(df):
    counters = ['SQ_INSTS_VALU_ADD_F16', 'SQ_INSTS_VALU_MUL_F16',       'SQ_INSTS_VALU_FMA_F16', 'SQ_INSTS_VALU_TRANS_F16',       'SQ_INSTS_VALU_ADD_F32', 'SQ_INSTS_VALU_MUL_F32',       'SQ_INSTS_VALU_FMA_F32', 'SQ_INSTS_VALU_TRANS_F32',       'SQ_INSTS_VALU_ADD_F64', 'SQ_INSTS_VALU_MUL_F64',       'SQ_INSTS_VALU_FMA_F64', 'SQ_INSTS_VALU_TRANS_F64',       'SQ_INSTS_VALU_MFMA_MOPS_F16', 'SQ_INSTS_VALU_MFMA_MOPS_BF16',       'SQ_INSTS_VALU_MFMA_MOPS_F32', 'SQ_INSTS_VALU_MFMA_MOPS_F64', 'SQ_LDS_IDX_ACTIVE', 'SQ_LDS_BANK_CONFLICT',       'TCP_TCC_READ_REQ_sum', 'TCP_TCC_WRITE_REQ_sum',       'TCP_TCC_ATOMIC_WITH_RET_REQ_sum', 'TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum',       'TCP_TOTAL_CACHE_ACCESSES_sum', 'SQ_INSTS_VALU_INT32',       'SQ_INSTS_VALU_INT64',  'SQ_INSTS_SALU']

    # Checks for counters that were added in later versions of rooflineExtractor (to stay compatible with earlier counter files)
    if 'TCC_REQ_sum' in df.columns:
        counters.append('TCC_REQ_sum')
    if 'SQ_INSTS_VALU' in df.columns:
        counters.append('SQ_INSTS_VALU')
    if 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns:
        counters.append('SQ_INSTS_VALU_MFMA_MOPS_I8')
    if 'SQ_INSTS_VALU_MFMA_MOPS_F8' in df.columns:
        counters.append('SQ_INSTS_VALU_MFMA_MOPS_F8')
    if 'SQ_INSTS_VALU_MFMA_MOPS_F6F4' in df.columns:
        counters.append('SQ_INSTS_VALU_MFMA_MOPS_F6F4')
    for mfma_inst in MFMA_INST_COUNTERS:
        if mfma_inst in df.columns:
            counters.append(mfma_inst)
    # Check for CDNA2 vs. CDNA3-4 counters
    if 'TCC_BUBBLE_sum' in df.columns:
        counters.append('TCC_BUBBLE_sum')
        counters.append('TCC_EA0_RDREQ_sum')
        counters.append('TCC_EA0_RDREQ_32B_sum')
        counters.append('TCC_EA0_WRREQ_sum')
        counters.append('TCC_EA0_WRREQ_64B_sum')
    else:
        counters.append('TCC_EA_RDREQ_sum')
        counters.append('TCC_EA_RDREQ_32B_sum')
        counters.append('TCC_EA_WRREQ_sum')
        counters.append('TCC_EA_WRREQ_64B_sum')
    for c in (
        'TCC_EA0_WRREQ_WRITE_DRAM_32B_sum',
        'TCC_EA0_RDREQ_DRAM_32B_sum',
        'TCC_EA0_WRREQ_ATOMIC_DRAM_32B_sum',
    ):
        if c in df.columns:
            counters.append(c)

    for counter in counters:
        df[counter] = pd.to_numeric(df[counter], errors='coerce').astype(int)
    return df


def load_from_directory(dir_path):
    """Load counters.csv and trace_kernel_trace.csv from each immediate subdirectory.

    Each subdirectory with both files is treated as one application; rows are tagged
    with an Application column (subdirectory name) for correct counter–trace pairing.

    Returns (df_roof, df_runtime, base_name) for use with extract(..., base_name=...).
    """
    root = Path(dir_path).resolve()
    if not root.is_dir():
        print(f"Error: not a directory: {dir_path}")
        quit()
    roof_parts = []
    runtime_parts = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        c = sub / "counters.csv"
        t = sub / "trace_kernel_trace.csv"
        if not c.is_file() or not t.is_file():
            print(f"Skipping {sub.name}: missing counters.csv or trace_kernel_trace.csv")
            continue
        dfr = pd.read_csv(c)
        dft = pd.read_csv(t)
        dfr["Application"] = sub.name
        dft["Application"] = sub.name
        roof_parts.append(dfr)
        runtime_parts.append(dft)
    if not roof_parts:
        print(
            "Error: no subdirectory contains both counters.csv and trace_kernel_trace.csv"
        )
        quit()
    return (
        pd.concat(roof_parts, ignore_index=True),
        pd.concat(runtime_parts, ignore_index=True),
        root.name,
    )


def _safe_divide(numerator, denominator):
    """Divide two Series, replacing inf and NaN with 0."""
    return numerator.divide(denominator).replace(np.inf, 0).replace(np.nan, 0)


def _format_throughput(gflops):
    """Format a throughput in GFLOPs/s; switch to TFLOPs/s when >= 1 TFLOPs/s."""
    x = float(pd.to_numeric(gflops, errors="coerce"))
    if not np.isfinite(x):
        return "N/A"
    if abs(x) >= 1000.0:
        return f"{x / 1000:.3f} TFLOPs/s"
    return f"{x:.3f} GFLOPs/s"


def _format_bandwidth(gb_per_s):
    """Format a bandwidth in GB/s; switch to TB/s when >= 1 TB/s."""
    x = float(pd.to_numeric(gb_per_s, errors="coerce"))
    if not np.isfinite(x):
        return "N/A"
    if abs(x) >= 1000.0:
        return f"{x / 1000:.3f} TB/s"
    return f"{x:.3f} GB/s"


def _format_byte_size(n):
    """Format a byte count using B, KB, MB, or GB (1024-based); TB if needed."""
    x = float(pd.to_numeric(n, errors="coerce"))
    if np.isnan(x):
        return "nan"
    if np.isinf(x):
        return "inf" if x > 0 else "-inf"
    abs_x = abs(x)
    sign = -1.0 if x < 0 else 1.0
    units = ("B", "KB", "MB", "GB", "TB")
    u = 0
    v = abs_x
    while v >= 1024.0 and u < len(units) - 1:
        v /= 1024.0
        u += 1
    v *= sign
    if u == 0:
        if float(int(v)) == v:
            return f"{int(v)} B"
        return f"{v:.3f} B"
    return f"{v:.3f} {units[u]}"


def _compute_kernel_overlap_pct(df_runtime):
    """Percentage of summed kernel runtime that overlapped, plus the wall-active time.

    Returns ``(overlap_pct, wall_active_ns)`` where ``overlap_pct`` is
    ``(sum_of_durations - wall_active_time) / sum_of_durations * 100`` and
    ``wall_active_ns`` is the total wall-clock duration during which at least
    one kernel was running (summed across traces/agents). A 0% overlap means
    kernels ran fully sequentially; higher values indicate more concurrent
    execution (e.g. across multiple HSA queues on the same GPU).

    Timestamps are only comparable within a single trace/agent, so overlap is
    computed per (Application, Agent_Id) group when those columns exist, then
    aggregated. Returns ``(NaN, NaN)`` when the metric cannot be computed.
    """
    if df_runtime is None or len(df_runtime) == 0:
        return float("nan"), float("nan")
    if "Start_Timestamp" not in df_runtime.columns or "End_Timestamp" not in df_runtime.columns:
        return float("nan"), float("nan")

    group_cols = [c for c in ("Application", "Agent_Id") if c in df_runtime.columns]
    groups = df_runtime.groupby(group_cols, sort=False) if group_cols else [(None, df_runtime)]

    total_kernel_ns = 0
    total_wall_ns = 0
    for _, g in groups:
        intervals = []
        for start_raw, end_raw in zip(g["Start_Timestamp"], g["End_Timestamp"]):
            try:
                start = int(start_raw)
                end = int(end_raw)
            except (TypeError, ValueError):
                continue
            if end < start:
                continue
            intervals.append((start, end))

        if not intervals:
            continue

        total_kernel_ns += sum(end - start for start, end in intervals)

        # Sweep-line over interval endpoints to measure wall time during which
        # at least one kernel is running. Process kernel ends (-1) before starts
        # (+1) at equal timestamps so back-to-back kernels are treated as
        # contiguous rather than overlapping.
        events = []
        for start, end in intervals:
            events.append((start, 1))
            events.append((end, -1))
        events.sort(key=lambda event: (event[0], event[1]))

        active = 0
        last_t = events[0][0]
        wall = 0
        for t, d in events:
            if active > 0:
                wall += t - last_t
            active += d
            last_t = t
        total_wall_ns += wall

    if total_kernel_ns <= 0:
        return float("nan"), float("nan")
    overlap_pct = (total_kernel_ns - total_wall_ns) / total_kernel_ns * 100.0
    return overlap_pct, total_wall_ns


def _compute_ai_columns(df, mem_level):
    """Compute arithmetic intensity columns for one memory hierarchy level."""
    bw = df[f'BW_{mem_level}']
    cols = {}
    for suffix, total_col in [
        ('TOT', 'TOTAL_OPS'), ('SALU', 'TOTAL_SALU'),
        ('VALU_F16', 'TOTAL_VALU_F16'), ('VALU_F32', 'TOTAL_VALU_F32'),
        ('VALU_F64', 'TOTAL_VALU_F64'), ('VALU_I32', 'TOTAL_VALU_I32'),
        ('VALU_I64', 'TOTAL_VALU_I64'),
    ]:
        cols[f'AI_{mem_level}_{suffix}'] = _safe_divide(df[total_col], bw)
    for guard_col, suffix, total_col in [
        ('SQ_INSTS_VALU_MFMA_MOPS_F8', 'MOPS_F8', 'TOTAL_MOPS_F8'),
        ('SQ_INSTS_VALU_MFMA_MOPS_I8', 'MOPS_I8', 'TOTAL_MOPS_I8'),
        ('SQ_INSTS_VALU_MFMA_MOPS_F6F4', 'MOPS_F6F4', 'TOTAL_MOPS_F6F4'),
    ]:
        if guard_col in df.columns:
            cols[f'AI_{mem_level}_{suffix}'] = _safe_divide(df[total_col], bw)
    for suffix, total_col in [
        ('MOPS_F16', 'TOTAL_MOPS_F16'), ('MOPS_BF16', 'TOTAL_MOPS_BF16'),
        ('MOPS_F32', 'TOTAL_MOPS_F32'), ('MOPS_F64', 'TOTAL_MOPS_F64'),
    ]:
        cols[f'AI_{mem_level}_{suffix}'] = _safe_divide(df[total_col], bw)
    return cols


def _lookup_peak(df_bw, datatype, operation, arch_col):
    """Look up a single peak value from the benchWarmer DataFrame."""
    return df_bw.loc[
        (df_bw['Datatype'] == datatype) & (df_bw['Operation'] == operation), arch_col
    ].values[0]


def _default_flat_compute_peak(peaks):
    """Default flat compute roof (FP64 MFMA) for the roofline plot."""
    return peaks['fp64_mfma']


def _load_peaks(df_bw, arch_col, df):
    """Load all peak throughput values for a given architecture from benchWarmer data."""
    peaks = {}
    for dtype in ['fp16', 'fp32', 'fp64']:
        for op, key in [(' Add', 'add'), (' Mul', 'mul'), (' MulAdd', 'muladd'), (' Rsqrt', 'trans')]:
            peaks[f'{dtype}_{key}'] = _lookup_peak(df_bw, dtype, op, arch_col)

    if 'SQ_INSTS_VALU_MFMA_MOPS_F8' in df.columns:
        peaks['fp8_mfma'] = _lookup_peak(df_bw, 'fp8', ' mfma', arch_col)
    if 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns:
        peaks['i8_mfma'] = _lookup_peak(df_bw, 'int8', ' mfma', arch_col)
    # The F6F4 counter aggregates both FP4 and FP6 MFMA ops; use the FP4 MFMA peak
    # (slightly higher than FP6) as the achievable upper bound.
    if 'SQ_INSTS_VALU_MFMA_MOPS_F6F4' in df.columns:
        fp4_mfma_row = df_bw.loc[
            (df_bw['Datatype'] == 'fp4') & (df_bw['Operation'] == ' mfma'), arch_col
        ]
        if not fp4_mfma_row.empty and pd.notna(fp4_mfma_row.values[0]):
            peaks['f6f4_mfma'] = fp4_mfma_row.values[0]
    peaks['fp16_mfma'] = _lookup_peak(df_bw, 'fp16', ' mfma', arch_col)
    peaks['bf16_mfma'] = _lookup_peak(df_bw, 'bf16', ' mfma', arch_col)
    peaks['fp32_mfma'] = _lookup_peak(df_bw, 'fp32', ' mfma', arch_col)
    # Handle case where fp64 mfma is not present (MI100)
    fp64_mfma_row = df_bw.loc[
        (df_bw['Datatype'] == 'fp64') & (df_bw['Operation'] == ' mfma'), arch_col
    ]
    peaks['fp64_mfma'] = fp64_mfma_row.values[0] if not fp64_mfma_row.empty else 0

    # INT32 peak is the best of the int32 Add and int32 Mul benchWarmer tests, since both
    # are recognized by the hardware counters as INT32 ops and the achieved throughput
    # can differ between adds and multiplies on a given arch.
    peaks['int32'] = max(
        _lookup_peak(df_bw, 'int32', ' Add', arch_col),
        _lookup_peak(df_bw, 'int32', ' Mul', arch_col),
    )
    # INT32 MulAdd is the best performing test that hardware counters recognize as INT64; divide by 2 for GInsts/s
    peaks['int64'] = _lookup_peak(df_bw, 'int32', ' MulAdd', arch_col) / 2
    # Assume 'other' operations are int8 left shifts (best performing operation without a dedicated counter)
    peaks['other'] = _lookup_peak(df_bw, 'int8', ' ShiftLeft', arch_col)

    return peaks


def _compute_kernel_peak(df, peaks):
    """Compute kernel-specific compute peak from instruction counts and peak throughputs."""
    weighted_time = (
        64 * df['SQ_INSTS_VALU_ADD_F16'] / peaks['fp16_add'] +
        64 * df['SQ_INSTS_VALU_MUL_F16'] / peaks['fp16_mul'] +
        64 * 2 * df['SQ_INSTS_VALU_FMA_F16'] / peaks['fp16_muladd'] +
        64 * df['SQ_INSTS_VALU_TRANS_F16'] / peaks['fp16_trans'] +
        64 * df['SQ_INSTS_VALU_ADD_F32'] / peaks['fp32_add'] +
        64 * df['SQ_INSTS_VALU_MUL_F32'] / peaks['fp32_mul'] +
        64 * 2 * df['SQ_INSTS_VALU_FMA_F32'] / peaks['fp32_muladd'] +
        64 * df['SQ_INSTS_VALU_TRANS_F32'] / peaks['fp32_trans'] +
        64 * df['SQ_INSTS_VALU_ADD_F64'] / peaks['fp64_add'] +
        64 * df['SQ_INSTS_VALU_MUL_F64'] / peaks['fp64_mul'] +
        64 * 2 * df['SQ_INSTS_VALU_FMA_F64'] / peaks['fp64_muladd'] +
        64 * df['SQ_INSTS_VALU_TRANS_F64'] / peaks['fp64_trans'] +
        64 * df['SQ_INSTS_VALU_INT32'] / peaks['int32'] +
        64 * df['SQ_INSTS_VALU_INT64'] / peaks['int64'] +
        df['TOTAL_MOPS_F16'] / peaks['fp16_mfma'] +
        df['TOTAL_MOPS_BF16'] / peaks['bf16_mfma'] +
        df['TOTAL_MOPS_F32'] / peaks['fp32_mfma'] +
        df['TOTAL_MOPS_F64'] / peaks['fp64_mfma']
    )

    if 'TOTAL_VALU_OTHER' in df.columns:
        weighted_time += df['TOTAL_VALU_OTHER'] / peaks['other']
    if 'fp8_mfma' in peaks:
        weighted_time += df['TOTAL_MOPS_F8'] / peaks['fp8_mfma']
    if 'i8_mfma' in peaks:
        weighted_time += df['TOTAL_MOPS_I8'] / peaks['i8_mfma']
    if 'f6f4_mfma' in peaks and 'TOTAL_MOPS_F6F4' in df.columns:
        weighted_time += df['TOTAL_MOPS_F6F4'] / peaks['f6f4_mfma']

    return ((df['TOTAL_OPS'] - df['TOTAL_SALU']) / weighted_time).replace(np.inf, 0)


_JKL_ALPHA_BY_LABEL_CACHE = {}


def _hbm_alpha_cache_key(arch):
    return str(arch).strip().lower().replace(" ", "_")


def _alpha_summary_csv_path(arch):
    """Return path to ``config/{arch}_alpha_summary.csv`` if it exists (e.g. mi300x_alpha_summary.csv)."""
    stem = _hbm_alpha_cache_key(arch)
    if not stem:
        return None
    p = _config_dir / f"{stem}_alpha_summary.csv"
    return p if p.is_file() else None


def _use_hbm_alpha_model(arch):
    """True when a per-arch α summary CSV is present for *arch*."""
    return _alpha_summary_csv_path(arch) is not None


def _load_jkl_alpha_by_label(arch, sweep):
    """Load α table from ``{arch}_alpha_summary.csv`` for a given sweep (``hbm`` or ``lds``)."""
    global _JKL_ALPHA_BY_LABEL_CACHE
    arch_key = _hbm_alpha_cache_key(arch)
    sw = str(sweep).strip().lower()
    cache_key = (arch_key, sw)
    if cache_key not in _JKL_ALPHA_BY_LABEL_CACHE:
        path = _alpha_summary_csv_path(arch)
        if path is None:
            _JKL_ALPHA_BY_LABEL_CACHE[cache_key] = {}
        else:
            d = pd.read_csv(path)
            if "sweep" in d.columns:
                d = d[d["sweep"].astype(str).str.lower() == sw]
            elif "log_basename" in d.columns:
                prefix = "hbm_" if sw == "hbm" else "lds_" if sw == "lds" else f"{sw}_"
                d = d[d["log_basename"].astype(str).str.lower().str.startswith(prefix)]
            _JKL_ALPHA_BY_LABEL_CACHE[cache_key] = {
                str(row["label"]): float(row["alpha"])
                for _, row in d.iterrows()
                if pd.notna(row.get("alpha"))
            }
    return _JKL_ALPHA_BY_LABEL_CACHE[cache_key]


def _load_hbm_alpha_by_label(arch):
    """Load HBM α table (HBM sweeps only; lazy, cached per arch)."""
    return _load_jkl_alpha_by_label(arch, "hbm")


def _default_hbm_alpha_fp64_mfma(arch):
    """Alpha for 100% FP64 MFMA from HBM table (for default, non-kernel roofline)."""
    m = _load_hbm_alpha_by_label(arch)
    return m.get("fp64_mfma")


def _use_lds_alpha_model(arch):
    """True when JKL summary exists and contains LDS sweep rows."""
    if not _use_hbm_alpha_model(arch):
        return False
    return len(_load_jkl_alpha_by_label(arch, "lds")) > 0


def _default_lds_alpha_fp64_mfma(arch):
    """Alpha for 100% FP64 MFMA from LDS table (default roofline when no kernel selected)."""
    m = _load_jkl_alpha_by_label(arch, "lds")
    return m.get("fp64_mfma")


def _term_time(ops_over_peak):
    """Sanitize ops/peak time contribution (match _compute_kernel_peak behavior)."""
    return ops_over_peak.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _compute_weighted_jkl_alpha(df, peaks, arch, sweep):
    """Instruction-mix–weighted α using the same per-class times as _compute_kernel_peak."""
    alpha_map = _load_jkl_alpha_by_label(arch, sweep)
    terms = {}

    def add_term(label, expr):
        if label not in alpha_map:
            return
        terms[label] = _term_time(expr)

    add_term(
        "fp16_add",
        64 * df["SQ_INSTS_VALU_ADD_F16"] / peaks["fp16_add"],
    )
    add_term(
        "fp16_mul",
        64 * df["SQ_INSTS_VALU_MUL_F16"] / peaks["fp16_mul"],
    )
    add_term(
        "fp16_muladd",
        64 * 2 * df["SQ_INSTS_VALU_FMA_F16"] / peaks["fp16_muladd"],
    )
    add_term(
        "fp16_rsqrt",
        64 * df["SQ_INSTS_VALU_TRANS_F16"] / peaks["fp16_trans"],
    )
    add_term(
        "fp32_add",
        64 * df["SQ_INSTS_VALU_ADD_F32"] / peaks["fp32_add"],
    )
    add_term(
        "fp32_mul",
        64 * df["SQ_INSTS_VALU_MUL_F32"] / peaks["fp32_mul"],
    )
    add_term(
        "fp32_muladd",
        64 * 2 * df["SQ_INSTS_VALU_FMA_F32"] / peaks["fp32_muladd"],
    )
    add_term(
        "fp32_rsqrt",
        64 * df["SQ_INSTS_VALU_TRANS_F32"] / peaks["fp32_trans"],
    )
    add_term(
        "fp64_add",
        64 * df["SQ_INSTS_VALU_ADD_F64"] / peaks["fp64_add"],
    )
    add_term(
        "fp64_mul",
        64 * df["SQ_INSTS_VALU_MUL_F64"] / peaks["fp64_mul"],
    )
    add_term(
        "fp64_muladd",
        64 * 2 * df["SQ_INSTS_VALU_FMA_F64"] / peaks["fp64_muladd"],
    )
    add_term(
        "fp64_rsqrt",
        64 * df["SQ_INSTS_VALU_TRANS_F64"] / peaks["fp64_trans"],
    )
    add_term(
        "int32_mul",
        64 * df["SQ_INSTS_VALU_INT32"] / peaks["int32"],
    )
    add_term(
        "int32_muladd",
        64 * df["SQ_INSTS_VALU_INT64"] / peaks["int64"],
    )
    add_term("fp16_mfma", df["TOTAL_MOPS_F16"] / peaks["fp16_mfma"])
    add_term("bf16_mfma", df["TOTAL_MOPS_BF16"] / peaks["bf16_mfma"])
    add_term("fp32_mfma", df["TOTAL_MOPS_F32"] / peaks["fp32_mfma"])
    add_term("fp64_mfma", df["TOTAL_MOPS_F64"] / peaks["fp64_mfma"])
    if "TOTAL_VALU_OTHER" in df.columns:
        add_term("int8_leftshift", df["TOTAL_VALU_OTHER"] / peaks["other"])
    if "fp8_mfma" in peaks and "TOTAL_MOPS_F8" in df.columns:
        add_term("fp8_mfma", df["TOTAL_MOPS_F8"] / peaks["fp8_mfma"])
    if "i8_mfma" in peaks and "TOTAL_MOPS_I8" in df.columns:
        add_term("int8_mfma", df["TOTAL_MOPS_I8"] / peaks["i8_mfma"])
    # FP6/FP4 MFMA ops use the FP4 MFMA peak; the alpha summary CSVs label this as fp4_mfma.
    if "f6f4_mfma" in peaks and "TOTAL_MOPS_F6F4" in df.columns:
        add_term("fp4_mfma", df["TOTAL_MOPS_F6F4"] / peaks["f6f4_mfma"])

    if not terms:
        return pd.Series(0.0, index=df.index)

    term_df = pd.DataFrame(terms)
    t_sum = term_df.sum(axis=1)
    t_safe = t_sum.replace(0, np.nan)

    a_arr = np.array([alpha_map[c] for c in term_df.columns])
    alpha_series = (term_df * a_arr).sum(axis=1) / t_safe
    alpha_series = alpha_series.fillna(0.0)
    return alpha_series


def _compute_weighted_hbm_alpha(df, peaks, arch):
    """Instruction-mix–weighted HBM α."""
    return _compute_weighted_jkl_alpha(df, peaks, arch, "hbm")


def _compute_weighted_lds_alpha(df, peaks, arch):
    """Instruction-mix–weighted LDS α."""
    return _compute_weighted_jkl_alpha(df, peaks, arch, "lds")


def _hbm_roof_throughput(x, B, P, alpha):
    """HBM roof: (1/P + 1/(x*B) - alpha*min(1/P, 1/(x*B)))^-1; x=AI, B=BW peak, P=kernel compute peak."""
    eps = 1e-30
    B = float(B)
    if B <= 0:
        return pd.Series(0.0, index=x.index)

    x = x.clip(lower=eps)
    inv_p = 1.0 / P.clip(lower=eps)
    inv_xb = 1.0 / (x * B)
    m = np.minimum(inv_p, inv_xb)
    denom = inv_p + inv_xb - alpha * m
    linear = np.minimum(x * B, P)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 1.0 / denom
    out = raw.where(np.isfinite(raw) & (denom > eps), linear)
    return out.fillna(linear).clip(lower=0.0)


def _percent_roof_achieved(throughput, peak):
    """Achieved throughput as % of roof PEAK; NaN when ratio is undefined or non-finite."""
    t = throughput.astype(float)
    p = peak.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = t / p * 100.0
    return r.where(np.isfinite(t) & np.isfinite(p) & (p > 0) & np.isfinite(r))


def _exclude_bw_roof_when_no_traffic(peak, bw):
    """
    If observed byte traffic at this memory level is zero, that level does not bound the kernel.
    Use +inf so min() ignores it (linear AI×B would be inf anyway; curved α can misbehave at AI→∞).
    """
    b = bw.astype(float)
    return peak.where(b > 0, np.inf)


def _safe_limiters(masked_df, strip_len, limiter_suffix):
    """Compute limiter labels per row; use 'Unknown' + suffix when a row has no valid (positive) peaks."""
    # Avoid idxmin on all-NA rows (deprecated → future ValueError); those rows become Unknown via fillna below.
    all_na = masked_df.isna().all(axis=1)
    idx = pd.Series(np.nan, index=masked_df.index, dtype=object)
    has_any = ~all_na
    if has_any.any():
        idx.loc[has_any] = masked_df.loc[has_any].idxmin(axis=1)
    base = idx.str[:-strip_len]
    # The fixed-length strip leaves residual `_PEAK_L` on linear-roof labels (e.g.
    # `HBM_BW_PEAK_LINEAR` → `HBM_BW_PEAK_L` after stripping `_PEAK`). Trim it so
    # linear-limiter labels read as e.g. `HBM_BW` rather than `HBM_BW_PEAK_L`.
    base = base.str.replace(r"_PEAK_L$", "", regex=True)
    return (base + limiter_suffix).fillna("Unknown" + limiter_suffix)


_MEMORY_LIMITER_LEVELS = ("HBM", "L2", "vL1d", "LDS")


def _limiter_memory_level(limiter_str):
    """Return the memory level ('HBM', 'L2', 'vL1d', 'LDS') if `limiter_str` is a memory-bandwidth
    roof label (e.g. 'HBM_BW (gfx942)'), otherwise None."""
    if not isinstance(limiter_str, str) or not limiter_str:
        return None
    tag = limiter_str.split()[0]
    for level in _MEMORY_LIMITER_LEVELS:
        if tag.startswith(f"{level}_BW"):
            return level
    return None


def _align_curved_limiter_with_linear_compute(limiter, limiter_linear):
    """
    When the curved HBM/LDS α roof falls below compute but the linear roof is compute-bound,
    idxmin attributes the bottleneck to HBM/LDS. Report KERNEL_COMPUTE as the curved limiter only;
    PEAK (curved roof) and all linear columns stay unchanged.

    Treat any limiter whose first token starts with `HBM_BW` or `LDS_BW` as the
    corresponding memory roof family so the alignment still applies.
    """
    curved_tag = limiter.str.split().str[0]
    is_memory_bw = curved_tag.str.startswith("HBM_BW", na=False) | curved_tag.str.startswith(
        "LDS_BW", na=False
    )
    mask = limiter_linear.str.contains("KERNEL_COMPUTE", na=False) & is_memory_bw
    return limiter.where(~mask, limiter_linear)


# Compute total flops, AI's
def compute_flops(df, arch):
    # Drop dispatches whose VALU counters are inconsistent across PMC passes.
    # SQ_INSTS_VALU is captured in a different PMC counter-collection run than the typed
    # counters (SQ_INSTS_VALU_ADD_*, _MUL_*, _FMA_*, _TRANS_*, _INT32, _INT64, plus
    # MFMA instruction counters). The leftover 64 * (SQ_INSTS_VALU - typed_sum)
    # represents bitwise ops not
    # captured by the typed counters and must be >= 0. A negative leftover means one of
    # the PMC counter-collection files is corrupted/inconsistent for that dispatch
    # (typically the SQ_INSTS_VALU pass came back as 0). Drop those dispatches with a
    # warning so they don't poison TOTAL_OPS, the kernel compute peak, and the achieved-% ratio.
    if 'SQ_INSTS_VALU' in df.columns:
        _valu_typed_sum = (
            df['SQ_INSTS_VALU_ADD_F16'] + df['SQ_INSTS_VALU_MUL_F16'] + df['SQ_INSTS_VALU_TRANS_F16'] + df['SQ_INSTS_VALU_FMA_F16']
            + df['SQ_INSTS_VALU_ADD_F32'] + df['SQ_INSTS_VALU_MUL_F32'] + df['SQ_INSTS_VALU_TRANS_F32'] + df['SQ_INSTS_VALU_FMA_F32']
            + df['SQ_INSTS_VALU_ADD_F64'] + df['SQ_INSTS_VALU_MUL_F64'] + df['SQ_INSTS_VALU_TRANS_F64'] + df['SQ_INSTS_VALU_FMA_F64']
            + df['SQ_INSTS_VALU_INT32'] + df['SQ_INSTS_VALU_INT64'] + _mfma_inst_sum(df)
        )
        _valu_other_raw = 64 * (df['SQ_INSTS_VALU'] - _valu_typed_sum)
        _corrupt_mask = _valu_other_raw < 0
        if _corrupt_mask.any():
            _bad = df.loc[_corrupt_mask].copy()
            _bad['__VALU_LEFTOVER'] = _valu_other_raw.loc[_corrupt_mask]
            _kernel_col = 'Kernel_Name' if 'Kernel_Name' in _bad.columns else (
                'KernelName' if 'KernelName' in _bad.columns else None
            )
            _id_col = 'Dispatch_Id' if 'Dispatch_Id' in _bad.columns else (
                'Index' if 'Index' in _bad.columns else None
            )
            print(
                f"WARNING: {int(_corrupt_mask.sum())}/{len(df)} dispatch(es) have "
                f"SQ_INSTS_VALU < sum of typed VALU counters, indicating one of the "
                f"counter-collection PMC files is corrupted for these dispatches. "
                f"Dropping them from analysis."
            )
            _max_listed = 10
            _to_show = _bad.head(_max_listed)
            for _, _r in _to_show.iterrows():
                _id_str = f"Dispatch {int(_r[_id_col])}" if _id_col is not None else "<unknown id>"
                _kn_str = str(_r[_kernel_col]) if _kernel_col is not None else "<unknown kernel>"
                _valu = float(_r.get('SQ_INSTS_VALU', float('nan')))
                _typed = _valu - float(_r['__VALU_LEFTOVER']) / 64.0
                print(
                    f"  - {_id_str}: SQ_INSTS_VALU={int(_valu)}, "
                    f"typed_sum={int(_typed)}, leftover={int(_r['__VALU_LEFTOVER'])} ({_kn_str})"
                )
            if int(_corrupt_mask.sum()) > _max_listed:
                print(f"  ... and {int(_corrupt_mask.sum()) - _max_listed} more")
            df = df.loc[~_corrupt_mask].copy()

    # Create a dictionary to store all new columns
    new_columns = {}

    # Compute total achieved FLOPs for each datatype (FP16, FP32, FP64)
    ## Scalar Ops
    new_columns['TOTAL_SALU'] = df['SQ_INSTS_SALU']

    ## Vector Ops
    new_columns['TOTAL_VALU_F16'] = 64 * (df['SQ_INSTS_VALU_ADD_F16'] + df['SQ_INSTS_VALU_MUL_F16'] + df['SQ_INSTS_VALU_TRANS_F16'] + 2 * df['SQ_INSTS_VALU_FMA_F16'])
    new_columns['TOTAL_VALU_F32'] = 64 * (df['SQ_INSTS_VALU_ADD_F32'] + df['SQ_INSTS_VALU_MUL_F32'] + df['SQ_INSTS_VALU_TRANS_F32'] + 2 * df['SQ_INSTS_VALU_FMA_F32'])
    new_columns['TOTAL_VALU_F64'] = 64 * (df['SQ_INSTS_VALU_ADD_F64'] + df['SQ_INSTS_VALU_MUL_F64'] + df['SQ_INSTS_VALU_TRANS_F64'] + 2 * df['SQ_INSTS_VALU_FMA_F64'])
    new_columns['TOTAL_VALU_I32'] = 64 * df['SQ_INSTS_VALU_INT32']
    new_columns['TOTAL_VALU_I64'] = 64 * df['SQ_INSTS_VALU_INT64']

    ## Matrix Ops
    new_columns['TOTAL_MOPS_F16'] = 512 * df['SQ_INSTS_VALU_MFMA_MOPS_F16']
    new_columns['TOTAL_MOPS_BF16'] = 512 * df['SQ_INSTS_VALU_MFMA_MOPS_BF16']
    new_columns['TOTAL_MOPS_F32'] = 512 * df['SQ_INSTS_VALU_MFMA_MOPS_F32']
    new_columns['TOTAL_MOPS_F64'] = 512 * df['SQ_INSTS_VALU_MFMA_MOPS_F64']
    if 'SQ_INSTS_VALU_MFMA_MOPS_F8' in df.columns:
        new_columns['TOTAL_MOPS_F8'] = 512 * df['SQ_INSTS_VALU_MFMA_MOPS_F8']
    if 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns:
        new_columns['TOTAL_MOPS_I8'] = 512 * df['SQ_INSTS_VALU_MFMA_MOPS_I8']
    if 'SQ_INSTS_VALU_MFMA_MOPS_F6F4' in df.columns:
        new_columns['TOTAL_MOPS_F6F4'] = 512 * df['SQ_INSTS_VALU_MFMA_MOPS_F6F4']

    # Other VALU Ops (e.g. bitwise ops, INT16). Corrupted dispatches (where this leftover would
    # be negative) were dropped at the top of compute_flops, so this is guaranteed >= 0.
    if 'SQ_INSTS_VALU' in df.columns:
        new_columns['TOTAL_VALU_OTHER'] = 64 * (
            df['SQ_INSTS_VALU']
            - (
                df['SQ_INSTS_VALU_ADD_F16'] + df['SQ_INSTS_VALU_MUL_F16'] + df['SQ_INSTS_VALU_TRANS_F16'] + df['SQ_INSTS_VALU_FMA_F16']
                + df['SQ_INSTS_VALU_ADD_F32'] + df['SQ_INSTS_VALU_MUL_F32'] + df['SQ_INSTS_VALU_TRANS_F32'] + df['SQ_INSTS_VALU_FMA_F32']
                + df['SQ_INSTS_VALU_ADD_F64'] + df['SQ_INSTS_VALU_MUL_F64'] + df['SQ_INSTS_VALU_TRANS_F64'] + df['SQ_INSTS_VALU_FMA_F64']
                + df['SQ_INSTS_VALU_INT32'] + df['SQ_INSTS_VALU_INT64'] + _mfma_inst_sum(df)
            )
        )

    # Concat first batch of columns
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    ## Total
    TOTAL_OPS = df['TOTAL_SALU'] + df['TOTAL_VALU_F16'] + df['TOTAL_VALU_F32'] + df['TOTAL_VALU_F64'] + df['TOTAL_VALU_I32'] + df['TOTAL_VALU_I64'] + df['TOTAL_MOPS_F16'] + df['TOTAL_MOPS_BF16'] + df['TOTAL_MOPS_F32'] + df['TOTAL_MOPS_F64']
    if 'SQ_INSTS_VALU' in df.columns:
        TOTAL_OPS = TOTAL_OPS + df['TOTAL_VALU_OTHER']
    if 'SQ_INSTS_VALU_MFMA_MOPS_F8' in df.columns:
        TOTAL_OPS = TOTAL_OPS + df['TOTAL_MOPS_F8']
    if 'SQ_INSTS_VALU_MFMA_MOPS_I8' in df.columns:
        TOTAL_OPS = TOTAL_OPS + df['TOTAL_MOPS_I8']
    if 'SQ_INSTS_VALU_MFMA_MOPS_F6F4' in df.columns:
        TOTAL_OPS = TOTAL_OPS + df['TOTAL_MOPS_F6F4']

    new_columns = {}
    new_columns['TOTAL_OPS'] = TOTAL_OPS

    # Compute Bandwidths
    ## LDS
    new_columns['BW_LDS'] = 32 * 4 * (df['SQ_LDS_IDX_ACTIVE'] - df['SQ_LDS_BANK_CONFLICT'])
    new_columns['BW_LDS_ATOMICS'] = 64 * (df['TCP_TCC_ATOMIC_WITH_RET_REQ_sum'] + df['TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum'])

    ## L2
    if 'TCC_REQ_sum' in df.columns:
        new_columns['BW_L2'] = 128 * df['TCC_REQ_sum']
    else:
        # Less reliable calculation kept to be backwards compatible with earlier rooflineExtractor versions
        new_columns['BW_L2'] = 64 * df['TCP_TCC_READ_REQ_sum'] + 64 * df['TCP_TCC_WRITE_REQ_sum'] + new_columns['BW_LDS_ATOMICS']

    ## vL1D
    new_columns['BW_vL1d'] = 64 * df['TCP_TOTAL_CACHE_ACCESSES_sum']

    ## HBM
    ### Check architecture
    _gfx950_hbm_dram = (
        'TCC_EA0_WRREQ_WRITE_DRAM_32B_sum',
        'TCC_EA0_RDREQ_DRAM_32B_sum',
        'TCC_EA0_WRREQ_ATOMIC_DRAM_32B_sum',
    )
    if ('MI35' in arch) and all(c in df.columns for c in _gfx950_hbm_dram):
        # Count 32-byte read, write, and atomic requests to HBM
        # 1 64-byte request will be counted as 2, 1 128-byte (read) will be counted as 4
        new_columns['BW_HBM'] = (
            32 * df['TCC_EA0_WRREQ_WRITE_DRAM_32B_sum']
            + 32 * df['TCC_EA0_RDREQ_DRAM_32B_sum']
            + 32 * df['TCC_EA0_WRREQ_ATOMIC_DRAM_32B_sum']
        )
    elif df.keys().str.contains('TCC_BUBBLE').sum() > 0:
        # We have a gfx942 or gfx950 arch counter file (legacy HBM model without gfx950 DRAM 32B sums for backward compatibility)
        new_columns['BW_HBM'] = 128 * df['TCC_BUBBLE_sum'] + 32 * df['TCC_EA0_RDREQ_32B_sum'] + 64 * (df['TCC_EA0_RDREQ_sum'] - df['TCC_BUBBLE_sum'] - df['TCC_EA0_RDREQ_32B_sum']) + 32 * (df['TCC_EA0_WRREQ_sum'] - df['TCC_EA0_WRREQ_64B_sum']) + 64 * df['TCC_EA0_WRREQ_64B_sum']
    else:
        # Assuming gfx90a
        new_columns['BW_HBM'] = 32 * df['TCC_EA_RDREQ_32B_sum'] + 64 * (df['TCC_EA_RDREQ_sum'] - df['TCC_EA_RDREQ_32B_sum']) + 32 * (df['TCC_EA_WRREQ_sum'] - df['TCC_EA_WRREQ_64B_sum']) + 64 * df['TCC_EA_WRREQ_64B_sum']

    # Concat bandwidth columns
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    # Compute AI for each part of memory hierarchy (HBM, L2, L1)
    new_columns = {}

    for mem_level in ['LDS', 'L2', 'vL1d', 'HBM']:
        new_columns.update(_compute_ai_columns(df, mem_level))

    # Concat AI columns
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    # Add columns for peaks
    new_columns = {}
    new_columns['HBM_BW_PEAK'] = df['AI_HBM_TOT'] * caches[arch]['HBM']
    new_columns['L2_BW_PEAK'] = df['AI_L2_TOT'] * caches[arch]['L2']
    new_columns['vL1d_BW_PEAK'] = df['AI_vL1d_TOT'] * caches[arch]['vL1d']
    new_columns['LDS_BW_PEAK'] = df['AI_LDS_TOT'] * caches[arch]['LDS']
    # Linear AI×peak-BW roofs for HBM/LDS (same functional form as L2/vL1d); retained when α model replaces curved HBM/LDS peaks.
    new_columns['HBM_BW_PEAK_LINEAR'] = df['AI_HBM_TOT'] * caches[arch]['HBM']
    new_columns['LDS_BW_PEAK_LINEAR'] = df['AI_LDS_TOT'] * caches[arch]['LDS']

    # Concat peak columns
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    bw_path = _config_dir / "benchWarmer.csv"
    df_bw = pd.read_csv(bw_path)
    peaks = _load_peaks(df_bw, arch, df)

    # Calculate kernel-specific compute peak for current architecture
    new_columns = {}
    new_columns['KERNEL_COMPUTE_PEAK'] = _compute_kernel_peak(df, peaks)
    new_columns['COMPUTE_PEAK'] = _default_flat_compute_peak(peaks)

    # Concat compute peak columns
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    if _use_hbm_alpha_model(arch):
        alpha_base = _compute_weighted_hbm_alpha(df, peaks, arch)
        df["HBM_BW_PEAK"] = _exclude_bw_roof_when_no_traffic(
            _hbm_roof_throughput(
                df["AI_HBM_TOT"],
                caches[arch]["HBM"],
                df["KERNEL_COMPUTE_PEAK"],
                alpha_base,
            ),
            df["BW_HBM"],
        )
        df["HBM_ALPHA"] = alpha_base

    if _use_lds_alpha_model(arch):
        lds_alpha_base = _compute_weighted_lds_alpha(df, peaks, arch)
        df["LDS_BW_PEAK"] = _exclude_bw_roof_when_no_traffic(
            _hbm_roof_throughput(
                df["AI_LDS_TOT"],
                caches[arch]["LDS"],
                df["KERNEL_COMPUTE_PEAK"],
                lds_alpha_base,
            ),
            df["BW_LDS"],
        )
        df["LDS_ALPHA"] = lds_alpha_base

    # Determine performance peak/limiter
    df_peaks = df[['HBM_BW_PEAK','L2_BW_PEAK','vL1d_BW_PEAK','LDS_BW_PEAK','KERNEL_COMPUTE_PEAK']]
    peaks = df_peaks.where(df_peaks > 0).min(axis=1)
    limiters = _safe_limiters(df_peaks.where(df_peaks > 0), 5, f" ({arch})")

    df_peaks_linear = df[
        ['HBM_BW_PEAK_LINEAR', 'L2_BW_PEAK', 'vL1d_BW_PEAK', 'LDS_BW_PEAK_LINEAR', 'KERNEL_COMPUTE_PEAK']
    ]
    peaks_linear = df_peaks_linear.where(df_peaks_linear > 0).min(axis=1)
    limiters_linear = _safe_limiters(df_peaks_linear.where(df_peaks_linear > 0), 5, f" ({arch})")

    limiters = _align_curved_limiter_with_linear_compute(limiters, limiters_linear)

    new_columns = {}
    new_columns['PEAK'] = peaks
    new_columns['LIMITER'] = limiters
    new_columns['PEAK_LINEAR'] = peaks_linear
    new_columns['LIMITER_LINEAR'] = limiters_linear

    # Concat peak/limiter columns
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)


    return df


def _wrap_kernel_name_tooltip(name, width=50, max_lines=3):
    """Wrap long kernel names for tooltip display (newline-separated)."""
    if not name:
        return ""
    max_chars = width * max_lines
    if len(name) <= max_chars:
        return "\n".join(name[i : i + width] for i in range(0, len(name), width))
    truncated = name[: max_chars - 3] + "..."
    return "\n".join(truncated[i : i + width] for i in range(0, len(truncated), width))


def _json_safe_float(x):
    """Finite float for JSON, or None. Browsers' JSON.parse rejects Infinity/NaN (Python json emits them by default)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


_D3_CDN_URL = "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"
_D3_LOCAL_PATH = Path(__file__).parent / "d3.min.js"


def _d3_script_tag():
    """Return an inline <script> with the bundled D3 source, or a CDN <script src> fallback."""
    if _D3_LOCAL_PATH.is_file():
        d3_src = _D3_LOCAL_PATH.read_text(encoding="utf-8")
        return f"<script>{d3_src}</script>"
    return f'<script src="{_D3_CDN_URL}"></script>'


def _build_roofline_d3_html(payload):
    json_str = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    json_str = json_str.replace("</", "<\\/")
    html = _ROOFLINE_D3_HTML_TEMPLATE.replace("__D3_SCRIPT__", _d3_script_tag())
    return html.replace("__PAYLOAD__", json_str)


_ROOFLINE_D3_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="theme-dark">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Roofline</title>
__D3_SCRIPT__
<style>
  :root.theme-light {
    --plot-bg: #ffffff;
    --paper-bg: #f9f9f9;
    --fg: #111111;
    --grid: #d3d3d3;
    --axis: #000000;
    --tooltip-bg: rgba(255,255,255,0.95);
    --tooltip-fg: #111111;
    --tooltip-border: #888;
  }
  :root.theme-dark {
    --plot-bg: #000000;
    --paper-bg: #0a0a0a;
    --fg: #eeeeee;
    --grid: #555555;
    --axis: #ffffff;
    --tooltip-bg: rgba(30,30,30,0.95);
    --tooltip-fg: #eeeeee;
    --tooltip-border: #666;
  }
  html, body { height: 100%; margin: 0; }
  body {
    font-family: system-ui, sans-serif;
    background: var(--paper-bg);
    color: var(--fg);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    box-sizing: border-box;
  }
  #roofline-app {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
    width: 100%;
    max-width: none;
    margin: 0 auto;
    padding: 12px 16px 24px;
    box-sizing: border-box;
  }
  .chart-row {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: row;
    align-items: stretch;
    gap: 14px;
    flex-wrap: wrap;
  }
  #chart-wrap {
    flex: 1 1 480px;
    min-width: 200px;
    /* min-height:0 lets flex constrain height; without it, min-size follows SVG and fights ResizeObserver */
    min-height: 0;
    overflow: hidden;
    width: 100%;
    background: var(--plot-bg);
    border: 1px solid var(--grid);
    border-radius: 4px;
    box-sizing: border-box;
  }
  .kernel-legend-panel {
    flex: 0 1 320px;
    max-width: min(480px, 100%);
    align-self: stretch;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    max-height: min(560px, calc(100vh - 200px));
    border: 1px solid var(--grid);
    border-radius: 6px;
    padding: 10px 12px;
    background: var(--plot-bg);
    font-size: 11px;
    line-height: 1.35;
    box-sizing: border-box;
  }
  /* Legend to the right of plot: height at most matches the chart row; shorter if few kernels */
  @media (min-width: 900px) {
    .kernel-legend-panel {
      align-self: flex-start;
      max-height: 100%;
    }
  }
  .kernel-legend-list {
    list-style: none;
    margin: 0;
    padding: 0;
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
  }
  .kernel-swatch {
    flex-shrink: 0;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-top: 3px;
    border: 1px solid rgba(0,0,0,0.2);
  }
  .theme-dark .kernel-swatch { border-color: rgba(255,255,255,0.25); }
  .kernel-name { flex: 1 1 auto; min-width: 0; }
  .kernel-pct { flex-shrink: 0; opacity: 0.85; white-space: nowrap; }
  .kernel-legend-header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 6px;
  }
  .kernel-legend-heading {
    margin: 0;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--fg);
  }
  .kernel-legend-clear {
    font-size: 0.75rem;
    padding: 3px 10px;
    cursor: pointer;
    border: 1px solid var(--grid);
    border-radius: 4px;
    background: var(--paper-bg);
    color: var(--fg);
  }
  .kernel-legend-help {
    font-size: 0.72rem;
    opacity: 0.78;
    margin: 0 0 8px 0;
    color: var(--fg);
    line-height: 1.3;
  }
  .kernel-legend-list li {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    word-break: break-word;
    cursor: pointer;
    user-select: none;
    border-radius: 4px;
    padding: 3px 5px;
    margin: 1px -5px;
    transition: opacity 0.12s ease, background 0.12s ease;
  }
  .kernel-legend-list li.kernel-legend-selected {
    background: rgba(99, 110, 250, 0.2);
    outline: 1px solid rgba(99, 110, 250, 0.5);
  }
  .kernel-legend-list li.kernel-legend-dimmed {
    opacity: 0.4;
  }
  .theme-dark .kernel-legend-list li.kernel-legend-selected {
    background: rgba(99, 110, 250, 0.25);
  }
  h1 { font-size: 1.1rem; font-weight: 600; margin: 0; }
  .title-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0 0 12px 0;
    position: relative;
  }
  .info-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    margin-left: auto;
    border-radius: 50%;
    border: 1px solid var(--grid);
    background: var(--paper-bg);
    color: var(--fg);
    font-size: 13px;
    font-weight: 600;
    font-style: italic;
    font-family: Georgia, "Times New Roman", serif;
    line-height: 1;
    cursor: pointer;
    padding: 0;
    opacity: 0.85;
    transition: opacity 0.12s ease, background 0.12s ease;
  }
  .info-btn:hover, .info-btn[aria-expanded="true"] {
    opacity: 1;
    background: rgba(99, 110, 250, 0.18);
    border-color: rgba(99, 110, 250, 0.7);
  }
  .info-popover {
    position: absolute;
    top: calc(100% + 6px);
    right: 0;
    z-index: 60;
    max-width: 460px;
    background: var(--tooltip-bg);
    color: var(--tooltip-fg);
    border: 1px solid var(--tooltip-border);
    border-radius: 6px;
    padding: 10px 14px 12px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    font-size: 12px;
    line-height: 1.4;
  }
  .info-popover[hidden] { display: none; }
  .info-popover h3 {
    margin: 0 0 6px 0;
    font-size: 0.85rem;
    font-weight: 600;
  }
  .info-popover ul {
    margin: 0;
    padding-left: 18px;
  }
  .info-popover li { margin: 3px 0; }
  .toolbar { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: center; margin-bottom: 12px; }
  .toolbar label { display: flex; flex-direction: column; gap: 4px; font-size: 0.85rem; }
  .toolbar select, .toolbar button { font-size: 0.9rem; padding: 4px 8px; }
  #chart-wrap svg { display: block; }
  .roofline-path { fill: none; stroke-linejoin: round; pointer-events: none; }
  .roofline-hit { fill: none; stroke: transparent; pointer-events: stroke; cursor: pointer; }
  .dot { stroke: rgba(0,0,0,0.25); stroke-width: 0.5; }
  .axis text { fill: var(--fg); font-size: 11px; }
  .axis line, .axis path { stroke: var(--axis); }
  .grid line { stroke: var(--grid); stroke-opacity: 0.9; }
  #disclaimer { font-size: 12px; margin-top: 8px; text-align: right; color: var(--fg); opacity: 0.85; }
  .chart-hint { font-size: 0.8rem; color: var(--fg); opacity: 0.75; margin: 0 0 8px 0; }
  .axis-title { fill: var(--fg); font-weight: 600; font-size: 13px; }
  .legend-box { font-size: 11px; }
  .legend-bg { fill: var(--plot-bg); stroke: var(--grid); stroke-width: 1; opacity: 0.94; }
  .legend-title { font-weight: 600; font-size: 11px; fill: var(--fg); }
  .legend-row-label { fill: var(--fg); font-size: 11px; }
  #tooltip {
    position: fixed; pointer-events: none; z-index: 50;
    background: var(--tooltip-bg); color: var(--tooltip-fg);
    border: 1px solid var(--tooltip-border); border-radius: 4px;
    padding: 8px 10px; font-size: 12px; line-height: 1.35; max-width: 420px;
    white-space: pre-line; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    display: none;
  }
</style>
</head>
<body>
<div id="roofline-app">
  <div class="title-row">
    <h1 id="chart-title"></h1>
    <button type="button" id="btn-plot-info" class="info-btn" aria-label="How to read this plot" aria-expanded="false" aria-controls="plot-info-popover" title="How to read this plot">i</button>
    <div id="plot-info-popover" class="info-popover" role="region" aria-labelledby="plot-info-heading" hidden>
      <h3 id="plot-info-heading">Implementation Details and Disclaimers</h3>
      <ul>
        <li><b>Peak compute:</b> The peak compute is based on the instruction mix of each kernel. It is the weighted average of the measured throughput of each instruction counted in the kernel. Clicking on or hovering the mouse over a kernel dot will show that kernel's empirical peak compute. If none is selected, the default peak compute displayed is FP64 MFMA.</li>
        <li><b>Total FLOPs:</b> The "flops" value reported includes floating point AND integer operations. These operations all go through the same VALU pipe, so they should be counted together. </li>
        <li><b>Curved rooflines:</b> HBM and LDS rooflines are drawn with a curved shape. This is because workloads that are demanding in bandwidth AND compute consume more power, and subsequently get their clocks throttled. So, the roofline peak is lower in the "knee" region, where the bandwidth and compute lines meet. </li>
        <li><b>Packed instructions:</b> Any packed instructions are counted as 1 operation by the profiler. If your workload uses packed instructions, the flops value may be underreported. </li>
        <li><b>Other VALU operations:</b> Some VALU operations do not have dedicated hardware counters, such as bitwise operations. These operations are counted as "Other VALU" in roofline extractor output. The throughput value of `v_lshlrev_b32_e32` is used for this category.</li>
        <li><b>Compute partitioning:</b> Roofline extractor assumes the application was run in SPX mode. </li>
        <li><b>Overlapping kernels:</b> If the application has multiple kernels running concurrently, e.g. using HIP streams, multiple kernels may be contending for the same hardware resources and thus unable to achieve their roofline peaks. </li>
      </ul>
    </div>
  </div>
  <div class="toolbar">
    <label>Memory region
      <select id="memory-region-select" aria-label="Memory hierarchy for arithmetic intensity"></select>
    </label>
    <label>View
      <select id="view-type-select" aria-label="Aggregate or per-dispatch kernels">
        <option value="aggregate" selected>Aggregate</option>
        <option value="individual">Individual (dispatches)</option>
      </select>
    </label>
    <label>Total percent runtime displayed
      <input type="range" id="threshold-slider" min="0" max="0" value="0" step="1"/>
      <span id="threshold-label"></span>
    </label>
    <label>&nbsp;
      <span style="display:flex; gap:8px; flex-wrap:wrap;">
        <button type="button" id="btn-theme-toggle" aria-pressed="true" title="Switch to light theme">Light mode</button>
        <button type="button" id="btn-hbm-lds-roofline-toggle" hidden aria-pressed="true" title="Use piecewise-linear roofs for HBM and LDS">Linear HBM/LDS</button>
        <button type="button" id="btn-reset-zoom" title="Reset axes to default range">Reset zoom</button>
        <button type="button" id="btn-export-png" title="Download the current plot as a PNG image">Export PNG</button>
      </span>
    </label>
  </div>
  <p class="chart-hint">Scroll to zoom, drag to pan (double-click chart to reset).</p>
  <div class="chart-row">
    <div id="chart-wrap"></div>
    <aside class="kernel-legend-panel" aria-label="Kernel legends">
      <div class="kernel-legend-header">
        <h2 class="kernel-legend-heading" id="kernel-legend-heading">Kernels</h2>
        <button type="button" class="kernel-legend-clear" id="kernel-legend-clear" hidden>Show all kernels</button>
      </div>
      <p class="kernel-legend-help" id="kernel-legend-help">Click a row to show only that kernel; click again to show all. Ctrl+click (⌘+click on Mac) to add or remove kernels.</p>
      <ul class="kernel-legend-list" id="kernel-legend-list" role="list"></ul>
    </aside>
  </div>
  <div id="disclaimer"></div>
</div>
<script type="application/json" id="roofline-json">__PAYLOAD__</script>
<script>
(function() {
  const data = JSON.parse(document.getElementById("roofline-json").textContent);
  const meta = data.meta;
  const cacheKeys = data.cacheKeys;
  // Memory regions ordered from closest to compute to furthest away. When a single
  // kernel is plotted against all regions, dots are drawn in this order so that
  // further-away regions (e.g. HBM) are painted on top of closer ones when they overlap.
  const MEM_DISTANCE_ORDER = ["LDS", "vL1d", "L2", "HBM"];
  function memDistanceRank(key) {
    const i = MEM_DISTANCE_ORDER.indexOf(key);
    return i < 0 ? -1 : i;
  }
  const rooflinePalette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"];
  function rooflineColorForIndex(i) { return rooflinePalette[i % rooflinePalette.length]; }
  function rooflineColorForKey(key) {
    const idx = cacheKeys.indexOf(key);
    return idx < 0 ? rooflinePalette[0] : rooflineColorForIndex(idx);
  }
  /** When false, HBM/LDS use linear min(AI×B, P) even if α model metadata is present. */
  let useCurvedHbmLdsRooflines = true;
  document.getElementById("chart-title").textContent = meta.title;
  const disc = document.getElementById("disclaimer");
  disc.textContent = meta.disclaimer || "";

  let W = 960, H = 560;
  const margin = { top: 36, right: 36, bottom: 62, left: 86 };
  let iw = W - margin.left - margin.right;
  let ih = H - margin.top - margin.bottom;

  const x0 = d3.scaleLog().domain([meta.xMin, meta.xMax]).range([0, iw])
  const y0 = d3.scaleLog().domain([meta.yMin, meta.yMax]).range([ih, 0])
  let transform = d3.zoomIdentity;
  function currentScales() {
    return { x: transform.rescaleX(x0), y: transform.rescaleY(y0) };
  }

  const svg = d3.select("#chart-wrap").append("svg")
    .attr("width", W).attr("height", H).attr("viewBox", "0 0 " + W + " " + H);

  svg.append("defs").append("clipPath").attr("id", "roofline-plot-clip")
    .append("rect").attr("id", "roofline-clip-rect").attr("width", iw).attr("height", ih);
  const clipRect = svg.select("#roofline-clip-rect");

  const g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  const xGridG = g.append("g").attr("class", "grid");
  const yGridG = g.append("g").attr("class", "grid");

  const tooltip = d3.select("body").append("div").attr("id", "tooltip");

  function rooflineTooltipHtml(cacheKey) {
    const nl = String.fromCharCode(10);
    const hbmAlphaOn = useCurvedHbmLdsRooflines && meta.useHbmAlphaModel === true && cacheKey === "HBM" && hbmAlphaForRoofline() != null;
    const ldsAlphaOn = useCurvedHbmLdsRooflines && meta.useLdsAlphaModel === true && cacheKey === "LDS" && ldsAlphaForRoofline() != null;
    const curvedOn = hbmAlphaOn || ldsAlphaOn;
    const curvedBand = hbmAlphaOn
      ? (" (" + (meta.hbmAlphaArch || "") + " HBM)")
      : ldsAlphaOn
        ? (" (" + (meta.ldsAlphaArch || "") + " LDS)")
        : "";
    const lines = [
      cacheKey + " bandwidth roofline",
      curvedOn
        ? ("Model: (1/P + 1/(AI×B) − α·min(1/P, 1/(AI×B)))⁻¹; P=compute, B=bandwidth" + curvedBand + ".")
        : "Model: throughput = min(bandwidth × AI, compute peak)."
    ];
    const bw = data.bandwidths && data.bandwidths[cacheKey];
    if (bw != null && Number.isFinite(bw)) {
      const bwTbs = bw / 1000;
      lines.push("Bandwidth (slope): " + bwTbs.toLocaleString(undefined, { maximumFractionDigits: 4 }) + " TB/s");
    }
    const cp = effectiveComputePeak();
    if (cp != null && Number.isFinite(cp)) {
      lines.push("Compute peak (flat roof): " + cp.toLocaleString(undefined, { maximumFractionDigits: 2 }) + " GFLOPs/s");
    }
    if (hbmAlphaOn && cacheKey === "HBM") {
      if (kernelHbmAlpha() != null) {
        lines.push("α: instruction-mix weighted (hovered or selected kernel).");
      } else if (meta.defaultHbmAlphaFp64Mfma != null && Number.isFinite(meta.defaultHbmAlphaFp64Mfma)) {
        lines.push("α: FP64 MFMA microbenchmark (100% MFMA default).");
      }
      lines.push(computeRoofFlatDisclaimer());
    } else if (ldsAlphaOn && cacheKey === "LDS") {
      if (kernelLdsAlpha() != null) {
        lines.push("α: instruction-mix weighted (hovered or selected kernel).");
      } else if (meta.defaultLdsAlphaFp64Mfma != null && Number.isFinite(meta.defaultLdsAlphaFp64Mfma)) {
        lines.push("α: FP64 MFMA microbenchmark (100% MFMA default).");
      }
      lines.push(computeRoofFlatDisclaimer());
    } else if (roofHoverDispatchComputePeak != null) {
      lines.push("(Instruction-mix ceiling for hovered dispatch)");
    } else if (roofHoverKernelName != null) {
      lines.push("(Instruction-mix ceiling for hovered kernel)");
    } else if (kernelFilterActive() && selectedKernelNames.size === 1) {
      lines.push("(Instruction-mix ceiling for selected kernel)");
    } else {
      lines.push("Compute peak is currently drawn according to FP64 MFMA, not kernel-specific instruction mix.");
    }
    return lines.join(nl);
  }

  const plotInner = g.append("g").attr("clip-path", "url(#roofline-plot-clip)");
  const roofGroup = plotInner.append("g").attr("class", "rooflines");
  data.rooflines.forEach((rl, i) => {
    const pts = rl.x.map((xv, j) => [xv, rl.y[j]]);
    roofGroup.append("path")
      .datum(pts)
      .attr("class", "roofline-path")
      .attr("data-idx", i)
      .attr("stroke", rooflineColorForIndex(i));
    roofGroup.append("path")
      .datum(pts)
      .attr("class", "roofline-hit")
      .attr("data-idx", i)
      .attr("data-cache-key", rl.key)
      .attr("stroke-width", 16);
  });
  roofGroup.selectAll(".roofline-hit")
    .on("mouseenter", function() {
      const key = d3.select(this).attr("data-cache-key");
      tooltip.style("display", "block").text(rooflineTooltipHtml(key));
    })
    .on("mousemove", function(event) {
      tooltip.style("left", (event.clientX + 14) + "px").style("top", (event.clientY + 14) + "px");
    })
    .on("mouseleave", function() {
      tooltip.style("display", "none");
    })
    .on("click", function(event) {
      event.stopPropagation();
      const key = d3.select(this).attr("data-cache-key");
      tooltip.style("display", "none");
      onRooflineLegendActivate(event, key);
    });

  const dotLayer = plotInner.append("g").attr("class", "dots");

  const gx = g.append("g").attr("class", "axis x-axis").attr("transform", "translate(0," + ih + ")");
  const gy = g.append("g").attr("class", "axis y-axis");

  const xAxisTitleEl = g.append("text").attr("class", "axis-title").attr("text-anchor", "middle")
    .attr("x", iw / 2).attr("y", ih + 44).text(meta.xAxisTitle);
  const yAxisTitleEl = g.append("text").attr("class", "axis-title").attr("text-anchor", "middle")
    .attr("transform", "rotate(-90)").attr("x", -ih / 2).attr("y", -62).text(meta.yAxisTitle);

  const legW = 172;
  const legH = 26 + cacheKeys.length * 16 + 10;
  function legendTopY() { return ih - legH - 4; }
  const legendG = g.append("g").attr("class", "legend-box").attr("transform", "translate(" + (iw - legW - 4) + "," + legendTopY() + ")");
  legendG.append("rect").attr("class", "legend-bg").attr("width", legW).attr("height", legH).attr("rx", 5);
  legendG.append("text").attr("class", "legend-title").attr("x", 8).attr("y", 16).text("Achievable peak (by memory)");
  const legRows = legendG.selectAll("g.lrow").data(cacheKeys).enter().append("g")
    .attr("class", "lrow")
    .attr("transform", (d, i) => "translate(8," + (24 + i * 16) + ")");
  legRows.append("line").attr("x1", 0).attr("x2", 28).attr("y1", 0).attr("y2", 0)
    .attr("stroke", (d, i) => rooflineColorForIndex(i)).attr("class", "leg-line");
  legRows.append("text").attr("class", "legend-row-label").attr("x", 34).attr("y", 4);

  const kLeg = data.kernelLegend || [];
  const kUl = document.getElementById("kernel-legend-list");
  const kHead = document.getElementById("kernel-legend-heading");
  const kClear = document.getElementById("kernel-legend-clear");
  const selectedKernelNames = new Set();
  const selectedRooflineKeys = new Set();
  let roofHoverKernelName = null;
  /** When Individual (dispatches) view: instruction-mix ceiling for the hovered dispatch (not kernel aggregate). */
  let roofHoverDispatchComputePeak = null;
  /** Kernel dispatch row currently showing the dot tooltip (for refresh after curved/linear toggle). */
  let dotTooltipDatum = null;

  function kernelFilterActive() {
    return selectedKernelNames.size > 0;
  }

  function rooflineFilterActive() {
    return selectedRooflineKeys.size > 0;
  }

  function effectiveComputePeak() {
    if (roofHoverDispatchComputePeak != null && Number.isFinite(roofHoverDispatchComputePeak) && roofHoverDispatchComputePeak > 0) {
      return roofHoverDispatchComputePeak;
    }
    let cp = meta.computePeak;
    let mixName = null;
    if (roofHoverKernelName != null) {
      mixName = roofHoverKernelName;
    } else if (kernelFilterActive() && selectedKernelNames.size === 1) {
      mixName = Array.from(selectedKernelNames)[0];
    }
    if (mixName != null) {
      const kp = data.kernelComputePeakByName && data.kernelComputePeakByName[mixName];
      if (kp != null && Number.isFinite(kp) && kp > 0) {
        cp = kp;
      }
    }
    return cp;
  }

  /** Matches effectiveComputePeak(): text for the flat compute roof P in tooltips (incl. HBM/LDS α model). */
  function computeRoofFlatDisclaimer() {
    if (roofHoverDispatchComputePeak != null && Number.isFinite(roofHoverDispatchComputePeak) && roofHoverDispatchComputePeak > 0) {
      return "(Compute roof: instruction-mix ceiling for hovered dispatch.)";
    }
    let mixName = null;
    if (roofHoverKernelName != null) {
      mixName = roofHoverKernelName;
    } else if (kernelFilterActive() && selectedKernelNames.size === 1) {
      mixName = Array.from(selectedKernelNames)[0];
    }
    if (mixName != null) {
      const kp = data.kernelComputePeakByName && data.kernelComputePeakByName[mixName];
      if (kp != null && Number.isFinite(kp) && kp > 0) {
        if (roofHoverKernelName != null) {
          return "(Compute roof: instruction-mix ceiling for hovered kernel.)";
        }
        return "(Compute roof: instruction-mix ceiling for selected kernel.)";
      }
    }
    return "(Compute roof: FP64 MFMA baseline, or kernel mix when hovered/selected.)";
  }

  function kernelHbmAlpha() {
    let mixName = null;
    if (roofHoverKernelName != null) {
      mixName = roofHoverKernelName;
    } else if (kernelFilterActive() && selectedKernelNames.size === 1) {
      mixName = Array.from(selectedKernelNames)[0];
    }
    if (mixName != null && data.kernelHbmAlphaByName) {
      const o = data.kernelHbmAlphaByName[mixName];
      if (o != null && Number.isFinite(o.alpha)) {
        return o;
      }
    }
    return null;
  }

  function kernelLdsAlpha() {
    let mixName = null;
    if (roofHoverKernelName != null) {
      mixName = roofHoverKernelName;
    } else if (kernelFilterActive() && selectedKernelNames.size === 1) {
      mixName = Array.from(selectedKernelNames)[0];
    }
    if (mixName != null && data.kernelLdsAlphaByName) {
      const o = data.kernelLdsAlphaByName[mixName];
      if (o != null && Number.isFinite(o.alpha)) {
        return o;
      }
    }
    return null;
  }

  function hbmAlphaForRoofline() {
    const k = kernelHbmAlpha();
    if (k != null) {
      return k;
    }
    if (meta.useHbmAlphaModel === true && meta.defaultHbmAlphaFp64Mfma != null && Number.isFinite(meta.defaultHbmAlphaFp64Mfma)) {
      return { alpha: meta.defaultHbmAlphaFp64Mfma };
    }
    return null;
  }

  function ldsAlphaForRoofline() {
    const k = kernelLdsAlpha();
    if (k != null) {
      return k;
    }
    if (meta.useLdsAlphaModel === true && meta.defaultLdsAlphaFp64Mfma != null && Number.isFinite(meta.defaultLdsAlphaFp64Mfma)) {
      return { alpha: meta.defaultLdsAlphaFp64Mfma };
    }
    return null;
  }

  function hbmRoofThroughput(x, bw, cp, alpha) {
    const eps = 1e-30;
    const xs = Math.max(x, eps);
    if (!(bw > 0) || !(cp > 0)) {
      return NaN;
    }
    const linear = Math.min(bw * xs, cp);
    const a = Number(alpha);
    if (!Number.isFinite(a)) {
      return linear;
    }
    const invP = 1 / cp;
    const invXb = 1 / (xs * bw);
    const m = Math.min(invP, invXb);
    const denom = invP + invXb - a * m;
    if (!Number.isFinite(denom) || denom <= 1e-30) {
      return linear;
    }
    const raw = 1 / denom;
    if (!Number.isFinite(raw) || raw <= 0) {
      return linear;
    }
    return raw;
  }

  function logspacePos(lo, hi, n) {
    if (!(lo > 0) || !(hi > 0) || !isFinite(lo) || !isFinite(hi)) return [];
    let a = lo;
    let b = hi;
    if (b < a) { const t = a; a = b; b = t; }
    if (b <= a * (1 + 1e-15)) return [a];
    const la = Math.log10(a);
    const lb = Math.log10(b);
    const steps = Math.max(2, Math.min(n | 0, 256));
    const out = [];
    for (let i = 0; i < steps; i++) {
      const u = i / (steps - 1);
      out.push(Math.pow(10, la + u * (lb - la)));
    }
    return out;
  }

  function rooflineXSamplesForView(xScale, bw, cp) {
    const r = xScale.range();
    const pLo = Math.min(r[0], r[1]);
    const pHi = Math.max(r[0], r[1]);
    let xLo = xScale.invert(pLo);
    let xHi = xScale.invert(pHi);
    const eps = 1e-300;
    if (!(xLo > 0) || !isFinite(xLo)) xLo = eps;
    if (!(xHi > 0) || !isFinite(xHi)) xHi = eps;
    if (xHi < xLo) { const t = xLo; xLo = xHi; xHi = t; }
    const xKnee = cp / bw;
    const nSeg = 120;
    const xsSet = new Set();
    logspacePos(xLo, xHi, nSeg).forEach(v => xsSet.add(v));
    xsSet.add(xLo);
    xsSet.add(xHi);
    if (xKnee > 0 && isFinite(xKnee) && xKnee > xLo && xKnee < xHi) xsSet.add(xKnee);
    return Array.from(xsSet).filter(v => v > 0 && isFinite(v)).sort((a, b) => a - b);
  }

  function rooflinePointsForIndex(i, xScale) {
    const rl = data.rooflines[i];
    const key = rl.key != null ? rl.key : cacheKeys[i];
    const bw = data.bandwidths[key];
    const cp = effectiveComputePeak();
    if (!rl || !rl.x) {
      return [];
    }
    if (bw == null || !Number.isFinite(bw) || cp == null || !Number.isFinite(cp)) {
      return rl.x.map((xv, j) => [xv, rl.y[j]]);
    }
    if (!(bw > 0) || !(cp > 0)) {
      return rl.x.map((xv, j) => [xv, rl.y[j]]);
    }
    const curvedAlpha =
      useCurvedHbmLdsRooflines && meta.useHbmAlphaModel === true && key === "HBM"
        ? hbmAlphaForRoofline()
        : useCurvedHbmLdsRooflines && meta.useLdsAlphaModel === true && key === "LDS"
          ? ldsAlphaForRoofline()
          : null;
    if (curvedAlpha != null) {
      const xs = rooflineXSamplesForView(xScale, bw, cp);
      return xs.map(function(xv) {
        return [xv, hbmRoofThroughput(xv, bw, cp, curvedAlpha.alpha)];
      });
    }
    const xs = rooflineXSamplesForView(xScale, bw, cp);
    return xs.map(xv => [xv, Math.min(bw * xv, cp)]);
  }

  function updateRooflinePaths(lineGen, xScale) {
    roofGroup.selectAll(".roofline-path").each(function() {
      const i = +d3.select(this).attr("data-idx");
      const pts = rooflinePointsForIndex(i, xScale);
      d3.select(this).datum(pts).attr("d", lineGen(pts));
    });
    roofGroup.selectAll(".roofline-hit").each(function() {
      const i = +d3.select(this).attr("data-idx");
      const pts = rooflinePointsForIndex(i, xScale);
      d3.select(this).datum(pts).attr("d", lineGen(pts));
    });
  }

  function computeKernelsVisibleAtThreshold() {
    const t = th[thresholdIndex];
    const key = regionKey();
    const agg = currentMode().aggregate;
    const src = agg ? data.aggregate : data.dispatch;
    const visible = new Set();
    for (let i = 0; i < src.length; i++) {
      const d = src[i];
      const c = d.cumulativePct;
      if (c == null || !Number.isFinite(c) || c > t) continue;
      const a = d.ai[key];
      if (a == null || !(a > 0) || !Number.isFinite(a)) continue;
      const tp = d.throughput;
      if (tp == null || !(tp > 0) || !Number.isFinite(tp)) continue;
      visible.add(d.kernelName);
    }
    return visible;
  }

  function updateKernelLegendUI() {
    const visibleKernels = computeKernelsVisibleAtThreshold();
    const totalCount = kLeg.length;
    const shownCount = visibleKernels.size;
    if (kHead) {
      let t = shownCount < totalCount
        ? "Kernels (" + shownCount + " / " + totalCount + ")"
        : "Kernels (" + totalCount + ")";
      if (kernelFilterActive()) {
        t += " — " + selectedKernelNames.size + " selected";
      }
      kHead.textContent = t;
    }
    if (kClear) kClear.hidden = !kernelFilterActive();
    if (kUl) {
      kUl.querySelectorAll("li").forEach(function(li) {
        const name = li._kernelName;
        if (!name) return;
        const inThreshold = visibleKernels.has(name);
        li.style.display = inThreshold ? "" : "none";
        const on = selectedKernelNames.has(name);
        const active = kernelFilterActive();
        li.classList.toggle("kernel-legend-selected", on);
        li.classList.toggle("kernel-legend-dimmed", active && !on);
      });
    }
  }

  function onKernelLegendActivate(event, name) {
    if (event.ctrlKey || event.metaKey) {
      if (!kernelFilterActive()) {
        kLeg.forEach(function(k) { selectedKernelNames.add(k.name); });
        selectedKernelNames.delete(name);
      } else if (selectedKernelNames.has(name)) {
        selectedKernelNames.delete(name);
      } else {
        selectedKernelNames.add(name);
      }
    } else {
      if (selectedKernelNames.size === 1 && selectedKernelNames.has(name)) {
        selectedKernelNames.clear();
      } else {
        selectedKernelNames.clear();
        selectedKernelNames.add(name);
      }
    }
    updateKernelLegendUI();
    redraw();
  }

  function onRooflineLegendActivate(event, key) {
    if (event.ctrlKey || event.metaKey) {
      if (!rooflineFilterActive()) {
        cacheKeys.forEach(function(k) { selectedRooflineKeys.add(k); });
        selectedRooflineKeys.delete(key);
      } else if (selectedRooflineKeys.has(key)) {
        selectedRooflineKeys.delete(key);
      } else {
        selectedRooflineKeys.add(key);
      }
    } else {
      if (selectedRooflineKeys.size === 1 && selectedRooflineKeys.has(key)) {
        selectedRooflineKeys.clear();
      } else {
        selectedRooflineKeys.clear();
        selectedRooflineKeys.add(key);
      }
    }
    redraw();
  }

  kLeg.forEach(function(k) {
    const li = document.createElement("li");
    li._kernelName = k.name;
    li.setAttribute("role", "listitem");
    li.tabIndex = 0;
    const sw = document.createElement("span");
    sw.className = "kernel-swatch";
    sw.style.backgroundColor = k.color;
    sw.setAttribute("aria-hidden", "true");
    const lab = document.createElement("span");
    lab.className = "kernel-name";
    lab.textContent = k.name;
    lab.title = k.name;
    const pct = document.createElement("span");
    pct.className = "kernel-pct";
    pct.textContent = (k.pct != null && Number.isFinite(k.pct)) ? (" " + k.pct.toFixed(2) + "%") : "";
    li.appendChild(sw);
    li.appendChild(lab);
    li.appendChild(pct);
    li.addEventListener("click", function(ev) {
      onKernelLegendActivate(ev, k.name);
    });
    li.addEventListener("keydown", function(ev) {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        onKernelLegendActivate(ev, k.name);
      }
    });
    kUl.appendChild(li);
  });

  if (kClear) {
    kClear.addEventListener("click", function() {
      selectedKernelNames.clear();
      updateKernelLegendUI();
      redraw();
    });
  }

  legRows.style("cursor", "pointer")
    .on("click", function(event, d) {
      onRooflineLegendActivate(event, d);
    });

  let thresholdIndex = 0;

  const memSel = document.getElementById("memory-region-select");
  cacheKeys.forEach((k, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = k;
    memSel.appendChild(opt);
  });
  memSel.value = "0";

  const viewSel = document.getElementById("view-type-select");
  viewSel.value = "aggregate";

  function currentMode() {
    const r = +memSel.value;
    const agg = viewSel.value === "aggregate";
    return {
      region: r,
      aggregate: agg,
      label: cacheKeys[r] + (agg ? " Agg" : "")
    };
  }

  const th = data.thresholds;
  const slider = document.getElementById("threshold-slider");
  slider.min = "0";
  slider.max = String(Math.max(0, th.length - 1));
  slider.value = "0";
  const thLabel = document.getElementById("threshold-label");

  updateKernelLegendUI();

  function isDarkTheme() {
    return document.documentElement.classList.contains("theme-dark");
  }
  function syncThemeToggle() {
    const btn = document.getElementById("btn-theme-toggle");
    const dark = isDarkTheme();
    btn.textContent = dark ? "Light mode" : "Dark mode";
    btn.setAttribute("aria-pressed", String(dark));
    btn.title = dark ? "Switch to light theme" : "Switch to dark theme";
  }
  function setTheme(dark) {
    const root = document.documentElement;
    root.classList.remove("theme-light", "theme-dark");
    root.classList.add(dark ? "theme-dark" : "theme-light");
    syncThemeToggle();
  }
  document.getElementById("btn-theme-toggle").addEventListener("click", () => setTheme(!isDarkTheme()));
  syncThemeToggle();

  (function setupPlotInfoPopover() {
    const btn = document.getElementById("btn-plot-info");
    const pop = document.getElementById("plot-info-popover");
    if (!btn || !pop) return;
    function setOpen(open) {
      pop.hidden = !open;
      btn.setAttribute("aria-expanded", String(open));
    }
    btn.addEventListener("click", function(ev) {
      ev.stopPropagation();
      setOpen(pop.hidden);
    });
    document.addEventListener("click", function(ev) {
      if (pop.hidden) return;
      if (ev.target === btn || btn.contains(ev.target)) return;
      if (pop.contains(ev.target)) return;
      setOpen(false);
    });
    document.addEventListener("keydown", function(ev) {
      if (ev.key === "Escape" && !pop.hidden) {
        setOpen(false);
        btn.focus();
      }
    });
  })();

  (function setupHbmLdsRooflineToggle() {
    const btn = document.getElementById("btn-hbm-lds-roofline-toggle");
    if (!btn) return;
    function hbmLdsAlphaModelsActive() {
      return meta.useHbmAlphaModel === true || meta.useLdsAlphaModel === true;
    }
    function syncHbmLdsRooflineToggle() {
      const on = hbmLdsAlphaModelsActive();
      btn.hidden = !on;
      if (!on) return;
      btn.setAttribute("aria-pressed", String(useCurvedHbmLdsRooflines));
      btn.textContent = "Toggle curved rooflines";
      btn.title = useCurvedHbmLdsRooflines
        ? "Use piecewise-linear roofs (min(AI×bandwidth, compute)) for HBM and LDS"
        : "Use α-blended curved roofs for HBM and LDS";
    }
    btn.addEventListener("click", function() {
      if (!hbmLdsAlphaModelsActive()) return;
      useCurvedHbmLdsRooflines = !useCurvedHbmLdsRooflines;
      syncHbmLdsRooflineToggle();
      redraw();
    });
    syncHbmLdsRooflineToggle();
  })();

  function regionKey() { return cacheKeys[currentMode().region]; }

  function visibleDataset() {
    const t = th[thresholdIndex];
    const key = regionKey();
    const agg = currentMode().aggregate;
    const src = agg ? data.aggregate : data.dispatch;
    return src.filter(d => {
      const c = d.cumulativePct;
      if (c == null || !Number.isFinite(c) || c > t) return false;
      if (kernelFilterActive() && !selectedKernelNames.has(d.kernelName)) return false;
      const a = d.ai[key];
      if (a == null || !(a > 0) || !Number.isFinite(a)) return false;
      const tp = d.throughput;
      if (tp == null || !(tp > 0) || !Number.isFinite(tp)) return false;
      return true;
    });
  }

  function updateRooflineWidths() {
    const r = currentMode().region;
    const filterOn = rooflineFilterActive();
    roofGroup.selectAll(".roofline-path").each(function() {
      const idx = +d3.select(this).attr("data-idx");
      const key = cacheKeys[idx];
      const inSel = !filterOn || selectedRooflineKeys.has(key);
      const baseW = idx === r ? 3 : 1;
      const sel = d3.select(this);
      if (!inSel) {
        sel.attr("stroke-width", 1).attr("opacity", 0.18);
      } else {
        sel.attr("stroke-width", baseW).attr("opacity", 1);
      }
    });
    roofGroup.selectAll(".roofline-hit").each(function() {
      const idx = +d3.select(this).attr("data-idx");
      const key = cacheKeys[idx];
      const inSel = !filterOn || selectedRooflineKeys.has(key);
      d3.select(this).attr("pointer-events", inSel ? "stroke" : "none");
    });
  }

  function updateLegend() {
    const r = currentMode().region;
    const filterOn = rooflineFilterActive();
    legendG.selectAll("g.lrow").each(function(d, i) {
      const key = d;
      const inSel = !filterOn || selectedRooflineKeys.has(key);
      const line = d3.select(this).select("line");
      const text = d3.select(this).select("text");
      if (!inSel) {
        line.attr("stroke-width", 1).attr("opacity", 0.35);
        text.attr("opacity", 0.45);
      } else {
        line.attr("opacity", 1).attr("stroke-width", i === r ? 3 : 1);
        text.attr("opacity", 1);
      }
      text.text(d + (i === r ? " (AI axis)" : ""));
    });
    let personalTitle = false;
    if (roofHoverDispatchComputePeak != null && Number.isFinite(roofHoverDispatchComputePeak) && roofHoverDispatchComputePeak > 0) {
      personalTitle = true;
    } else if (roofHoverKernelName != null) {
      const kp = data.kernelComputePeakByName && data.kernelComputePeakByName[roofHoverKernelName];
      personalTitle = kp != null && Number.isFinite(kp) && kp > 0;
    } else if (kernelFilterActive() && selectedKernelNames.size === 1) {
      const n = Array.from(selectedKernelNames)[0];
      const kp = data.kernelComputePeakByName && data.kernelComputePeakByName[n];
      personalTitle = kp != null && Number.isFinite(kp) && kp > 0;
    }
    legendG.select(".legend-title").text(
      personalTitle ? "Achievable peak (instruction-mix roof)" : "Achievable peak (by memory)"
    );
  }

  /** Peak at kernel AI matching chart rooflines: curved (PEAK) vs linear linear HBM/LDS (PEAK_LINEAR). */
  function tooltipPeakForDot(d) {
    const alphaCharts =
      meta.useHbmAlphaModel === true || meta.useLdsAlphaModel === true;
    if (alphaCharts && !useCurvedHbmLdsRooflines) {
      const pl = d.peakLinear;
      if (pl != null && Number.isFinite(pl)) {
        return pl;
      }
    }
    return d.peak;
  }

  function tooltipHtml(d) {
    const key = d.memRegion || regionKey();
    const rawAi = d.ai[key];
    const aiVal = rawAi != null && Number.isFinite(rawAi) ? rawAi : "N/A";
    const aiLabel = d.memRegion ? ("AI (" + d.memRegion + "): ") : "AI: ";
    const m = currentMode();
    const nl = String.fromCharCode(10);
    const peakVal = tooltipPeakForDot(d);
    const peakStr = Number.isFinite(peakVal) ? peakVal.toFixed(3) : "N/A";
    // Recompute percent against the currently shown roof so curved/linear toggle is reflected.
    const pctRoof = (Number.isFinite(peakVal) && peakVal > 0
                     && d.throughput != null && Number.isFinite(d.throughput))
      ? (d.throughput / peakVal * 100).toFixed(4) + " %"
      : "N/A";
    const alphaCharts = meta.useHbmAlphaModel === true || meta.useLdsAlphaModel === true;
    const limiterStr = (alphaCharts && !useCurvedHbmLdsRooflines && d.limiterLinear)
      ? d.limiterLinear
      : d.limiter;
    const lines = m.aggregate ? [
      "Name: " + d.nameDisplay,
      aiLabel + aiVal,
      "Achieved throughput: " + d.throughput.toFixed(3) + " GFLOPs/s",
      "Peak throughput: " + peakStr + " GFLOPs/s",
      "Percent of roofline achieved: " + pctRoof,
      "Performance limiter: " + limiterStr,
      "Total dispatches: " + d.count,
      "Aggregate percent runtime: " + d.percentage.toFixed(5) + " %"
    ] : [
      "Name: " + d.nameDisplay,
      "Index: " + d.index + " / " + d.totalKernels,
      aiLabel + aiVal,
      "Achieved throughput: " + d.throughput.toFixed(3) + " GFLOPs/s",
      "Peak throughput: " + peakStr + " GFLOPs/s",
      "Percent of roofline achieved: " + pctRoof,
      "Performance limiter: " + limiterStr,
      "Aggregate percent runtime: " + d.percentageAggregate.toFixed(5) + " %",
      "Individual percent runtime: " + d.percentage.toFixed(5) + " %"
    ];
    return lines.join(nl);
  }

  function refreshDotTooltipIfNeeded() {
    if (dotTooltipDatum != null) {
      tooltip.text(tooltipHtml(dotTooltipDatum));
    }
  }

  function refreshRooflinesForDotHover() {
    const { x, y } = currentScales();
    const lineGen = d3.line()
      .defined(d => d[0] > 0 && d[1] > 0 && isFinite(d[0]) && isFinite(d[1]))
      .x(d => x(d[0])).y(d => y(d[1]))
      .curve(d3.curveLinear);
    updateRooflinePaths(lineGen, x);
    updateRooflineWidths();
    updateLegend();
  }

  function computeDotData() {
    const singleKernelSelected = kernelFilterActive() && selectedKernelNames.size === 1;
    if (!singleKernelSelected) {
      return visibleDataset().map(d => Object.assign({}, d, { memRegion: null, dotColor: d.color }));
    }
    const t = th[thresholdIndex];
    const agg = currentMode().aggregate;
    const src = agg ? data.aggregate : data.dispatch;
    const expanded = [];
    // Closest-to-furthest order so further-away regions are appended last and
    // therefore drawn on top of closer ones when their dots overlap.
    const orderedMemKeys = cacheKeys.slice().sort(function(a, b) {
      return memDistanceRank(a) - memDistanceRank(b);
    });
    src.forEach(function(d) {
      const c = d.cumulativePct;
      if (c == null || !Number.isFinite(c) || c > t) return;
      if (!selectedKernelNames.has(d.kernelName)) return;
      const tp = d.throughput;
      if (tp == null || !(tp > 0) || !Number.isFinite(tp)) return;
      orderedMemKeys.forEach(function(memKey) {
        const a = d.ai[memKey];
        if (a == null || !(a > 0) || !Number.isFinite(a)) return;
        expanded.push(Object.assign({}, d, {
          id: d.id + "||" + memKey,
          memRegion: memKey,
          dotColor: rooflineColorForKey(memKey),
        }));
      });
    });
    return expanded;
  }

  function drawDots(xFn, yFn) {
    const pts = computeDotData();
    const key = regionKey();
    const sel = dotLayer.selectAll("circle.dot").data(pts, d => d.id);
    sel.enter().append("circle")
      .attr("class", "dot")
      .attr("r", 4)
      .style("cursor", "pointer")
      .merge(sel)
      .attr("cx", d => xFn(d.ai[d.memRegion || key]))
      .attr("cy", d => yFn(d.throughput))
      .attr("fill", d => d.dotColor || d.color)
      .on("click", function(event, d) {
        event.stopPropagation();
        roofHoverKernelName = null;
        dotTooltipDatum = null;
        roofHoverDispatchComputePeak = null;
        tooltip.style("display", "none");
        onKernelLegendActivate(event, d.kernelName);
      })
      .on("mouseenter", function(event, d) {
        roofHoverKernelName = d.kernelName;
        dotTooltipDatum = d;
        if (!currentMode().aggregate && d.kernelComputePeak != null && Number.isFinite(d.kernelComputePeak) && d.kernelComputePeak > 0) {
          roofHoverDispatchComputePeak = d.kernelComputePeak;
        } else {
          roofHoverDispatchComputePeak = null;
        }
        refreshRooflinesForDotHover();
        tooltip.style("display", "block").text(tooltipHtml(d));
      })
      .on("mousemove", function(event) {
        tooltip.style("left", (event.clientX + 14) + "px").style("top", (event.clientY + 14) + "px");
      })
      .on("mouseleave", function() {
        roofHoverKernelName = null;
        dotTooltipDatum = null;
        roofHoverDispatchComputePeak = null;
        refreshRooflinesForDotHover();
        tooltip.style("display", "none");
      })
      // Reorder DOM nodes to match data order (closest first) so that, on updates,
      // further-away memory regions stay painted on top of closer overlapping dots.
      .order();
    sel.exit().remove();
  }

  function updateThresholdLabel() {
    thLabel.textContent = th[thresholdIndex].toFixed(3) + "%";
  }

  function applyView() {
    const { x, y } = currentScales();
    gx.call(d3.axisBottom(x).ticks(8, "~s"));
    gy.call(d3.axisLeft(y).ticks(8, "~s"));
    xGridG.attr("transform", "translate(0," + ih + ")")
      .call(d3.axisBottom(x).ticks(8, "~s").tickSize(-ih).tickFormat(""))
      .call(z => z.select(".domain").remove());
    yGridG.call(d3.axisLeft(y).ticks(8, "~s").tickSize(-iw).tickFormat(""))
      .call(z => z.select(".domain").remove());

    const lineGen = d3.line()
      .defined(d => d[0] > 0 && d[1] > 0 && isFinite(d[0]) && isFinite(d[1]))
      .x(d => x(d[0])).y(d => y(d[1]))
      .curve(d3.curveLinear);
    updateRooflinePaths(lineGen, x);

    updateRooflineWidths();
    drawDots(x, y);
    updateLegend();
    updateThresholdLabel();
    refreshDotTooltipIfNeeded();
  }

  function redraw() {
    const { x, y } = currentScales();
    const lineGen = d3.line()
      .defined(d => d[0] > 0 && d[1] > 0 && isFinite(d[0]) && isFinite(d[1]))
      .x(d => x(d[0])).y(d => y(d[1]))
      .curve(d3.curveLinear);
    updateRooflinePaths(lineGen, x);
    updateRooflineWidths();
    const m = currentMode();
    disc.style.display = (!m.aggregate && meta.disclaimer) ? "block" : "none";
    drawDots(x, y);
    updateLegend();
    updateKernelLegendUI();
    updateThresholdLabel();
    refreshDotTooltipIfNeeded();
  }

  const zoom = d3.zoom()
    .scaleExtent([0.12, 100])
    .extent([[margin.left, margin.top], [margin.left + iw, margin.top + ih]])
    .on("zoom", (event) => {
      transform = event.transform;
      applyView();
    });

  const chartWrapEl = document.getElementById("chart-wrap");
  let lastLayoutW = -1;
  let lastLayoutH = -1;

  function layoutChart() {
    let nw = chartWrapEl.clientWidth;
    let nh = chartWrapEl.clientHeight;
    if (nw < 100) nw = Math.min(960, Math.max(400, window.innerWidth - 48));
    if (nh < 100) nh = Math.min(640, Math.max(320, window.innerHeight - 220));
    if (nw === lastLayoutW && nh === lastLayoutH) return;
    lastLayoutW = nw;
    lastLayoutH = nh;
    W = Math.max(280, nw);
    H = Math.max(240, nh);
    iw = W - margin.left - margin.right;
    ih = H - margin.top - margin.bottom;
    x0.range([0, iw]);
    y0.range([ih, 0]);
    svg.attr("width", W).attr("height", H).attr("viewBox", "0 0 " + W + " " + H);
    clipRect.attr("width", iw).attr("height", ih);
    gx.attr("transform", "translate(0," + ih + ")");
    xAxisTitleEl.attr("x", iw / 2).attr("y", ih + 44);
    yAxisTitleEl.attr("x", -ih / 2);
    legendG.attr("transform", "translate(" + (iw - legW - 4) + "," + legendTopY() + ")");
    zoom.extent([[margin.left, margin.top], [margin.left + iw, margin.top + ih]]);
    transform = d3.zoomIdentity;
    svg.call(zoom);
    applyView();
  }

  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(function() { layoutChart(); }).observe(chartWrapEl);
  } else {
    window.addEventListener("resize", layoutChart);
  }
  requestAnimationFrame(layoutChart);

  function resetZoom() {
    svg.transition().duration(200).call(zoom.transform, d3.zoomIdentity);
  }

  document.getElementById("btn-reset-zoom").addEventListener("click", resetZoom);
  g.on("dblclick", function(event) {
    event.preventDefault();
    resetZoom();
  });

  function sanitizeFilename(name) {
    return (name || "roofline").replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "") || "roofline";
  }
  function exportPng() {
    const btn = document.getElementById("btn-export-png");
    const svgNode = document.querySelector("#chart-wrap svg");
    if (!svgNode) return;
    const prevLabel = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Exporting...";
    // Resolve themed CSS variables to concrete colors so the rasterized SVG
    // (which is rendered detached from the document) still picks up the right theme.
    const cs = getComputedStyle(document.documentElement);
    const cssVar = (name, fallback) => (cs.getPropertyValue(name) || "").trim() || fallback;
    const plotBg = cssVar("--plot-bg", "#ffffff");
    const fg = cssVar("--fg", "#111111");
    const axisColor = cssVar("--axis", "#000000");
    const gridColor = cssVar("--grid", "#d3d3d3");

    const clone = svgNode.cloneNode(true);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
    const widthAttr = parseFloat(svgNode.getAttribute("width")) || svgNode.clientWidth || W;
    const heightAttr = parseFloat(svgNode.getAttribute("height")) || svgNode.clientHeight || H;
    clone.setAttribute("width", widthAttr);
    clone.setAttribute("height", heightAttr);
    if (!clone.getAttribute("viewBox")) {
      clone.setAttribute("viewBox", "0 0 " + widthAttr + " " + heightAttr);
    }

    // Inline the SVG-relevant rules from the document <head>. The cloned SVG
    // is rasterized detached from the page, so these rules (and the CSS
    // custom properties they depend on) would otherwise be lost — most
    // visibly turning every <path class="roofline-path"> into a black-filled
    // region under the curve.
    const styleEl = document.createElementNS("http://www.w3.org/2000/svg", "style");
    styleEl.textContent = [
      "text { font-family: system-ui, sans-serif; fill: " + fg + "; }",
      ".roofline-path { fill: none; stroke-linejoin: round; }",
      ".roofline-hit { fill: none; stroke: transparent; }",
      ".dot { stroke: rgba(0,0,0,0.25); stroke-width: 0.5; }",
      ".axis text { fill: " + fg + "; font-size: 11px; }",
      ".axis line, .axis path { fill: none; stroke: " + axisColor + "; }",
      ".grid line { stroke: " + gridColor + "; stroke-opacity: 0.9; }",
      ".axis-title { fill: " + fg + "; font-weight: 600; font-size: 13px; }",
      ".legend-box { font-size: 11px; }",
      ".legend-bg { fill: " + plotBg + "; stroke: " + gridColor + "; stroke-width: 1; opacity: 0.94; }",
      ".legend-title { font-weight: 600; font-size: 11px; fill: " + fg + "; }",
      ".legend-row-label { fill: " + fg + "; font-size: 11px; }",
    ].join(" ");
    clone.insertBefore(styleEl, clone.firstChild);

    // Render an SVG-native version of the on-screen kernel legend to the right of the chart.
    // The on-screen kernel legend is an HTML <aside>, which would be lost when serializing only the SVG.
    // Bandwidth roofline legend is already part of the SVG and is preserved by the clone.
    // This mirrors the current threshold filter, selection state, swatch colors, and per-kernel percentages.
    const SVG_NS = "http://www.w3.org/2000/svg";
    function truncateLabel(s, maxChars) {
      if (typeof s !== "string") return "";
      return s.length > maxChars ? s.slice(0, Math.max(1, maxChars - 1)) + "\u2026" : s;
    }
    function buildExportLegendPanel(panelX, chartHeight) {
      const visibleKernels = computeKernelsVisibleAtThreshold();
      const kernelRows = kLeg.filter(k => visibleKernels.has(k.name));
      const pad = 12;
      const panelW = 360;
      const headLine = 22;
      const rowH = 16;
      const swatchSize = 12;
      const labelX = pad + swatchSize + 8;
      const labelMaxChars = 36;
      const baseline = swatchSize - 1;
      const kFilterActive = kernelFilterActive();

      const panel = document.createElementNS(SVG_NS, "g");
      panel.setAttribute("transform", "translate(" + panelX + ",0)");
      const bg = document.createElementNS(SVG_NS, "rect");
      bg.setAttribute("x", "0");
      bg.setAttribute("y", "0");
      bg.setAttribute("width", String(panelW));
      bg.setAttribute("fill", plotBg);
      bg.setAttribute("stroke", gridColor);
      panel.appendChild(bg);

      let y = pad;

      const kHeadEl = document.createElementNS(SVG_NS, "text");
      kHeadEl.setAttribute("x", String(pad));
      kHeadEl.setAttribute("y", String(y + 14));
      kHeadEl.setAttribute("font-size", "13");
      kHeadEl.setAttribute("font-weight", "600");
      kHeadEl.setAttribute("fill", fg);
      let kHeadText = kernelRows.length < kLeg.length
        ? "Kernels (" + kernelRows.length + " / " + kLeg.length + ")"
        : "Kernels (" + kLeg.length + ")";
      if (kFilterActive) {
        kHeadText += " \u2014 " + selectedKernelNames.size + " selected";
      }
      kHeadEl.textContent = kHeadText;
      panel.appendChild(kHeadEl);
      y += headLine;

      kernelRows.forEach(function(k) {
        const dimmed = kFilterActive && !selectedKernelNames.has(k.name);
        const opacity = dimmed ? "0.45" : "1";
        const sw = document.createElementNS(SVG_NS, "rect");
        sw.setAttribute("x", String(pad));
        sw.setAttribute("y", String(y));
        sw.setAttribute("width", String(swatchSize));
        sw.setAttribute("height", String(swatchSize));
        sw.setAttribute("fill", k.color || "#888");
        sw.setAttribute("opacity", opacity);
        panel.appendChild(sw);

        const lab = document.createElementNS(SVG_NS, "text");
        lab.setAttribute("x", String(labelX));
        lab.setAttribute("y", String(y + baseline));
        lab.setAttribute("font-size", "11");
        lab.setAttribute("fill", fg);
        lab.setAttribute("opacity", opacity);
        lab.textContent = truncateLabel(k.name, labelMaxChars);
        panel.appendChild(lab);

        if (k.pct != null && Number.isFinite(k.pct)) {
          const pctEl = document.createElementNS(SVG_NS, "text");
          pctEl.setAttribute("x", String(panelW - pad));
          pctEl.setAttribute("y", String(y + baseline));
          pctEl.setAttribute("font-size", "11");
          pctEl.setAttribute("text-anchor", "end");
          pctEl.setAttribute("fill", fg);
          pctEl.setAttribute("opacity", opacity);
          pctEl.textContent = k.pct.toFixed(2) + "%";
          panel.appendChild(pctEl);
        }
        y += rowH;
      });

      const naturalH = y + pad;
      const finalH = Math.max(chartHeight, naturalH);
      bg.setAttribute("height", String(finalH));
      return { panel: panel, width: panelW, height: finalH };
    }

    const legendGap = 12;
    const legend = buildExportLegendPanel(widthAttr + legendGap, heightAttr);
    clone.appendChild(legend.panel);
    const exportW = widthAttr + legendGap + legend.width;
    const exportH = Math.max(heightAttr, legend.height);
    clone.setAttribute("width", String(exportW));
    clone.setAttribute("height", String(exportH));
    clone.setAttribute("viewBox", "0 0 " + exportW + " " + exportH);

    const serializer = new XMLSerializer();
    const svgString = '<?xml version="1.0" standalone="no"?>\\n' + serializer.serializeToString(clone);
    const svgBlob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
    const svgUrl = URL.createObjectURL(svgBlob);

    const scale = Math.max(1, Math.min(4, window.devicePixelRatio || 2));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(exportW * scale);
    canvas.height = Math.round(exportH * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      console.error("PNG export failed: could not acquire 2D canvas context.");
      URL.revokeObjectURL(svgUrl);
      btn.disabled = false;
      btn.textContent = prevLabel;
      return;
    }
    ctx.fillStyle = plotBg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(scale, 0, 0, scale, 0, 0);

    const img = new Image();
    function cleanup() {
      URL.revokeObjectURL(svgUrl);
      btn.disabled = false;
      btn.textContent = prevLabel;
    }
    img.onload = function() {
      try {
        ctx.drawImage(img, 0, 0, exportW, exportH);
      } catch (e) {
        console.error("PNG export draw failed:", e);
        cleanup();
        return;
      }
      const fileName = sanitizeFilename(meta && meta.title) + ".png";
      const triggerDownload = (href) => {
        const a = document.createElement("a");
        a.href = href;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      };
      const finish = () => cleanup();
      if (canvas.toBlob) {
        canvas.toBlob(function(blob) {
          if (!blob) { finish(); return; }
          const url = URL.createObjectURL(blob);
          triggerDownload(url);
          setTimeout(() => { URL.revokeObjectURL(url); finish(); }, 0);
        }, "image/png");
      } else {
        triggerDownload(canvas.toDataURL("image/png"));
        finish();
      }
    };
    img.onerror = function(err) {
      console.error("PNG export load failed:", err);
      cleanup();
    };
    img.src = svgUrl;
  }
  document.getElementById("btn-export-png").addEventListener("click", exportPng);

  function onViewControlsChange() {
    redraw();
  }
  memSel.addEventListener("change", onViewControlsChange);
  viewSel.addEventListener("change", onViewControlsChange);
  slider.addEventListener("input", () => {
    thresholdIndex = +slider.value;
    redraw();
  });
})();
</script>
</body>
</html>
"""


def extract(
    counter,
    results,
    sig_runtime,
    plot,
    dump,
    arch=None,
    base_name=None,
    output_stem=None,
):
    """output_stem: if set (e.g. --directory mode), HTML/CSV outputs use this path prefix (no extension)."""
    if isinstance(counter, str):
        last_dot = counter.rfind(".")
        file_stem = counter[:last_dot] if last_dot >= 0 else counter
        roofCountFilename = file_stem
        df_roof = pd.read_csv(counter)
    elif isinstance(counter, pd.DataFrame):
        if output_stem is not None:
            file_stem = output_stem
            roofCountFilename = Path(output_stem).name
        else:
            file_stem = base_name or "combined"
            roofCountFilename = file_stem
        df_roof = counter
    else:
        raise TypeError("counter must be a file path (str) or pandas DataFrame")

    sigRuntime = sig_runtime

    # Load roof counters into pandas df (if not already a DataFrame)


    # Check if empty
    if df_roof.empty:
        print('Input roof counters file is empty')
        quit()
    # Check for wrong file
    if "CompleteNs" in df_roof.columns:
        print('Error: "results.csv" log file submitted with "-c" flag, which is for "roof-counters.csv" log file')
        quit()

    # Build a unique-dispatch identifier. In --directory mode each subdirectory's
    # counters.csv numbers Dispatch_Id from 1, so the same Dispatch_Id value appears
    # in multiple applications; we must qualify it with Application to count
    # kernels correctly.
    dispatch_id_col = "Dispatch_Id" if "Dispatch_Id" in df_roof.columns else "Index"
    dispatch_id_cols = (
        ["Application", dispatch_id_col]
        if "Application" in df_roof.columns
        else [dispatch_id_col]
    )
    total_kernels = df_roof[dispatch_id_cols].drop_duplicates().shape[0]

    # Check for 'None' values and remove them
    bad_row_mask = df_roof.isnull().any(axis=1) | (df_roof == 'None').any(axis=1)
    total_omitted_kernels_round_1 = (
        df_roof.loc[bad_row_mask, dispatch_id_cols].drop_duplicates().shape[0]
    )

    if total_omitted_kernels_round_1 > 0:
        print(f'WARNING: {total_omitted_kernels_round_1}/{total_kernels} kernels can\'t be analyzed due to non-deterministic runs during counter collection. Attempting to continue without them.')
        # Materialize a fresh DataFrame so downstream column assignments (e.g.
        # in convert_columns_to_int) don't trigger SettingWithCopyWarning.
        df_roof = df_roof[~bad_row_mask].copy()

    if arch is None:
        raise ValueError("arch must be provided; pass --arch on the command line.")

    df_roof = convert_columns_to_int(df_roof)
    df_roof = compute_flops(df_roof, arch)  # Compute AI's for each kernel dispatch

    # Check for rocprofv3 and convert to v1 format
    if 'Agent_Id' in df_roof.columns:
        df_roof = df_roof.rename(columns={'Dispatch_Id':'Index'})
        df_roof = df_roof.rename(columns={'Kernel_Name':'KernelName'})
        # Track the renamed dispatch id column for downstream unique-kernel counts.
        dispatch_id_cols = [
            'Index' if c == 'Dispatch_Id' else c for c in dispatch_id_cols
        ]

    # Get runtime stats
    if isinstance(results, str):
        df_runtime = pd.read_csv(results)
    elif isinstance(results, pd.DataFrame):
        df_runtime = results
    else:
        raise TypeError("results must be a file path (str) or pandas DataFrame")

    # Confirm these columns match between df_roof and df_runtime
    indexes = [
        "Index",
        "Agent_Id",
        "Grid_Size",
        "KernelName",
        "Queue_Id",
        "Workgroup_Size",
    ]
    if "Application" in df_roof.columns and "Application" in df_runtime.columns:
        indexes = ["Application"] + indexes

    # Adjust column names and calculate runtime
    if 'Kernel_Name' in df_runtime.columns:
        df_runtime = df_runtime.rename(columns={'Kernel_Name':'KernelName'})
        df_runtime['Grid_Size'] = df_runtime['Grid_Size_X'] * df_runtime['Grid_Size_Y'] * df_runtime['Grid_Size_Z']
        df_runtime['Workgroup_Size'] = df_runtime['Workgroup_Size_X'] * df_runtime['Workgroup_Size_Y'] * df_runtime['Workgroup_Size_Z']
        df_runtime['DurationNs'] = df_runtime['End_Timestamp'] - df_runtime['Start_Timestamp']
        df_runtime = df_runtime.rename(columns={'Dispatch_Id':'Index'})

    negative_duration = (df_runtime['DurationNs'] < 0).sum()
    if negative_duration > 0:
        df_runtime = df_runtime[df_runtime['DurationNs'] >= 0]
        print(f"WARNING: {negative_duration}/{total_kernels} kernel-trace rows removed for having an end timestamp before the start timestamp. Attempting to continue without them.")

    pre_merge_unique_kernels = df_roof[dispatch_id_cols].drop_duplicates().shape[0]

    # Merge runtimes into df_roof to get per-dispatch weight (Percentage of total runtime)
    df_roof = df_roof.merge(df_runtime[indexes + ['DurationNs']], on=indexes)

    post_merge_unique_kernels = df_roof[dispatch_id_cols].drop_duplicates().shape[0]
    total_omitted_kernels_round_2 = pre_merge_unique_kernels - post_merge_unique_kernels

    if total_omitted_kernels_round_2 > 0:
        print(f"WARNING: {total_omitted_kernels_round_2}/{total_kernels} kernels can't be analyzed due to non-deterministic runs between counter collection and timing. Attempting to continue without them.")

    df_roof['Percentage'] = df_roof['DurationNs'] / df_roof['DurationNs'].sum() * 100

    # Aggregate kernels with weighted average by percentage of total runtime
    exclude = {'Percentage', 'DurationNs', 'TOTAL_OPS', 'BW_HBM', 'BW_L2', 'BW_vL1d', 'BW_LDS'}
    num_cols = [c for c in df_roof.select_dtypes(include=[np.number]).columns if c not in exclude]
    weight_sum = df_roof.groupby('KernelName', sort=False)['Percentage'].sum()
    df = (df_roof[num_cols].mul(df_roof['Percentage'], axis=0)
          .groupby(df_roof['KernelName'], sort=False).sum()
          .div(weight_sum, axis=0)).reset_index()

    # Calculate total, average, and percentage runtimes for aggregated kernels
    totalRuntimes = df_roof.groupby('KernelName')['DurationNs'].sum()
    averageRuntimes = df_roof.groupby('KernelName')['DurationNs'].mean()
    percentRuntimes = df_roof.groupby('KernelName').sum()['DurationNs']/df_roof['DurationNs'].sum()*100

    # Merge runtimes into df
    df = pd.merge(df, totalRuntimes, on='KernelName').rename(columns={"DurationNs":"RuntimeNs"})
    df = pd.merge(df, averageRuntimes, on='KernelName').rename(columns={"DurationNs":"AverageNs"})
    df = pd.merge(df, percentRuntimes, on='KernelName').rename(columns={"DurationNs":"Percentage"})

    # Add column for number of dispatches
    df = df.merge(df_roof.groupby('KernelName', sort=False).size().reset_index(name='Count'), on='KernelName')
    # Add total operations for aggregated kernels
    totalOps = df_roof.groupby('KernelName')['TOTAL_OPS'].mean()
    totalBwHbm = df_roof.groupby('KernelName')['BW_HBM'].mean()
    totalBwL2 = df_roof.groupby('KernelName')['BW_L2'].mean()
    totalBwVl1d = df_roof.groupby('KernelName')['BW_vL1d'].mean()
    totalBwLds = df_roof.groupby('KernelName')['BW_LDS'].mean()
    df = pd.merge(df, totalOps, on='KernelName')
    df = pd.merge(df, totalBwHbm, on='KernelName')
    df = pd.merge(df, totalBwL2, on='KernelName')
    df = pd.merge(df, totalBwVl1d, on='KernelName')
    df = pd.merge(df, totalBwLds, on='KernelName')
    # Recalculate arithmetic intensity for aggregated kernels
    df['AI_HBM_TOT'] = df['TOTAL_OPS'] / df['BW_HBM']
    df['AI_L2_TOT'] = df['TOTAL_OPS'] / df['BW_L2']
    df['AI_vL1d_TOT'] = df['TOTAL_OPS'] / df['BW_vL1d']
    df['AI_LDS_TOT'] = df['TOTAL_OPS'] / df['BW_LDS']
    # Recalculate peaks for aggregated kernels
    df['HBM_BW_PEAK'] = df['AI_HBM_TOT'] * caches[arch]['HBM']
    df['HBM_BW_PEAK_LINEAR'] = df['HBM_BW_PEAK']
    df['L2_BW_PEAK'] = df['AI_L2_TOT'] * caches[arch]['L2']
    df['vL1d_BW_PEAK'] = df['AI_vL1d_TOT'] * caches[arch]['vL1d']
    df['LDS_BW_PEAK'] = df['AI_LDS_TOT'] * caches[arch]['LDS']
    df['LDS_BW_PEAK_LINEAR'] = df['LDS_BW_PEAK']

    _bw_agg_path = _config_dir / "benchWarmer.csv"
    _df_bw_agg = pd.read_csv(_bw_agg_path)
    _peaks_agg = _load_peaks(_df_bw_agg, arch, df)
    if _use_hbm_alpha_model(arch):
        _alpha_agg = _compute_weighted_hbm_alpha(df, _peaks_agg, arch)
        df["HBM_BW_PEAK"] = _exclude_bw_roof_when_no_traffic(
            _hbm_roof_throughput(
                df["AI_HBM_TOT"],
                caches[arch]["HBM"],
                df["KERNEL_COMPUTE_PEAK"],
                _alpha_agg,
            ),
            df["BW_HBM"],
        )
        df["HBM_ALPHA"] = _alpha_agg

    if _use_lds_alpha_model(arch):
        _alpha_lds_agg = _compute_weighted_lds_alpha(df, _peaks_agg, arch)
        df["LDS_BW_PEAK"] = _exclude_bw_roof_when_no_traffic(
            _hbm_roof_throughput(
                df["AI_LDS_TOT"],
                caches[arch]["LDS"],
                df["KERNEL_COMPUTE_PEAK"],
                _alpha_lds_agg,
            ),
            df["BW_LDS"],
        )
        df["LDS_ALPHA"] = _alpha_lds_agg

    insigKernels = len(df[df['Percentage'] < sigRuntime])  # save this value for guided analysis

    # Drop temporary weight columns so full merge with df_runtime does not create _x/_y duplicates
    df_roof = df_roof.drop(columns=['DurationNs', 'Percentage'])
    df_roof = df_roof.merge(df_runtime, on=indexes)

    df.set_index('KernelName', inplace=True)

    # Prepare for analysis
    df = df.sort_values(by='Percentage', ascending=False)
    df_peaks = df[['HBM_BW_PEAK','L2_BW_PEAK','vL1d_BW_PEAK','LDS_BW_PEAK','KERNEL_COMPUTE_PEAK']]
    df['PEAK'] = df_peaks.where(df_peaks > 0).min(axis=1)

    df['LIMITER'] = _safe_limiters(df_peaks.where(df_peaks > 0), 5, " (" + arch + ")")
    df_peaks_linear = df[
        ['HBM_BW_PEAK_LINEAR', 'L2_BW_PEAK', 'vL1d_BW_PEAK', 'LDS_BW_PEAK_LINEAR', 'KERNEL_COMPUTE_PEAK']
    ]
    df['PEAK_LINEAR'] = df_peaks_linear.where(df_peaks_linear > 0).min(axis=1)
    df['LIMITER_LINEAR'] = _safe_limiters(df_peaks_linear.where(df_peaks_linear > 0), 5, " (" + arch + ")")
    df["LIMITER"] = _align_curved_limiter_with_linear_compute(
        df["LIMITER"], df["LIMITER_LINEAR"]
    )
    df['Throughput'] = df['TOTAL_OPS'] / df['AverageNs']
    df['PercentAchieved'] = _percent_roof_achieved(df['Throughput'], df['PEAK'])
    df['PercentAchieved_LINEAR'] = _percent_roof_achieved(df['Throughput'], df['PEAK_LINEAR'])

    # When neither the HBM nor LDS α-blended (curved) model is available for
    # this arch, PEAK / LIMITER / PercentAchieved all collapse to the linear
    # piecewise-linear values. Track this so the CLI output avoids labeling
    # those numbers as "curved".
    curved_avail = _use_hbm_alpha_model(arch) or _use_lds_alpha_model(arch)

    # Guided analysis
    print(f"Total unique kernels: {len(df)}")
    print(f"Total kernel dispatches: {len(df_roof)}")
    print()

    for index, row in df.iterrows():  # loop through each kernel
        if row['Percentage'] < sigRuntime:
            continue
        print(f"\033[1m{index}\033[0m")
        print(f"  Total contribution to GPU time: {round(row['Percentage'], 3)} %")
        print(f"  Total runtime (all dispatches):", "{:.3e}".format(row['RuntimeNs']), "ns ({:.3f} s)".format(row['RuntimeNs'] / 1e9))
        print("  Average runtime per dispatch:".ljust(33), "{:.3e}".format(row['AverageNs']), "ns ({:.3f} ms)".format(row['AverageNs'] / 1e6))
        print("  Total dispatches:".ljust(33), int(row['Count']))
        print()

        print("  Total operations per dispatch:".ljust(33), "{:.3e}".format(row['TOTAL_OPS']), "FLOPs")
        print()

        print("  Total bytes moved (HBM):".ljust(33), _format_byte_size(row['BW_HBM']))
        print("  Total bytes moved (L2):".ljust(33), _format_byte_size(row['BW_L2']))
        print("  Total bytes moved (L1):".ljust(33), _format_byte_size(row['BW_vL1d']))
        print("  Total bytes moved (LDS):".ljust(33), _format_byte_size(row['BW_LDS']))
        print()

        print("  Arithmetic intensity (HBM):".ljust(33), round(row['AI_HBM_TOT'], 4), " FLOPs/B")
        print("  Arithmetic intensity (L2):".ljust(33), round(row['AI_L2_TOT'], 4), " FLOPs/B")
        print("  Arithmetic intensity (L1):".ljust(33), round(row['AI_vL1d_TOT'], 4), " FLOPs/B")
        print("  Arithmetic intensity (LDS):".ljust(33), round(row['AI_LDS_TOT'], 4), " FLOPs/B")
        print()

        if 'KERNEL_COMPUTE' in row['LIMITER']:
            inst_mix = [
                ("FP16 Add",    64 * row['SQ_INSTS_VALU_ADD_F16'],     _peaks_agg.get('fp16_add')),
                ("FP16 Mul",    64 * row['SQ_INSTS_VALU_MUL_F16'],     _peaks_agg.get('fp16_mul')),
                ("FP16 MulAdd", 64 * 2 * row['SQ_INSTS_VALU_FMA_F16'], _peaks_agg.get('fp16_muladd')),
                ("FP16 Trans",  64 * row['SQ_INSTS_VALU_TRANS_F16'],   _peaks_agg.get('fp16_trans')),
                ("FP32 Add",    64 * row['SQ_INSTS_VALU_ADD_F32'],     _peaks_agg.get('fp32_add')),
                ("FP32 Mul",    64 * row['SQ_INSTS_VALU_MUL_F32'],     _peaks_agg.get('fp32_mul')),
                ("FP32 MulAdd", 64 * 2 * row['SQ_INSTS_VALU_FMA_F32'], _peaks_agg.get('fp32_muladd')),
                ("FP32 Trans",  64 * row['SQ_INSTS_VALU_TRANS_F32'],   _peaks_agg.get('fp32_trans')),
                ("FP64 Add",    64 * row['SQ_INSTS_VALU_ADD_F64'],     _peaks_agg.get('fp64_add')),
                ("FP64 Mul",    64 * row['SQ_INSTS_VALU_MUL_F64'],     _peaks_agg.get('fp64_mul')),
                ("FP64 MulAdd", 64 * 2 * row['SQ_INSTS_VALU_FMA_F64'], _peaks_agg.get('fp64_muladd')),
                ("FP64 Trans",  64 * row['SQ_INSTS_VALU_TRANS_F64'],   _peaks_agg.get('fp64_trans')),
                ("INT32",       row['TOTAL_VALU_I32'],                 _peaks_agg.get('int32')),
                ("INT64",       row['TOTAL_VALU_I64'],                 _peaks_agg.get('int64')),
                ("FP16 MFMA",   row['TOTAL_MOPS_F16'],                 _peaks_agg.get('fp16_mfma')),
                ("BF16 MFMA",   row['TOTAL_MOPS_BF16'],                _peaks_agg.get('bf16_mfma')),
                ("FP32 MFMA",   row['TOTAL_MOPS_F32'],                 _peaks_agg.get('fp32_mfma')),
                ("FP64 MFMA",   row['TOTAL_MOPS_F64'],                 _peaks_agg.get('fp64_mfma')),
                ("SALU",        row['TOTAL_SALU'],                     None),
            ]
            if 'TOTAL_MOPS_F8' in row.index:
                inst_mix.append(("FP8 MFMA", row['TOTAL_MOPS_F8'], _peaks_agg.get('fp8_mfma')))
            if 'TOTAL_MOPS_I8' in row.index:
                inst_mix.append(("I8 MFMA", row['TOTAL_MOPS_I8'], _peaks_agg.get('i8_mfma')))
            if 'TOTAL_MOPS_F6F4' in row.index:
                inst_mix.append(("F6F4 MFMA", row['TOTAL_MOPS_F6F4'], _peaks_agg.get('f6f4_mfma')))
            if 'TOTAL_VALU_OTHER' in row.index:
                inst_mix.append(("Other VALU", row['TOTAL_VALU_OTHER'], _peaks_agg.get('other')))
            total_ops = sum(ops for _, ops, _ in inst_mix)
            if total_ops > 0:
                inst_mix = [
                    (label, ops / total_ops * 100, peak)
                    for label, ops, peak in inst_mix
                    if ops > 0
                ]
                inst_mix.sort(key=lambda x: x[1], reverse=True)
                print("  Instruction mix:".ljust(33), "(roofline peak)")
                other_valu_printed = False
                for label, pct, peak in inst_mix:
                    if peak is None or not np.isfinite(peak) or peak <= 0:
                        peak_str = "N/A"
                    else:
                        peak_str = f"{peak / 1000:.1f} TFLOPs/s"
                    display_label = label
                    if label == "Other VALU":
                        display_label = label + "*"
                        other_valu_printed = True
                    print(f"    {display_label + ':':20s} {pct:5.1f}%   ({peak_str})")
                if other_valu_printed:
                    print("    * \"Other VALU\" refers to instructions not covered by the other categories.")
                    print("      We use the peak throughput value of v_lshlrev_b32_e32 here.")
                print()

        # Calculate achieved
        achieved = row['Throughput']
        print("  Achieved throughput:".ljust(40), _format_throughput(achieved))
        _mem_levels = []
        for _lim in (row['LIMITER'], row.get('LIMITER_LINEAR')):
            _lvl = _limiter_memory_level(_lim)
            if _lvl is not None and _lvl not in _mem_levels:
                _mem_levels.append(_lvl)
        for _lvl in _mem_levels:
            _achieved_bw = row[f"BW_{_lvl}"] / row["AverageNs"]
            print(
                f"  Achieved {_lvl} bandwidth:".ljust(40),
                _format_bandwidth(_achieved_bw),
            )
        print()

        if curved_avail:
            print(
                "  Linear roofline performance limiter:".ljust(40),
                row["LIMITER_LINEAR"],
            )
            print(
                "  Linear roofline peak throughput:".ljust(40),
                _format_throughput(row["PEAK_LINEAR"]),
            )
            _lin_mem_level = _limiter_memory_level(row["LIMITER_LINEAR"])
            if _lin_mem_level is not None:
                print(
                    f"  Linear roofline peak {_lin_mem_level} bandwidth:".ljust(40),
                    _format_bandwidth(caches[arch][_lin_mem_level]),
                )
            print(
                "  Percent of linear roofline achieved:".ljust(40),
                round(row["PercentAchieved_LINEAR"], 4),
                "%",
            )
            print()

        _roof_label = "Curved" if curved_avail else "Linear"
        print(f"  {_roof_label} performance limiter:".ljust(40), row['LIMITER'])
        print(f"  {_roof_label} roofline peak throughput:".ljust(40), _format_throughput(row['PEAK']))
        _mem_level = _limiter_memory_level(row['LIMITER'])
        if _mem_level is not None:
            _ai = float(pd.to_numeric(row.get(f"AI_{_mem_level}_TOT"), errors="coerce"))
            _peak_thr = float(pd.to_numeric(row.get(f"{_mem_level}_BW_PEAK"), errors="coerce"))
            if np.isfinite(_ai) and _ai > 0 and np.isfinite(_peak_thr):
                _peak_bw = _peak_thr / _ai
            else:
                _peak_bw = float("nan")
            print(
                f"  {_roof_label} roofline peak {_mem_level} bandwidth:".ljust(40),
                _format_bandwidth(_peak_bw),
            )
        print(f"  Percent of {_roof_label.lower()} roofline achieved:".ljust(40), round(row['PercentAchieved'], 4), "%")
        print()

    print(f"\033[1m{insigKernels} kernels omitted from CLI output for having less than {sigRuntime} percent GPU time (use --sig-runtime to change threshold)\033[0m")
    print()

    _gpu_time_label = f"Total application GPU time on {arch}:"
    _wall_active_label = "Total application wall-clock kernel-active time:"
    _total_app_ns = df_runtime['DurationNs'].sum()
    overlap_pct, wall_active_ns = _compute_kernel_overlap_pct(df_runtime)
    _show_wall_active = (
        np.isfinite(overlap_pct) and overlap_pct > 1 and np.isfinite(wall_active_ns)
    )
    if _show_wall_active:
        _time_label_width = max(len(_gpu_time_label), len(_wall_active_label))
        _gpu_time_label_out = _gpu_time_label.ljust(_time_label_width)
        _wall_active_label_out = _wall_active_label.ljust(_time_label_width)
    else:
        _gpu_time_label_out = _gpu_time_label
        _wall_active_label_out = _wall_active_label
    print(
        f"{_gpu_time_label_out} "
        f"{_total_app_ns:.2e} ns ({_total_app_ns/1e9:.6f} s)"
    )
    if np.isfinite(overlap_pct) and overlap_pct > 1:
        if _show_wall_active:
            print(
                f"{_wall_active_label_out} "
                f"{wall_active_ns:.2e} ns ({wall_active_ns/1e9:.6f} s)"
            )
        print(
            f"Kernel runtime overlap: {overlap_pct:.3f} % of total kernel runtime consists of overlapping, concurrent kernels"
        )
    print()

    df_roof['Percentage'] = df_roof['DurationNs']/sum(totalRuntimes) * 100
    df_roof['Throughput'] = df_roof['TOTAL_OPS'] / df_roof['DurationNs']
    df_roof['PercentAchieved'] = _percent_roof_achieved(df_roof['Throughput'], df_roof['PEAK'])
    df_roof['PercentAchieved_LINEAR'] = _percent_roof_achieved(df_roof['Throughput'], df_roof['PEAK_LINEAR'])
    df_roof = df_roof.rename(columns={'KernelName_x':'KernelName'}).merge(df['Percentage'],on='KernelName')
    df_roof = df_roof.rename(columns={'Percentage_x':'Percentage'})
    df_roof = df_roof.rename(columns={'Percentage_y':'PercentageAggregate'})
    _pct = df_roof['Percentage'].astype(float)
    _pa = df_roof['PercentAchieved']
    _pa_linear = df_roof['PercentAchieved_LINEAR']
    # Drop rows whose PEAK collapsed to the eps floor in _hbm_roof_throughput / clip(eps).
    # Those rows contribute astronomically large achieved-% values (throughput / 1e-30)
    # that swamp the weighted average. The floor only triggers when input counters are
    # missing or inconsistent for that dispatch, so excluding them gives a representative
    # number rather than NaN/inf. Use a generous floor (1e-12) so we drop exactly the
    # sentinel rows without affecting legitimately-tiny peaks.
    _peak_floor = 1e-12
    _peak = df_roof['PEAK'].astype(float)
    _peak_linear = df_roof['PEAK_LINEAR'].astype(float)
    _valid_peak = _peak.notna() & (_peak > _peak_floor)
    _valid_peak_linear = _peak_linear.notna() & (_peak_linear > _peak_floor)
    _w = _pct.notna() & _pa.notna() & (_pct > 0) & _valid_peak
    _w_linear = _pct.notna() & _pa_linear.notna() & (_pct > 0) & _valid_peak_linear
    _dropped = int((_pct.notna() & _pa.notna() & (_pct > 0) & ~_valid_peak).sum())
    _dropped_linear = int((_pct.notna() & _pa_linear.notna() & (_pct > 0) & ~_valid_peak_linear).sum())
    if _w.any():
        roofline_percentage = (_pct[_w] * _pa[_w]).sum() / _pct[_w].sum()
    else:
        roofline_percentage = float("nan")
    if _w_linear.any():
        roofline_percentage_linear = (_pct[_w_linear] * _pa_linear[_w_linear]).sum() / _pct[_w_linear].sum()
    else:
        roofline_percentage_linear = float("nan")
    if _dropped or _dropped_linear:
        print(
            "Note: excluded dispatch(es) with degenerate roofline peaks "
            f"(missing/inconsistent counters) from the average percent calculations: "
            f"curved={_dropped}, linear={_dropped_linear}."
        )
    if curved_avail:
        print(f"Average percent of linear roofline achieved: {roofline_percentage_linear:.3f} %")
        print(f"Average percent of curved roofline achieved: {roofline_percentage:.3f} %")
    else:
        # PEAK == PEAK_LINEAR in this case, so there's only one meaningful number to report.
        print(f"Average percent of linear roofline achieved: {roofline_percentage_linear:.3f} %")
    print()

    # Interactive roofline plot (D3)
    if plot:
        df_plot = df_roof.copy()
        total_dispatches_orig = len(df_plot)
        df_plot["TotalKernels"] = total_dispatches_orig

        n_samples_cap = 50000
        if len(df_plot) > n_samples_cap:
            df_plot = df_plot.sort_values("Index")
            df_plot = df_plot.iloc[:: len(df_plot) // n_samples_cap]

        df_plot = df_plot.sort_values(by="PercentageAggregate", ascending=False)

        name_to_color = {name: _QUAL_COLORS[i % len(_QUAL_COLORS)] for i, name in enumerate(df.index)}
        df_plot["color"] = [name_to_color[name] for name in df_plot["KernelName"]]
        df = df.copy()
        df["color"] = [name_to_color[name] for name in df.index]

        percentage_desc = df["Percentage"].sort_values(ascending=False)
        cumulative_percentage_by_value = (
            percentage_desc.groupby(percentage_desc, sort=False).sum().cumsum()
        )
        df["CumulativePercentageAbove"] = df["Percentage"].map(cumulative_percentage_by_value)
        df_plot = df_plot.merge(df["CumulativePercentageAbove"], on="KernelName")

        x_min = 0.001
        x_max = 100000
        y_min = 0.5
        if df_plot['PEAK'].max() > 500000:
            y_max = 2000000
        else:
            y_max = 500000

        x_vals = np.logspace(np.log10(x_min), np.log10(x_max), 200)
        compute_peak = float(df["COMPUTE_PEAK"].iloc[0])
        x_vals = np.sort(
            np.append(x_vals, [compute_peak / caches[arch][cache] for cache in caches[arch]])
        )

        cache_key_list = list(caches[arch].keys())
        rooflines_payload = []
        for key in cache_key_list:
            y_line = np.minimum(caches[arch][key] * x_vals, compute_peak)
            rooflines_payload.append(
                {"key": key, "x": x_vals.tolist(), "y": y_line.tolist()}
            )

        dispatch = []
        for _, row in df_plot.iterrows():
            kid = row["KernelName"]
            idx = row["Index"]
            dispatch.append(
                {
                    "id": f"{kid}\x00{idx}",
                    "kernelName": str(kid),
                    "nameDisplay": _wrap_kernel_name_tooltip(str(kid)),
                    "ai": {
                        "HBM": _json_safe_float(row["AI_HBM_TOT"]),
                        "L2": _json_safe_float(row["AI_L2_TOT"]),
                        "vL1d": _json_safe_float(row["AI_vL1d_TOT"]),
                        "LDS": _json_safe_float(row["AI_LDS_TOT"]),
                    },
                    "throughput": _json_safe_float(row["Throughput"]),
                    "percentAchieved": _json_safe_float(row["PercentAchieved"]),
                    "percentage": _json_safe_float(row["Percentage"]),
                    "percentageAggregate": _json_safe_float(row["PercentageAggregate"]),
                    "cumulativePct": _json_safe_float(row["CumulativePercentageAbove"]),
                    "index": int(row["Index"]) if pd.notna(row["Index"]) else 0,
                    "totalKernels": int(row["TotalKernels"]),
                    "peak": _json_safe_float(row["PEAK"]),
                    "peakLinear": _json_safe_float(row.get("PEAK_LINEAR")),
                    "kernelComputePeak": _json_safe_float(row["KERNEL_COMPUTE_PEAK"]),
                    "limiter": str(row["LIMITER"]),
                    "limiterLinear": str(row["LIMITER_LINEAR"]) if "LIMITER_LINEAR" in row.index else None,
                    "color": str(row["color"]),
                }
            )

        aggregate = []
        for index, kernel in df.iterrows():
            aggregate.append(
                {
                    "id": str(index),
                    "kernelName": str(index),
                    "nameDisplay": _wrap_kernel_name_tooltip(str(index)),
                    "ai": {
                        "HBM": _json_safe_float(kernel["AI_HBM_TOT"]),
                        "L2": _json_safe_float(kernel["AI_L2_TOT"]),
                        "vL1d": _json_safe_float(kernel["AI_vL1d_TOT"]),
                        "LDS": _json_safe_float(kernel["AI_LDS_TOT"]),
                    },
                    "throughput": _json_safe_float(kernel["Throughput"]),
                    "percentAchieved": _json_safe_float(kernel["PercentAchieved"]),
                    "percentage": _json_safe_float(kernel["Percentage"]),
                    "cumulativePct": _json_safe_float(kernel["CumulativePercentageAbove"]),
                    "peak": _json_safe_float(kernel["PEAK"]),
                    "peakLinear": _json_safe_float(kernel.get("PEAK_LINEAR")),
                    "limiter": str(kernel["LIMITER"]),
                    "limiterLinear": str(kernel["LIMITER_LINEAR"]) if "LIMITER_LINEAR" in kernel.index else "",
                    "count": int(kernel["Count"]),
                    "color": str(kernel["color"]),
                }
            )

        thresholds = df["CumulativePercentageAbove"].tolist()
        thresholds.sort(reverse=True)
        thresholds = [100] + thresholds[-40:]

        disclaimer_text = ""
        if total_dispatches_orig > len(df_plot):
            disclaimer_text = (
                f"Depicting {len(df_plot)} kernel dispatches out of {total_dispatches_orig}"
            )

        df_kleg = df.sort_values("Percentage", ascending=False)
        kernel_legend = [
            {
                "name": str(kn),
                "color": str(row["color"]),
                "pct": _json_safe_float(row["Percentage"]),
            }
            for kn, row in df_kleg.iterrows()
        ]

        kernel_compute_peak_by_name = {
            str(kn): _json_safe_float(val)
            for kn, val in df["KERNEL_COMPUTE_PEAK"].items()
            if pd.notna(val) and np.isfinite(val) and float(val) > 0
        }

        use_hbm_alpha = _use_hbm_alpha_model(arch)
        kernel_hbm_alpha_by_name = {}
        if use_hbm_alpha and "HBM_ALPHA" in df.columns:
            for kn, r in df.iterrows():
                a = r.get("HBM_ALPHA")
                if pd.notna(a) and np.isfinite(a):
                    kernel_hbm_alpha_by_name[str(kn)] = {"alpha": float(a)}

        _def_alpha = _default_hbm_alpha_fp64_mfma(arch) if use_hbm_alpha else None

        use_lds_alpha = _use_lds_alpha_model(arch)
        kernel_lds_alpha_by_name = {}
        if use_lds_alpha and "LDS_ALPHA" in df.columns:
            for kn, r in df.iterrows():
                a = r.get("LDS_ALPHA")
                if pd.notna(a) and np.isfinite(a):
                    kernel_lds_alpha_by_name[str(kn)] = {"alpha": float(a)}

        _def_lds_alpha = _default_lds_alpha_fp64_mfma(arch) if use_lds_alpha else None

        payload = {
            "meta": {
                "title": f"Roofline Plot ({arch}) for kernels in {roofCountFilename}",
                "xMin": x_min,
                "xMax": x_max,
                "yMin": y_min,
                "yMax": y_max,
                "xAxisTitle": "Arithmetic Intensity (Flops per Byte Accessed)",
                "yAxisTitle": "Throughput (GFLOPs/s)",
                "disclaimer": disclaimer_text,
                "computePeak": _json_safe_float(compute_peak),
                "useHbmAlphaModel": use_hbm_alpha,
                "hbmAlphaArch": arch if use_hbm_alpha else None,
                "defaultHbmAlphaFp64Mfma": _json_safe_float(_def_alpha),
                "useLdsAlphaModel": use_lds_alpha,
                "ldsAlphaArch": arch if use_lds_alpha else None,
                "defaultLdsAlphaFp64Mfma": _json_safe_float(_def_lds_alpha),
            },
            "kernelComputePeakByName": kernel_compute_peak_by_name,
            "kernelHbmAlphaByName": kernel_hbm_alpha_by_name,
            "kernelLdsAlphaByName": kernel_lds_alpha_by_name,
            "cacheKeys": cache_key_list,
            "bandwidths": {
                k: _json_safe_float(caches[arch][k]) for k in cache_key_list
            },
            "rooflines": rooflines_payload,
            "thresholds": [float(t) for t in thresholds],
            "dispatch": dispatch,
            "aggregate": aggregate,
            "kernelLegend": kernel_legend,
        }

        html_out = _build_roofline_d3_html(payload)
        with open(f"{file_stem}.html", "w", encoding="utf-8") as _hf:
            _hf.write(html_out)
        print(f"Roofline plot saved to                  {file_stem}.html")

    # Output Results to CSV
    if dump:
        df.to_csv(f'{file_stem}_EXTRACTED_AGG.csv')
        df_roof.to_csv(f'{file_stem}_EXTRACTED.csv')
        print(f"Full dataframe dumped to                {file_stem}_EXTRACTED.csv")
        print(f"Aggregate kernels dataframe dumped to   {file_stem}_EXTRACTED_AGG.csv")

def main():

    # Get filenames from input args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--counter', required=False, help='Provide roof_counters.csv filename from rocprof -i (required unless --directory)')
    parser.add_argument('-r', '--results', required=False, help='Provide results.csv / trace filename from rocprof --stats (required unless --directory)')
    parser.add_argument('-D', '--directory', required=False, metavar='DIR', help='Directory of subdirs, each with counters.csv and trace_kernel_trace.csv; plot/CSV outputs are written here')
    parser.add_argument('--sig-runtime', required=False, default=10, type=float, help='Provide the percentage of runtime considered "significant" for analysis (kernels under that percentage will be omitted)')
    parser.add_argument('-p', '--plot', action='store_true', required=False, help='Generate plots')
    parser.add_argument('-d', '--dump', action='store_true', required=False, help='Dump DataFrame to csv')
    parser.add_argument('--arch', required=False, help='Supply architecture (to aid in guided analysis). Options: MI250, MI250X, MI300A, MI300X, MI325X, MI350X, MI355X')

    args = parser.parse_args()
    if args.directory:
        if args.counter is not None or args.results is not None:
            print("Note: --directory is set; ignoring -c/--counter and -r/--results.")
    elif not args.counter or not args.results:
        parser.error("Provide --directory/-D, or both -c/--counter and -r/--results.")

    supported_arch = set(caches.keys())
    if args.arch is None:
        parser.error(
            f"--arch is required. Supported: {', '.join(sorted(supported_arch))}"
        )
    args.arch = args.arch.replace('_', ' ').upper()

    if args.arch not in supported_arch:
        parser.error(
            f"Unsupported architecture '{args.arch}'. Supported: {', '.join(sorted(supported_arch))}"
        )

    if args.directory:
        df_roof, df_runtime, base = load_from_directory(args.directory)
        out_stem = str(Path(args.directory).resolve() / base)
        extract(
            df_roof,
            df_runtime,
            args.sig_runtime,
            args.plot,
            args.dump,
            args.arch,
            base_name=base,
            output_stem=out_stem,
        )
    else:
        extract(
            args.counter,
            args.results,
            args.sig_runtime,
            args.plot,
            args.dump,
            args.arch,
            base_name=None,
            output_stem=None,
        )

if __name__ == "__main__":
    main()

