# Roofline Extractor Metrics Documentation

This document provides equations, descriptions, and hardware counter information for each metric displayed in the roofline extractor text output.

> **Looking for a quick reference?** See [METRICS_SUMMARY.md](METRICS_SUMMARY.md) for high-level descriptions and interpretation guidelines.

## Table of Contents
- [Output Metrics](#output-metrics)
  - [Total Operations Per Dispatch](#total-operations-per-dispatch)
  - [Average GPU Time Per Dispatch](#average-runtime-per-dispatch)
  - [Total GPU Time](#total-runtime)
  - [Total Contribution to GPU Time](#total-contribution-to-runtime)
  - [Total Dispatches](#total-dispatches)
  - [Arithmetic Intensity](#arithmetic-intensity)
  - [Performance Limiter](#performance-limiter)
  - [Peak Achievable Throughput](#peak-achievable-throughput)
  - [Distance from Roofline](#distance-from-roofline)
  - [Achieved Throughput](#achieved-throughput)
- [Operation Categories](#operation-categories)
- [Bandwidth Calculations](#bandwidth-calculations)
- [Hardware Counters Reference](#hardware-counters-reference)

---

## Output Metrics

### Total Operations Per Dispatch

**Description**: The total number of operations (FLOPs and integer ops) executed per kernel dispatch.

**Equation**:
```
TOTAL_OPS = TOTAL_SALU + TOTAL_VALU_F16 + TOTAL_VALU_F32 + TOTAL_VALU_F64 + 
            TOTAL_VALU_I32 + TOTAL_VALU_I64 + TOTAL_MOPS_F16 + TOTAL_MOPS_BF16 + 
            TOTAL_MOPS_F32 + TOTAL_MOPS_F64 [+ TOTAL_MOPS_F8] [+ TOTAL_MOPS_I8] [+ TOTAL_VALU_OTHER]
```

Where:
- `TOTAL_SALU = SQ_INSTS_SALU`
- `TOTAL_VALU_F16 = 64 × (SQ_INSTS_VALU_ADD_F16 + SQ_INSTS_VALU_MUL_F16 + SQ_INSTS_VALU_TRANS_F16 + 2 × SQ_INSTS_VALU_FMA_F16)`
- `TOTAL_VALU_F32 = 64 × (SQ_INSTS_VALU_ADD_F32 + SQ_INSTS_VALU_MUL_F32 + SQ_INSTS_VALU_TRANS_F32 + 2 × SQ_INSTS_VALU_FMA_F32)`
- `TOTAL_VALU_F64 = 64 × (SQ_INSTS_VALU_ADD_F64 + SQ_INSTS_VALU_MUL_F64 + SQ_INSTS_VALU_TRANS_F64 + 2 × SQ_INSTS_VALU_FMA_F64)`
- `TOTAL_VALU_I32 = 64 × SQ_INSTS_VALU_INT32`
- `TOTAL_VALU_I64 = 64 × SQ_INSTS_VALU_INT64`
- `TOTAL_MOPS_F16 = 512 × SQ_INSTS_VALU_MFMA_MOPS_F16`
- `TOTAL_MOPS_BF16 = 512 × SQ_INSTS_VALU_MFMA_MOPS_BF16`
- `TOTAL_MOPS_F32 = 512 × SQ_INSTS_VALU_MFMA_MOPS_F32`
- `TOTAL_MOPS_F64 = 512 × SQ_INSTS_VALU_MFMA_MOPS_F64`
- `TOTAL_MOPS_F8 = 512 × SQ_INSTS_VALU_MFMA_MOPS_F8` (if available)
- `TOTAL_MOPS_I8 = 512 × SQ_INSTS_VALU_MFMA_MOPS_I8` (if available)

**Notes**: 
- The factor of 64 for VALU operations accounts for the wavefront size (64 threads per wavefront).
- The factor of 2 for FMA operations accounts for both the multiply and add operations.
- The factor of 512 for MFMA (Matrix FMA) operations accounts for the matrix dimensions.

### Average GPU Time Per Dispatch

**Description**: The average time (in nanoseconds) a single kernel dispatch takes to execute.

**Equation**:
```
AverageNs = RuntimeNs / Count
```

Where:
- `RuntimeNs` = Total cumulative runtime across all dispatches of this kernel (from kernel trace)
- `Count` = Number of times this kernel was dispatched

**Hardware Counters**: Derived from runtime statistics collected via `rocprofv3 --kernel-trace`

### Total GPU time

**Description**: The total cumulative runtime (in nanoseconds) for all dispatches of a kernel.

**Equation**:
```
RuntimeNs = Sum of DurationNs for all dispatches of this kernel
```

Where:
- `DurationNs` = Runtime of individual kernel dispatch (from kernel trace)

**Hardware Counters**: Derived from runtime statistics collected via `rocprofv3 --kernel-trace`

### Total Contribution to GPU time

**Description**: The percentage of total, application-wide GPU time consumed by this kernel.

**Equation**:
```
Percentage = (RuntimeNs / TotalApplicationRuntime) × 100
```

Where:
- `RuntimeNs` = Total runtime for this kernel
- `TotalApplicationRuntime` = Sum of RuntimeNs across all kernels

### Total Dispatches

**Description**: The number of times this kernel was dispatched (launched) during application execution.

**Hardware Counters**: Count of kernel invocations from kernel trace data

### Arithmetic Intensity

**Description**: The ratio of operations performed to bytes transferred for each memory hierarchy level. Higher values indicate more compute-intensive workloads.

**Equations**:

#### Arithmetic Intensity (HBM - High Bandwidth Memory)
```
AI_HBM_TOT = TOTAL_OPS / BW_HBM
```

Where:
```
BW_HBM (for gfx950) = 32 × TCC_EA0_WRREQ_WRITE_DRAM_32B_sum +
                      32 × TCC_EA0_RDREQ_DRAM_32B_sum +
                      32 × TCC_EA0_WRREQ_ATOMIC_DRAM_32B_sum

BW_HBM (for gfx942) = 128 × TCC_BUBBLE_sum + 
                              32 × TCC_EA0_RDREQ_32B_sum + 
                              64 × (TCC_EA0_RDREQ_sum - TCC_BUBBLE_sum - TCC_EA0_RDREQ_32B_sum) + 
                              32 × (TCC_EA0_WRREQ_sum - TCC_EA0_WRREQ_64B_sum) + 
                              64 × TCC_EA0_WRREQ_64B_sum

BW_HBM (for gfx90a) = 32 × TCC_EA_RDREQ_32B_sum + 
                      64 × (TCC_EA_RDREQ_sum - TCC_EA_RDREQ_32B_sum) + 
                      32 × (TCC_EA_WRREQ_sum - TCC_EA_WRREQ_64B_sum) + 
                      64 × TCC_EA_WRREQ_64B_sum
```

**Hardware Counters**:
- `TCC_EA0_WRREQ_WRITE_DRAM_32B_sum`: 32-byte write request to HBM. 1 64-byte request will be counted as 2.
- `TCC_EA0_RDREQ_DRAM_32B_sum`: 32-byte read request to HBM. 1 64-byte request will be counted as 2, 128-byte as 4.
- `TCC_EA0_WRREQ_ATOMIC_DRAM_32B_sum`: 32-byte atomic request to HBM. 1 64-byte request will be counted as 2.
- `TCC_BUBBLE_sum`: TCC (L2 cache controller) bubble cycles
- `TCC_EA0_RDREQ_sum`, `TCC_EA_RDREQ_sum`: Total read requests to HBM
- `TCC_EA0_RDREQ_32B_sum`, `TCC_EA_RDREQ_32B_sum`: 32-byte read requests to HBM
- `TCC_EA0_WRREQ_sum`, `TCC_EA_WRREQ_sum`: Total write requests to HBM
- `TCC_EA0_WRREQ_64B_sum`, `TCC_EA_WRREQ_64B_sum`: 64-byte write requests to HBM

#### Arithmetic Intensity (L2 Cache)
```
AI_L2_TOT = TOTAL_OPS / BW_L2
```

Where:
```
BW_L2 (preferred) = 128 × TCC_REQ_sum

BW_L2 (fallback) = 64 × TCP_TCC_READ_REQ_sum + 
                   64 × TCP_TCC_WRITE_REQ_sum + 
                   BW_LDS_ATOMICS
```

**Hardware Counters**:
- `TCC_REQ_sum`: Total requests to L2 cache (preferred, more reliable)
- `TCP_TCC_READ_REQ_sum`: Read requests from TCP to TCC
- `TCP_TCC_WRITE_REQ_sum`: Write requests from TCP to TCC

#### Arithmetic Intensity (vL1D - Vector L1 Data Cache)
```
AI_vL1d_TOT = TOTAL_OPS / BW_vL1d
```

Where:
```
BW_vL1d (preferred) = 256 × (SQ_INSTS_VMEM_WR + SQ_INSTS_VMEM_RD)

BW_vL1d (fallback) = 128 × TCP_TOTAL_CACHE_ACCESSES_sum
```

**Hardware Counters**:
- `SQ_INSTS_VMEM_WR`: Vector memory write instructions
- `SQ_INSTS_VMEM_RD`: Vector memory read instructions
- `TCP_TOTAL_CACHE_ACCESSES_sum`: Total cache accesses (fallback)

#### Arithmetic Intensity (LDS - Local Data Share)
```
AI_LDS_TOT = TOTAL_OPS / BW_LDS
```

Where:
```
BW_LDS = 32 × 4 × (SQ_LDS_IDX_ACTIVE - SQ_LDS_BANK_CONFLICT)

BW_LDS_ATOMICS = 64 × (TCP_TCC_ATOMIC_WITH_RET_REQ_sum + TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum)
```

**Hardware Counters**:
- `SQ_LDS_IDX_ACTIVE`: Active LDS index operations
- `SQ_LDS_BANK_CONFLICT`: LDS bank conflicts
- `TCP_TCC_ATOMIC_WITH_RET_REQ_sum`: Atomic operations with return value
- `TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum`: Atomic operations without return value

### Performance Limiter

**Description**: Identifies which resource (compute or memory bandwidth) is the bottleneck for the kernel.

**Equation**:
```
LIMITER = argmin(HBM_BW_PEAK, L2_BW_PEAK, vL1d_BW_PEAK, LDS_BW_PEAK, KERNEL_COMPUTE_PEAK)
```

Where:
- `HBM_BW_PEAK = AI_HBM_TOT × HBM_BANDWIDTH`
- `L2_BW_PEAK = AI_L2_TOT × L2_BANDWIDTH`
- `vL1d_BW_PEAK = AI_vL1d_TOT × vL1d_BANDWIDTH`
- `LDS_BW_PEAK = AI_LDS_TOT × LDS_BANDWIDTH`
- `KERNEL_COMPUTE_PEAK` = Calculated based on operation mix and architecture-specific peak performance

**Architecture-Specific Bandwidths (GB/s)**:
| Memory Level | MI250X | MI300A | MI300X | MI355X |
|-------------|--------|--------|--------|--------|
| HBM         | 1340   | 3688   | 4224   | 6198   |
| L2          | 5019   | 20343  | 26655  | 34800  |
| vL1d        | 9217   | 26092  | 34191  | 38263  |
| LDS         | 20816  | 57574  | 75657  | 67368  |

### Peak Achievable Throughput

**Description**: The maximum theoretical throughput (in GFLOPS/s) that this kernel can achieve on the target architecture, limited by its bottleneck resource.

**Equation**:
```
PEAK = min(HBM_BW_PEAK, L2_BW_PEAK, vL1d_BW_PEAK, LDS_BW_PEAK, KERNEL_COMPUTE_PEAK)
```

Where `KERNEL_COMPUTE_PEAK` is calculated as:

```
KERNEL_COMPUTE_PEAK = TOTAL_OPS / (
    64 × SQ_INSTS_VALU_ADD_F16 / fp16_add_peak +
    64 × SQ_INSTS_VALU_MUL_F16 / fp16_mul_peak +
    64 × 2 × SQ_INSTS_VALU_FMA_F16 / fp16_muladd_peak +
    64 × SQ_INSTS_VALU_TRANS_F16 / fp16_trans_peak +
    [similar terms for F32, F64, INT32, INT64] +
    TOTAL_MOPS_F16 / fp16_mfma_peak +
    TOTAL_MOPS_BF16 / bf16_mfma_peak +
    TOTAL_MOPS_F32 / fp32_mfma_peak +
    TOTAL_MOPS_F64 / fp64_mfma_peak
)
```

The `*_peak` values are architecture-specific peak performance values from `benchWarmer.csv`.

**Architecture-Specific Compute Peaks (GFLOPS/s)**:
| Architecture | Compute Peak |
|-------------|--------------|
| MI250X      | 36111        |
| MI300A      | 78716        |
| MI300X      | 94602        |
| MI355X      | 126857       |

### Distance from Roofline

**Description**: The percentage of the roofline peak throughput that the kernel is actually achieving. Higher values indicate better performance optimization.

**Equation**:
```
PercentAchieved = (Throughput / PEAK) × 100
```

Where:
- `Throughput` = Achieved throughput (see below)
- `PEAK` = Peak achievable throughput

**Average Distance from Roofline (Application-wide)**:
```
Average Distance = Σ(Percentage_i × PercentAchieved_i) / 100
```

Where the sum is over all kernels, weighted by their runtime contribution.

### Achieved Throughput

**Description**: The actual throughput (in GFLOPS/s) achieved by the kernel during execution.

**Equation**:
```
Throughput = TOTAL_OPS / DurationNs
```

Where:
- `TOTAL_OPS` = Total operations per dispatch
- `DurationNs` = Duration of the kernel dispatch in nanoseconds

**Units**: GFLOPS/s (billions of operations per second)

---

## Operation Categories

### Scalar ALU Operations (SALU)
Hardware counter: `SQ_INSTS_SALU`
- Operations performed by the scalar ALU
- Typically used for address calculations and control flow

### Vector ALU Operations (VALU)

#### FP16 Operations
- **Add**: `SQ_INSTS_VALU_ADD_F16` (× 64 ops/instruction)
- **Multiply**: `SQ_INSTS_VALU_MUL_F16` (× 64 ops/instruction)
- **FMA (Fused Multiply-Add)**: `SQ_INSTS_VALU_FMA_F16` (× 128 ops/instruction, counting both multiply and add)
- **Transcendental**: `SQ_INSTS_VALU_TRANS_F16` (× 64 ops/instruction)

#### FP32 Operations
- **Add**: `SQ_INSTS_VALU_ADD_F32` (× 64 ops/instruction)
- **Multiply**: `SQ_INSTS_VALU_MUL_F32` (× 64 ops/instruction)
- **FMA**: `SQ_INSTS_VALU_FMA_F32` (× 128 ops/instruction)
- **Transcendental**: `SQ_INSTS_VALU_TRANS_F32` (× 64 ops/instruction)

#### FP64 Operations
- **Add**: `SQ_INSTS_VALU_ADD_F64` (× 64 ops/instruction)
- **Multiply**: `SQ_INSTS_VALU_MUL_F64` (× 64 ops/instruction)
- **FMA**: `SQ_INSTS_VALU_FMA_F64` (× 128 ops/instruction)
- **Transcendental**: `SQ_INSTS_VALU_TRANS_F64` (× 64 ops/instruction)

#### Integer Operations
- **INT32**: `SQ_INSTS_VALU_INT32` (× 64 ops/instruction)
- **INT64**: `SQ_INSTS_VALU_INT64` (× 64 ops/instruction)

### Matrix Operations (MFMA)

Matrix Fused Multiply-Add operations, each counting 512 operations per instruction:

- **FP8**: `SQ_INSTS_VALU_MFMA_MOPS_F8` (× 512 ops/instruction, if available)
- **INT8**: `SQ_INSTS_VALU_MFMA_MOPS_I8` (× 512 ops/instruction, if available)
- **FP16**: `SQ_INSTS_VALU_MFMA_MOPS_F16` (× 512 ops/instruction)
- **BF16**: `SQ_INSTS_VALU_MFMA_MOPS_BF16` (× 512 ops/instruction)
- **FP32**: `SQ_INSTS_VALU_MFMA_MOPS_F32` (× 512 ops/instruction)
- **FP64**: `SQ_INSTS_VALU_MFMA_MOPS_F64` (× 512 ops/instruction)

---

## Bandwidth Calculations

### Memory Hierarchy

1. **HBM (High Bandwidth Memory)**: Main GPU memory
   - Highest capacity, lowest bandwidth
   - Measured by tracking TCC (L2 cache controller) to memory controller requests
   
2. **L2 Cache**: Shared across compute units
   - Intermediate bandwidth
   - Measured by tracking TCP (Texture Cache Pipe) to TCC requests
   
3. **vL1D (Vector L1 Data Cache)**: Per-compute-unit cache
   - Higher bandwidth than L2
   - Measured by vector memory instructions or cache accesses
   
4. **LDS (Local Data Share)**: Per-workgroup shared memory
   - Highest bandwidth, lowest capacity
   - Measured by LDS index operations minus bank conflicts

### Bandwidth Units

All bandwidth values are in bytes. The arithmetic intensities are in ops/byte (FLOP/byte).

---

## Hardware Counters Reference

### Counter Naming Convention

- **SQ_**: Shader Quad (compute unit instruction-level counters)
- **TCC_**: L2 Cache Controller
- **TCP_**: Texture Cache Pipe (L1/L2 interface)
- **_sum**: Aggregated across all compute units

### Full Counter List

#### Compute Counters
- `SQ_INSTS_SALU`: Scalar ALU instructions
- `SQ_INSTS_VALU`: Total vector ALU instructions
- `SQ_INSTS_VALU_*`: Vector ALU instructions by type and precision
- `SQ_INSTS_VALU_MFMA_MOPS_*`: Matrix multiply-accumulate operations
- `SQ_INSTS_VMEM_WR`: Vector memory write instructions
- `SQ_INSTS_VMEM_RD`: Vector memory read instructions

#### Memory Counters
- `SQ_LDS_IDX_ACTIVE`: Active LDS operations
- `SQ_LDS_BANK_CONFLICT`: LDS bank conflicts
- `TCP_TCC_READ_REQ_sum`: L1 to L2 read requests
- `TCP_TCC_WRITE_REQ_sum`: L1 to L2 write requests
- `TCP_TCC_ATOMIC_WITH_RET_REQ_sum`: Atomic operations with return
- `TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum`: Atomic operations without return
- `TCP_TOTAL_CACHE_ACCESSES_sum`: Total cache accesses
- `TCC_REQ_sum`: Total L2 requests (preferred counter)
- `TCC_BUBBLE_sum`: L2 bubble cycles
- `TCC_EA0_RDREQ_sum` / `TCC_EA_RDREQ_sum`: L2 to HBM read requests
- `TCC_EA0_RDREQ_32B_sum` / `TCC_EA_RDREQ_32B_sum`: 32-byte read requests to HBM
- `TCC_EA0_WRREQ_sum` / `TCC_EA_WRREQ_sum`: L2 to HBM write requests
- `TCC_EA0_WRREQ_64B_sum` / `TCC_EA_WRREQ_64B_sum`: 64-byte write requests to HBM

### Architecture Differences

- **gfx90a (MI250X)**: Uses `TCC_EA_*` counters for HBM tracking
- **gfx942 (MI300A/X)**: Uses `TCC_EA0_*` counters and includes `TCC_BUBBLE_sum`
- **gfx950 (MI355X)**: Similar to gfx942, uses `TCC_EA0_*` counters

---

## Example Interpretation

Consider a kernel with:
- `AI_HBM_TOT = 5.0` ops/byte
- `Performance Limiter = HBM (MI300X)`
- `Peak Achievable Throughput = 21120 GFLOPS/s`
- `Achieved Throughput = 19008 GFLOPS/s`
- `Distance from Roofline = 90%`

**Interpretation**:
1. The kernel performs 5 operations for every byte transferred to/from HBM
2. HBM bandwidth is the bottleneck (not compute)
3. Given the HBM bandwidth constraint, the kernel could theoretically achieve 21.1 TFLOPS/s
4. The kernel is achieving 19.0 TFLOPS/s, which is 90% of the theoretical peak
5. To improve performance further:
   - Reduce HBM traffic (increase data reuse, improve locality)
   - Or accept that 90% efficiency is near-optimal for this memory access pattern

---

## References

- ROCm Profiler Documentation: https://rocm.docs.amd.com/projects/rocprofiler/
- AMD GPU Architecture Specifications
- benchWarmer.csv: Contains architecture-specific peak performance values
