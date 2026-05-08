# Roofline Extractor Metrics - Quick Reference

This document provides high-level descriptions of each metric displayed in the roofline extractor command-line output. For detailed equations and hardware counter information, see [METRICS_DETAILED.md](METRICS_DETAILED.md).

---
## Data Sources
The following data and their sources/tools are used to generate the metrics:
- Compute counters: rocprofv3
- Memory counters: rocprofv3
- Kernel timing: rocprofv3 (independent run from counters)
- Roofline peaks
  - Primary option
    - benchWarmer (https://github.com/AMD-HPC-Internal/benchWarmer)
    - rocm-amdgpu-bench (https://github.com/ROCm/rocm-amdgpu-bench)
  - Secondary option (only used when hardware isn't available)
    - GPU specs pulled from internal spreadsheets
    - Roofline extractor uses achievable peaks for VALU throughput and bandwidths, and theoretical peaks for MFMA throughput. We've found that this setup has the most consistent results.

---

## Per-Kernel Output Metrics

When you run roofline extractor, the following metrics are displayed for each kernel:

### Basic Execution Metrics

**Total operations per dispatch**
- The total number of floating-point and integer operations executed in a single kernel launch
- Includes scalar, vector, and matrix operations across all data types (FP16, FP32, FP64, INT32, INT64, etc.)
- Collected from rocprofv3

**Average runtime per dispatch**
- The average execution time for a single kernel launch, measured in nanoseconds
- Collected from rocprofv3 (independent run from counter collection)

**Total runtime**
- The cumulative execution time across all launches of this kernel

**Total contribution to runtime**
- The percentage of total application runtime consumed by this kernel
- Kernels with contribution below the threshold (default 10%) are omitted from the detailed output

**Total dispatches**
- The number of times this kernel was launched during application execution

### Memory and Compute Characteristics

**Arithmetic intensity (HBM, L2, L1, LDS)**
- The ratio of operations performed to bytes transferred for each memory hierarchy level
- Measured in operations per byte (ops/byte or FLOP/byte)
- Higher values indicate more compute-intensive workloads
- Four values are shown for:
  - **HBM**: High Bandwidth Memory (main GPU memory)
  - **L2**: L2 cache (shared across compute units)
  - **L1**: L1 data cache (per compute unit)
  - **LDS**: Local Data Share (per workgroup shared memory)

**Performance limiter**
- Identifies which resource is the bottleneck for the kernel
- Can be compute-bound (limited by peak compute performance) or memory-bound (limited by bandwidth at HBM, L2, L1, or LDS)
- Shows the architecture being analyzed (e.g., "HBM (MI300X)")

### Performance Metrics

**Peak achievable throughput**
- The maximum theoretical throughput this kernel can achieve on the target architecture
- Measured in GFLOPS/s (billions of operations per second)
- Limited by the performance bottleneck (compute or memory bandwidth)

**Achieved throughput**
- The actual throughput achieved during kernel execution
- Measured in GFLOPS/s

**Percent of roofline achieved**
- The percentage of roofline peak throughput that the kernel is actually achieving
- Higher percentages indicate better optimization (closer to theoretical peak)
- Calculated as: (Achieved throughput / Peak achievable throughput) × 100%

---

## Application-Wide Summary Metrics

After displaying per-kernel metrics, the tool shows application-wide summaries:

**Total application runtime**
- The sum of all kernel execution times for the application

**Average distance from roofline**
- A weighted average of all kernels' distance from roofline values
- Weighted by each kernel's contribution to total runtime
- Provides a single metric for overall application optimization level

---

## Interpreting the Metrics

### Understanding Your Kernel's Performance

1. **Check the performance limiter**: This tells you whether your kernel is compute-bound or memory-bound
   - If compute-bound: Focus on algorithmic improvements or better utilize specialized hardware (e.g., matrix engines)
   - If memory-bound: Focus on reducing memory traffic or improving data locality

2. **Examine arithmetic intensity**: Higher is generally better for compute-intensive applications
   - Low AI (<1 ops/byte): Memory-bound, focus on data reuse
   - Medium AI (1-10 ops/byte): Balanced, may benefit from both compute and memory optimizations
   - High AI (>10 ops/byte): Compute-bound, focus on instruction-level optimization

3. **Review distance from roofline**: Indicates how close you are to theoretical peak
   - \>80%: Excellent optimization
   - 60-80%: Good, but potentially some room for improvement
   - <60%: Significant optimization opportunities likely exist
   
   _Some kernels have a bottleneck that isn't memory bandwidth or compute. These kernels will be further from the roofline, but may not have room for optimization. Examples include memory latency, low occupancy, and atomics._

4. **Analyze runtime contribution**: Focus optimization efforts on kernels with high runtime contribution
   - Kernels below the threshold (default 10%) have minimal impact on overall performance

### Example Interpretation

```
kernel_name
  Total operations per dispatch:  5.12e+09 ops
  Average runtime per dispatch:   2.00e+05 ns
  Total runtime:                  4.00e+06 ns
  Total contribution to runtime:  25.5 %
  Total dispatches:               20

  Arithmetic intensity (HBM):   4.5
  Arithmetic intensity (L2):    8.2
  Arithmetic intensity (L1):    12.1
  Arithmetic intensity (LDS):   25.0

  Performance limiter:              HBM (MI300X)
  Peak achievable throughput:       19008.00 GFLOPS/s
  Percentage of Roofline Achieved:  85.5 %

  Achieved throughput:          16252.80 GFLOPS/s
```

**Interpretation:**
- This kernel consumes 25.5% of total application runtime (significant)
- HBM bandwidth is the bottleneck (not compute)
- The kernel achieves 85.5% of its theoretical peak (well-optimized)
- AI of 4.5 ops/byte at HBM suggests moderate data reuse
- To improve further: Consider increasing data reuse to reduce HBM traffic or accept that 85.5% is near-optimal for this memory access pattern

---

## Additional Resources

- **Detailed documentation**: See [METRICS.md](METRICS.md) for complete equations and hardware counter mappings
- **ROCm profiler documentation**: https://rocm.docs.amd.com/projects/rocprofiler/
- **Hardware specifications**: Architecture-specific bandwidth and compute peak values are documented in METRICS.md
