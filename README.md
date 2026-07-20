# Roofline Extractor
Roofline Extractor is a tool that calculates the percent of the empirical peak performance that an application is achieving on a per-kernel basis.

## Install Python Packages
Run this to install all necessary packages:
```
pip install -r requirements.txt
```

## Option 1: Automated Profiling (Recommended)
The easiest way to use rooflineExtractor is with the `profile_app.py` script, which automates all the profiling steps:

```bash
python3 profile_app.py --arch <ARCH> -- <app exe> [args...]
```

This script will:
1. Use the GPU architecture you pass via `--arch` (one of MI250, MI250X, MI300A, MI300X, MI325X, MI350X, MI355X) to pick the matching counter input file
2. Run `rocprofv3` to collect hardware counters (3-4 runs of the application)
3. Perform post-processing on the counter data
4. Run `rocprofv3` to collect kernel trace data (1 run of the application)
5. Run `rooflineExtractor.py` to generate analysis and plots

**Note:** The script uses the `-f csv` flag with rocprofv3, which is only available in ROCm 7 and later. If the flag is not recognized, the script will automatically retry without it.

Required flags:
* `--arch [ARCH]`: Current GPU architecture. Options: MI250, MI250X, MI300A, MI300X, MI325X, MI350X, MI355X

Optional flags:
* `-o [OUTPUT_DIR]`: Specify output directory (default: `./output/`)

**Example:**
```bash
python3 profile_app.py -o my_results --arch MI355X -- ./my_app
```

All output files (counters, traces, plots, analysis) will be saved in the specified output directory.

### Multi-Rank (MPI) Runs
`profile_app.py` handles multi-rank / multi-process MPI jobs automatically. Just place your normal MPI launch command after the `--` separator, exactly as you would run the application yourself:

```bash
python3 profile_app.py -o my_mpi_results -- mpirun -np 4 ./my_app arg1 arg2
```

Supported launchers (auto-detected as the first token of the run command): `mpirun`, `mpiexec`, `srun`, `aprun`, `jsrun`.

**How it works:** Wrapping `rocprofv3` *around* the launcher would make every rank write to the same output file, producing corrupt, interleaved CSVs. To avoid this, when a launcher is detected the script instead invokes `rocprofv3` *inside* the launcher (once per rank). Each rank derives a unique id from its MPI environment (e.g. `OMPI_COMM_WORLD_RANK`, `PMI_RANK`, `PMIX_RANK`, `SLURM_PROCID`, `MV2_COMM_WORLD_RANK`, falling back to the shell PID) and writes to its own directory:

```text
my_mpi_results/
  mpi_counters/
    rank_0/pmc_1/..._counter_collection.csv
    rank_1/pmc_1/..._counter_collection.csv
    ...
  mpi_trace/
    rank_0/trace_kernel_trace.csv
    rank_1/trace_kernel_trace.csv
    ...
```

The script then:
- Merges the per-rank kernel-trace CSVs into a single `trace_kernel_trace.csv`.
- Combines all per-rank `*_counter_collection.csv` files (found recursively under `mpi_counters/`) into a single `counters.csv` during the conversion step.
- Runs `rooflineExtractor.py` on the combined data, so all ranks are analyzed together as one workload.

**Notes:**
- Pass the launcher and all of its flags (e.g. `-np`, `--ntasks`, `-host`, `--bind-to`) after `--`; they are forwarded unchanged to the launcher.

## Option 2: Manual Profiling
If you prefer to run the profiling steps manually:

### Data Generation
The following two runs of rocprofv3 are needed to use rooflineExtractor:
* Counters
  * This run gathers counters for the application. Pick the input file from this directory that is appropriate for your architecture.
  * `rocprofv3 -i roof-counters-<arch>.txt -f csv -- <app exe>`
    * **Note:** The `-f csv` flag is only recognized in ROCm 7 and later. For older versions, omit this flag.
    * If using rocprofv3, another command is needed to consolidate its output into a single file: `python3 convert-conters-collection-format.py -i <path to rocprofv3 output files> -o <singular output file>`
* Runtime stats
  * This run gathers timing information for the application
  * `rocprofv3 --kernel-trace -f csv -- <exe>`
    * **Note:** The `-f csv` flag is only recognized in ROCm 7 and later. For older versions, omit this flag.

> **Multi-rank (MPI) runs:** Do **not** wrap the MPI launcher (e.g. `rocprofv3 ... -- mpirun ...`), because every rank would write to the same output file and corrupt it. Instead, invoke `rocprofv3` *inside* the launcher so each rank profiles itself into a unique output directory, e.g. `mpirun -np 4 rocprofv3 -i roof-counters-<arch>.txt -o counters -f csv -d rank_${OMPI_COMM_WORLD_RANK} -- <exe>`. Then point `convert-counters-collection-format.py` at the parent directory (it collects every `pmc_*/*counter_collection.csv` recursively) and concatenate the per-rank trace CSVs before running `rooflineExtractor.py`. The automated `profile_app.py` flow (see [Multi-Rank (MPI) Runs](#multi-rank-mpi-runs)) does all of this for you.

### Run rooflineExtractor (single application)
Provide counter and trace CSVs with **`-c`** and **`-r`**, and the GPU architecture with **`--arch`**:

`python3 rooflineExtractor.py -c [roof-counters.csv] -r [trace or results CSV] --arch [ARCH]`

Required flags:
* `--arch [ARCH]`: Current GPU architecture. Options: MI250, MI250X, MI300A, MI300X, MI325X, MI350X, MI355X

Additional optional flags:
* `--plot`: Generate plots
* `--dump`: Dump per-dispatch and aggregate dataframes to `*_EXTRACTED.csv` and `*_EXTRACTED_AGG.csv` (next to the counter path stem)
* `--sig-runtime [% runtime]`: Specify what's the minimum percent runtime for a kernel to be considered "significant" and be included in analysis. Defaults to 10%.

### Multi-application combined analysis (`--directory`)

To analyze **several applications as one workload** (e.g. multiple jobs or phases), use **`-D` / `--directory`** instead of **`-c`** and **`-r`**.

**Layout:** point **`-D`** at a parent directory. Each **immediate subdirectory** that contains both of these files is loaded as one application:

| File | Role |
|------|------|
| `counters.csv` | Hardware counters (same format as single-app **`-c`**) |
| `trace_kernel_trace.csv` | Kernel trace / timing (same format as single-app **`-r`**) |

Subdirectories missing either file are skipped (with a message). The subdirectory name is stored as an **`Application`** column so counter rows stay matched to the correct trace; aggregation and roofline logic then treat all kernels together, as if they were one application.

**Outputs** (plot HTML and **`--dump`** CSVs) are written **inside the directory passed to `-D`**, using the parent folder’s name as the file stem, e.g.:

- `my_bundle/my_bundle.html`
- `my_bundle/my_bundle_EXTRACTED.csv`
- `my_bundle/my_bundle_EXTRACTED_AGG.csv`

If you pass **`-c`/`-r` while using `-D`**, those single-file arguments are ignored.

**Example:**

```text
my_bundle/
  job_a/counters.csv
  job_a/trace_kernel_trace.csv
  job_b/counters.csv
  job_b/trace_kernel_trace.csv
```

```bash
python3 rooflineExtractor.py -D my_bundle --plot --dump --arch MI300X
```

### Example (single application)
Here is an example using nbody-nvidia-mini with **rocprofv3**.
```
# Collect kernel counters with rocprofv3 (using -f csv if supported by your ROCm version)
rocprofv3 -i roof-counters-gfx942.txt -o counters -f csv -- ./nbody-orig 1048576

# Collect runtime stats with rocprofv3 (using -f csv if supported by your ROCm version)
rocprofv3 --kernel-trace -o trace -f csv -- ./nbody-orig 1048576

# Convert the hardware counter collection output to CSV (needed for rooflineExtractor)
python3 convert-counters-collection-format.py -i . -o counters.csv

# Run rooflineExtractor to generate plots and dataframes
python3 rooflineExtractor.py -c counters.csv -r trace_kernel_trace.csv --plot --dump --arch MI300X
```

## Output:
* A guided analysis via the terminal showing per-kernel performance metrics including arithmetic intensity, roofline peaks, and percentage of roofline achieved
* An HTML file with an interactive roofline plot showing the performance and arithmetic intensity of each kernel instance (with **`--plot`**)
* CSV dumps of per-dispatch and aggregate metrics (with **`--dump`**)
* **Single-app mode (`-c`/`-r`):** HTML and CSV names follow the counter file path (same directory as the counter stem unless you use a path prefix).
* **Multi-app mode (`-D`):** HTML and CSV files are written under the directory given to **`-D`** (see **Multi-application combined analysis** above).

## Metrics Documentation
* **Quick Reference**: [METRICS_SUMMARY.md](METRICS_SUMMARY.md) - High-level descriptions of each output metric and how to interpret them
* **Detailed Documentation**: [METRICS_DETAILED.md](METRICS_DETAILED.md) - Complete equations, hardware counters, and technical details for all metrics
