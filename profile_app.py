# Copyright (C) 2026 Advanced Micro Devices, Inc.
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file # or at https://opensource.org/licenses/MIT.

#!/usr/bin/env python3
"""
Script to run rocprofv3 with both counter collection and kernel tracing,
then analyze the results with rooflineExtractor.

Usage:
    python profile_app.py [-o OUTPUT_DIR] --arch ARCH -- <run_command> [args...]

Example:
    python profile_app.py --arch MI300X -- ./my_app arg1 arg2
    python profile_app.py -o ./results --arch MI300X -- ./my_app -v --debug

Default output directory is ./data/output/
The --arch flag is required; pass it explicitly to select the GPU architecture.

Note:
    The script attempts to use the -f csv flag with rocprofv3. This flag is not
    recognized in versions of ROCm older than ROCm 7. If the flag is not recognized,
    the script will automatically retry without it.
"""

import sys
import subprocess
import os
import argparse
import glob
import shlex
import shutil
import logging
import pdb
from pathlib import Path
from datetime import datetime

# Dependencies of convert-counters-collection-format.py
import pandas as pd

# Dependencies of rooflineExtractor.py
import numpy as np


class LogWriter:
    """Helper class to write output to both console and log file."""
    def __init__(self, log_file_path):
        self.log_file = open(log_file_path, 'w')
        self.log_file_path = log_file_path
    
    def write(self, message, end='\n', file=None):
        """Write message to both console and log file."""
        # Write to console
        if file == sys.stderr:
            print(message, end=end, file=sys.stderr)
        else:
            print(message, end=end)
        
        # Write to log file
        self.log_file.write(message + end)
        self.log_file.flush()
    
    def close(self):
        """Close the log file."""
        self.log_file.close()


# Known MPI/job launchers whose presence indicates rocprofv3 must be invoked
# *per rank* (with a unique output directory) rather than wrapping the launcher.
MPI_LAUNCHERS = {"mpirun", "mpiexec", "srun", "aprun", "jsrun"}

# Common launcher options that consume the next argv element as their value.
# Used to skip past launcher options when locating the application binary.
_LAUNCHER_VALUED_FLAGS = {
    "-n", "-np", "--n", "--np",
    "-c", "--cpus-per-task", "--cpus-per-rank",
    "-N", "--ntasks-per-node", "--ntasks",
    "-x",
    "-host", "--host", "-H", "-hostfile", "--hostfile",
    "-machinefile", "--machinefile",
    "-bind-to", "--bind-to", "-map-by", "--map-by",
    "-mca", "--mca", "-gmca", "--gmca",
    "-rf", "-rankfile", "--rankfile",
    "-output-filename", "--output-filename",
    "-am", "--am", "-tune", "--tune",
}

# Shell fragment that derives a per-rank identifier from whichever MPI/job
# env var the active launcher sets, falling back to the shell PID. Stored in
# ``$rank`` for use later in the wrapper.
_RANK_RESOLVE = (
    'rank="${OMPI_COMM_WORLD_RANK:-'
    '${PMI_RANK:-${PMIX_RANK:-${SLURM_PROCID:-'
    '${MV2_COMM_WORLD_RANK:-$$}}}}}"; '
)


def split_launcher(run_command):
    """Detect a leading MPI launcher in ``run_command``.

    Returns ``(launcher_args, app_args)`` if the first token is a known
    launcher, otherwise ``(None, run_command)``. ``launcher_args`` includes
    the launcher itself and all of its options up to (but not including) the
    application binary.
    """
    if not run_command:
        return None, run_command
    launcher_basename = os.path.basename(run_command[0])
    if launcher_basename not in MPI_LAUNCHERS:
        return None, run_command
    i = 1
    while i < len(run_command):
        arg = run_command[i]
        if arg.startswith("-"):
            if "=" in arg:
                # --flag=value style, single token
                i += 1
                continue
            if arg in _LAUNCHER_VALUED_FLAGS:
                i += 2
                continue
            # Treat unknown dash-options as boolean flags.
            i += 1
            continue
        # First non-flag token = application binary.
        return run_command[:i], run_command[i:]
    # All tokens consumed as launcher options; no app found. Don't wrap.
    return None, run_command


def build_rocprof_command(run_command, rocprof_args, per_rank_parent=None):
    """Build the command list to invoke rocprofv3.

    When ``run_command`` starts with an MPI launcher, rocprofv3 is placed
    *inside* the launcher via a small ``bash -c`` wrapper. The wrapper
    derives a per-rank id (falling back to the shell PID) and, if
    ``per_rank_parent`` is given, ``cd``s into ``<per_rank_parent>/rank_<id>``
    before exec'ing rocprofv3. This ensures rocprofv3's native ``pmc_N/``
    output layout lands in a unique directory per rank.

    Without an MPI launcher, rocprofv3 wraps ``run_command`` directly
    (legacy behavior); ``per_rank_parent`` is ignored.

    ``rocprof_args`` is the argv passed to rocprofv3 (excluding the program
    name and the trailing ``--`` separator). All file paths inside it must
    be absolute, since the wrapper may change the working directory.
    """
    launcher_args, app_args = split_launcher(run_command)
    if launcher_args is None:
        return ["rocprofv3"] + rocprof_args + ["--"] + run_command
    quoted = " ".join(shlex.quote(a) for a in rocprof_args)
    cd_fragment = ""
    if per_rank_parent is not None:
        quoted_parent = shlex.quote(per_rank_parent)
        cd_fragment = (
            f'outdir={quoted_parent}/rank_"$rank"; '
            'mkdir -p "$outdir" && cd "$outdir" || exit 1; '
        )
    wrapper = (
        _RANK_RESOLVE
        + cd_fragment
        + f'exec rocprofv3 {quoted} -- "$@"'
    )
    # The token immediately after ``-c <script>`` becomes ``$0`` inside the
    # script; the rest become ``$1``, ``$2``, ... which we forward via "$@".
    return launcher_args + ["bash", "-c", wrapper, "bash"] + app_args


def run_rocprofv3_with_retry(cmd_with_csv, cmd_without_csv, cwd, logger=None):
    """
    Run rocprofv3 command with -f csv flag. If it fails due to unrecognized flag,
    retry without the -f csv flag.
    
    Args:
        cmd_with_csv: Command list with -f csv included
        cmd_without_csv: Command list without -f csv
        cwd: Working directory for the command
        logger: LogWriter instance for logging output
        
    Returns:
        subprocess.CompletedProcess result
    """
    def stream_output(cmd, cwd):
        """Helper to stream output in real-time and capture it for error checking."""
        stdout_lines = []
        stderr_lines = []
        
        # Start process with pipes for stdout and stderr.
        # Use errors='replace' so binary or invalid-UTF-8 output from rocprofv3/benchmark
        # does not raise UnicodeDecodeError (e.g. byte 0xff in counter CSV stream).
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace',
            bufsize=1  # Line buffered
        )
        
        # Use select or threads to handle both stdout and stderr
        import selectors
        import io
        
        sel = selectors.DefaultSelector()
        sel.register(process.stdout, selectors.EVENT_READ)
        sel.register(process.stderr, selectors.EVENT_READ)
        
        # Track which streams are still open
        streams_open = {process.stdout, process.stderr}
        
        while streams_open:
            # Wait for data to be available
            for key, _ in sel.select(timeout=0.1):
                stream = key.fileobj
                line = stream.readline()
                
                if line:
                    if stream == process.stdout:
                        stdout_lines.append(line)
                        if logger:
                            logger.write(line, end='')
                        else:
                            print(line, end='')
                        sys.stdout.flush()
                    else:  # stderr
                        stderr_lines.append(line)
                        if logger:
                            logger.write(line, end='', file=sys.stderr)
                        else:
                            print(line, end='', file=sys.stderr)
                        sys.stderr.flush()
                else:
                    # Stream closed
                    sel.unregister(stream)
                    streams_open.discard(stream)
        
        # Wait for process to finish
        returncode = process.wait()
        
        # Create a result object similar to subprocess.CompletedProcess
        class Result:
            def __init__(self, returncode, stdout, stderr):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr
        
        return Result(returncode, ''.join(stdout_lines), ''.join(stderr_lines))
    
    # Try with -f csv first - stream output in real-time
    result = stream_output(cmd_with_csv, cwd)
    
    # Check if the command failed due to unrecognized -f flag
    if result.returncode != 0 and result.stderr:
        stderr_lower = result.stderr.lower()
        # Check for common error messages indicating the flag is not recognized
        flag_not_recognized = any(phrase in stderr_lower for phrase in [
            'unrecognized option',
            'invalid option',
            'unknown option',
            'unrecognized argument',
            'invalid argument',
            'unknown argument',
            '-f',
            'usage:',
        ])
        
        if flag_not_recognized and '-f' in result.stderr:
            # Print retry message
            if logger:
                logger.write("Note: -f csv flag not recognized. Retrying with pre-ROCm 7 syntax...")
                logger.write(f"Retry command: {' '.join(cmd_without_csv)}")
            else:
                print(f"Note: -f csv flag not recognized. Retrying with pre-ROCm 7 syntax...")
                print(f"Retry command: {' '.join(cmd_without_csv)}")
            
            # Retry without -f csv - stream output in real-time
            result = stream_output(cmd_without_csv, cwd)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run rocprofv3 with counter collection and kernel tracing',
        usage='%(prog)s [-o OUTPUT_DIR] [--arch ARCH] -- run_command [args...]'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='data/output',
        help='Directory for output files (default: ./data/output)'
    )
    parser.add_argument(
        '--arch',
        required=False,
        help='Supply architecture (to aid in guided analysis). Options: MI250, MI250X, MI300A, MI300X, MI325X, MI350X, MI355X.'
    )
    parser.add_argument(
        'run_command',
        nargs=argparse.REMAINDER,
        help='Command to profile and its arguments (use -- before command)'
    )
    
    args = parser.parse_args()
    output_dir = args.output_dir
    run_command = args.run_command

    # Validate --arch against same sources as rooflineExtractor.py
    script_dir = Path(__file__).parent
    _caches_path = script_dir / "config" / "cache_bandwidths.csv"
    _caches_df = pd.read_csv(_caches_path, index_col="Architecture")
    supported_arch = set(_caches_df.index)
    if args.arch is None:
        parser.error(
            f"--arch is required. Supported: {', '.join(sorted(supported_arch))}"
        )

    # Normalize --arch (same as rooflineExtractor.py)
    args.arch = args.arch.replace('_', ' ').upper()
    if args.arch not in supported_arch:
        parser.error(
            f"Unsupported architecture '{args.arch}'. Supported: {', '.join(sorted(supported_arch))}"
        )

    # Prepend ./data/ to output_dir if it's not already there and not an absolute path
    if not os.path.isabs(output_dir) and not output_dir.startswith('data/') and not output_dir.startswith('./data/'):
        output_dir = os.path.join('data', output_dir)
    
    # Remove the '--' separator if present
    if run_command and run_command[0] == '--':
        run_command = run_command[1:]
    
    # Validate that a command was provided
    if not run_command:
        parser.error("No run command provided. Use: profile_app.py [-o OUTPUT_DIR] -- run_command [args...]")
    
    # Map GPU model to gfx architecture for counter file selection
    gpu_to_gfx = {
        'MI250': 'gfx90a',
        'MI250X': 'gfx90a',
        'MI300A': 'gfx942',
        'MI300X': 'gfx942',
        'MI325X': 'gfx942',
        'MI350X': 'gfx950',
        'MI355X': 'gfx950'
    }
    
    detected_gpu = args.arch.upper()
    if detected_gpu not in gpu_to_gfx:
        parser.error(
            f"Architecture '{detected_gpu}' has no associated gfx target. "
            f"Supported: {', '.join(sorted(gpu_to_gfx.keys()))}."
        )
    gfx_arch = gpu_to_gfx[detected_gpu]
    gpu_message = f"Using user-specified architecture: {detected_gpu} ({gfx_arch})"
    
    # Save original working directory
    original_dir = os.getcwd()
    
    # Convert output_dir to absolute path
    output_path = Path(__file__).parent / output_dir
    output_dir = os.path.abspath(output_path)
    
    # Convert input counter file to absolute path (use selected architecture)
    roof_path = Path(__file__).parent / f'roof-counters-{gfx_arch}.txt'
    counter_input_file = os.path.abspath(roof_path)
    
    # Convert conversion script to absolute path
    convert_path = Path(__file__).parent / f'convert-counters-collection-format.py'
    conversion_script = os.path.abspath(convert_path)
    
    # Convert rooflineExtractor script to absolute path
    roofline_path = Path(__file__).parent / f'rooflineExtractor.py'
    roofline_script = os.path.abspath(roofline_path)
    
    # Convert all relative file paths in run_command to absolute paths
    # This ensures that when we change directory, the paths still work
    for i in range(len(run_command)):
        # Check if this argument looks like a file/directory that exists
        if os.path.exists(run_command[i]) and not os.path.isabs(run_command[i]):
            run_command[i] = os.path.abspath(run_command[i])
    
    run_command_str = ' '.join(run_command)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(output_dir, f'profile_{timestamp}.log')
    logger = LogWriter(log_file_path)
    
    logger.write(f"Profiling session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.write(gpu_message)
    logger.write(f"Run command: {run_command_str}")
    logger.write(f"Output directory: {output_dir}")
    logger.write(f"Log file: {log_file_path}")
    logger.write("=" * 80)

    # Clean up any rocprofv3 output left over from a previous run in the same
    # output directory. Stale pmc_*/ counter dirs or mpi_counters/mpi_trace
    # trees would otherwise get mixed with (or block moving in) the new
    # results. This intentionally leaves logs and other files untouched.
    stale_entries = sorted(glob.glob(os.path.join(output_dir, 'pmc_*')))
    for name in ('mpi_counters', 'mpi_trace'):
        candidate = os.path.join(output_dir, name)
        if os.path.exists(candidate):
            stale_entries.append(candidate)
    if stale_entries:
        logger.write("Removing existing rocprofv3 output from previous run(s)...")
        for entry in stale_entries:
            logger.write(f"  Removing {entry}")
            if os.path.isdir(entry) and not os.path.islink(entry):
                shutil.rmtree(entry)
            else:
                os.remove(entry)
        logger.write("=" * 80)
    
    # Detect MPI launcher so we can profile each rank separately. Running
    # rocprofv3 around mpirun (single profiler wrapping the launcher) causes
    # every rank to write the same counters_counter_collection.csv, which
    # corrupts the file via interleaved writes.
    mpi_used = split_launcher(run_command)[0] is not None
    if mpi_used:
        logger.write(
            "Detected MPI launcher; profiling each rank separately with per-rank rocprofv3 outputs."
        )

    # Run rocprofv3 with counter collection
    # Base the message on the number of pmc passes in the selected counter input file
    with open(counter_input_file, 'r', encoding='utf-8') as counter_file:
        n_runs = sum(1 for line in counter_file if line.strip().startswith('pmc:'))
    logger.write(f"\n[1/4] Running rocprofv3 with counter collection ({n_runs} runs of the application)...")

    counter_rocprof_args_csv = ['-i', counter_input_file, '-o', 'counters', '-f', 'csv']
    counter_rocprof_args_nocsv = ['-i', counter_input_file, '-o', 'counters']
    counter_per_rank_parent = 'mpi_counters' if mpi_used else None

    counter_cmd_with_csv = build_rocprof_command(
        run_command, counter_rocprof_args_csv, per_rank_parent=counter_per_rank_parent
    )
    counter_cmd_without_csv = build_rocprof_command(
        run_command, counter_rocprof_args_nocsv, per_rank_parent=counter_per_rank_parent
    )

    logger.write(f"Command: {' '.join(counter_cmd_with_csv)}")
    logger.write("-" * 80)
    try:
        result1 = run_rocprofv3_with_retry(counter_cmd_with_csv, counter_cmd_without_csv, None, logger)
        if result1.returncode != 0:
            logger.write(f"Error: Counter collection failed with exit code {result1.returncode}")
            logger.close()
            sys.exit(1)
        else:
            logger.write("Counter collection completed successfully")
            if mpi_used:
                # Move the per-rank parent directory (containing rank_*/pmc_*/...)
                if os.path.isdir('mpi_counters'):
                    dest_root = os.path.join(output_dir, 'mpi_counters')
                    logger.write(f"Moving mpi_counters to {dest_root}")
                    if os.path.isdir(dest_root):
                        for item in os.listdir('mpi_counters'):
                            src_item = os.path.join('mpi_counters', item)
                            dest_item = os.path.join(dest_root, item)
                            logger.write(f"  Moving {src_item} to {dest_item}")
                            shutil.move(src_item, dest_item)
                        os.rmdir('mpi_counters')
                    else:
                        shutil.move('mpi_counters', dest_root)
            else:
                # Move counter output directories to output_dir
                for counter_dir in glob.glob('pmc_*'):
                    logger.write(f"Moving {counter_dir} to {output_dir}")
                    try:
                        shutil.move(counter_dir, output_dir)
                    except shutil.Error as e:
                        # If the destination exists, move files individually
                        if "already exists" in str(e):
                            logger.write(f"Warning: {counter_dir} already exists in {output_dir}, moving files individually.")
                            dest_dir = os.path.join(output_dir, os.path.basename(counter_dir))
                            for item in os.listdir(counter_dir):
                                src_item = os.path.join(counter_dir, item)
                                dest_item = os.path.join(dest_dir, item)
                                logger.write(f"  Moving {src_item} to {dest_item}")
                                shutil.move(src_item, dest_item)
                            # Remove the now-empty source directory
                            os.rmdir(counter_dir)
                        else:
                            raise
                # Also move any other counters* files that may exist
                for counter_file in glob.glob('counters*'):
                    dest = os.path.join(output_dir, counter_file)
                    logger.write(f"Moving {counter_file} to {dest}")
                    shutil.move(counter_file, dest)
    except FileNotFoundError as e:
        # The missing executable is whatever subprocess tried to launch first,
        # i.e. the MPI launcher (mpirun/srun/...) when wrapping an MPI job, or
        # rocprofv3 otherwise. Report the actual name from the exception.
        missing = getattr(e, 'filename', None) or counter_cmd_with_csv[0]
        logger.write(f"Error: {missing} not found. Make sure it's installed and in your PATH.")
        logger.close()
        sys.exit(1)
    except Exception as e:
        logger.write(f"Error running counter collection: {e}")
        logger.close()
        sys.exit(1)
    
    logger.write("=" * 80)
    
    # Convert counter collection format
    logger.write("\n[2/4] Converting counter collection format...")
    counters_csv_path = os.path.join(output_dir, 'counters.csv')
    convert_cmd = ['python3', conversion_script, '-i', output_dir, '-o', counters_csv_path]
    logger.write(f"Command: python3 {' '.join(convert_cmd[1:])}")
    logger.write("-" * 80)
    
    try:
        result2 = subprocess.run(convert_cmd, check=False,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                universal_newlines=True)
        
        # Log output
        if result2.stdout:
            logger.write(result2.stdout, end='')
        if result2.stderr:
            logger.write(result2.stderr, end='', file=sys.stderr)
        
        if result2.returncode != 0:
            logger.write(f"Error: Counter conversion failed with exit code {result2.returncode}")
            logger.close()
            sys.exit(1)
        else:
            logger.write("Counter conversion completed successfully")
    except FileNotFoundError:
        logger.write(f"Error: convert_counters_collection_format.py not found. Expected it to be in the same directory as profile_app.py (path: {conversion_script}).")
        logger.close()
        sys.exit(1)
    except Exception as e:
        logger.write(f"Error running counter conversion: {e}")
        logger.close()
        sys.exit(1)
    
    logger.write("=" * 80)
    
    # Run rocprofv3 with kernel tracing
    logger.write("\n[3/4] Running rocprofv3 with kernel tracing (one run of the application)...")

    trace_rocprof_args_csv = ['--kernel-trace', '-o', 'trace', '-f', 'csv']
    trace_rocprof_args_nocsv = ['--kernel-trace', '-o', 'trace']
    trace_per_rank_parent = 'mpi_trace' if mpi_used else None

    trace_cmd_with_csv = build_rocprof_command(
        run_command, trace_rocprof_args_csv, per_rank_parent=trace_per_rank_parent
    )
    trace_cmd_without_csv = build_rocprof_command(
        run_command, trace_rocprof_args_nocsv, per_rank_parent=trace_per_rank_parent
    )

    logger.write(f"Command: {' '.join(trace_cmd_with_csv)}")
    logger.write("-" * 80)
    try:
        result3 = run_rocprofv3_with_retry(trace_cmd_with_csv, trace_cmd_without_csv, None, logger)
        if result3.returncode != 0:
            logger.write(f"Error: Kernel tracing failed with exit code {result3.returncode}")
            logger.close()
            sys.exit(1)
        else:
            logger.write("Kernel tracing completed successfully")
            if mpi_used:
                # Move the per-rank trace tree under output_dir
                if os.path.isdir('mpi_trace'):
                    dest_root = os.path.join(output_dir, 'mpi_trace')
                    logger.write(f"Moving mpi_trace to {dest_root}")
                    if os.path.isdir(dest_root):
                        for item in os.listdir('mpi_trace'):
                            src_item = os.path.join('mpi_trace', item)
                            dest_item = os.path.join(dest_root, item)
                            logger.write(f"  Moving {src_item} to {dest_item}")
                            shutil.move(src_item, dest_item)
                        os.rmdir('mpi_trace')
                    else:
                        shutil.move('mpi_trace', dest_root)
                # Concatenate per-rank trace CSVs into a single trace_kernel_trace.csv
                # so the downstream rooflineExtractor step finds the file it expects.
                per_rank_traces = sorted(glob.glob(os.path.join(
                    output_dir, 'mpi_trace', 'rank_*', 'trace_kernel_trace.csv')))
                if per_rank_traces:
                    logger.write(
                        f"Merging {len(per_rank_traces)} per-rank kernel-trace CSV(s) into trace_kernel_trace.csv"
                    )
                    per_rank_frames = []
                    for p in per_rank_traces:
                        rank_df = pd.read_csv(p)
                        # Tag with the rank (parent dir name, e.g. "rank_0") so the
                        # downstream rooflineExtractor merge keys on Application and
                        # keeps each rank's Dispatch_Id distinct. This must match the
                        # Application tag added to the per-rank counters.
                        rank_df['Application'] = os.path.basename(os.path.dirname(p))
                        per_rank_frames.append(rank_df)
                    merged = pd.concat(per_rank_frames, ignore_index=True)
                    merged.to_csv(
                        os.path.join(output_dir, 'trace_kernel_trace.csv'), index=False
                    )
                else:
                    logger.write(
                        "Warning: no per-rank trace_kernel_trace.csv files found to merge."
                    )
            else:
                # Move trace output files to output_dir
                for trace_file in glob.glob('trace*'):
                    dest = os.path.join(output_dir, trace_file)
                    logger.write(f"Moving {trace_file} to {dest}")
                    shutil.move(trace_file, dest)
    except Exception as e:
        logger.write(f"Error running kernel tracing: {e}")
        logger.close()
        sys.exit(1)
    
    logger.write("=" * 80)
    
    # Run rooflineExtractor
    logger.write("\n[4/4] Running rooflineExtractor...")
    
    # Build roofline command with arch flags
    roofline_cmd = ['python3', roofline_script, '-c', os.path.join(output_dir, 'counters.csv'), '-r', os.path.join(output_dir, 'trace_kernel_trace.csv'), '-p', '-d', '--arch', detected_gpu]
    
    logger.write(f"Command: python3 {' '.join(roofline_cmd[1:])}")
    logger.write("-" * 80)
    try:
        result4 = subprocess.run(roofline_cmd, cwd=original_dir, check=False,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                universal_newlines=True)
        
        # Log output
        if result4.stdout:
            logger.write(result4.stdout, end='')
        if result4.stderr:
            logger.write(result4.stderr, end='', file=sys.stderr)
        
        if result4.returncode != 0:
            logger.write(f"Error: rooflineExtractor failed with exit code {result4.returncode}")
            logger.close()
            sys.exit(1)
        else:
            logger.write("rooflineExtractor completed successfully")
    except FileNotFoundError:
        logger.write("Error: rooflineExtractor.py not found in current directory.")
        logger.close()
        sys.exit(1)
    except Exception as e:
        logger.write(f"Error running rooflineExtractor: {e}")
        logger.close()
        sys.exit(1)
    
    logger.write("=" * 80)
    logger.write("\nAll profiling and conversion steps completed!")
    logger.write(f"Output files in {output_dir}:")
    
    # List of output files to check
    output_files = [
        ("Kernel runtime trace", os.path.join(output_dir, 'trace_kernel_trace.csv')),
        ("Hardware counters", os.path.join(output_dir, 'counters.csv')),
        ("Roofline statistics per kernel dispatch", os.path.join(output_dir, 'counters_EXTRACTED.csv')),
        ("Roofline statistics (aggregated dispatches)", os.path.join(output_dir, 'counters_EXTRACTED_AGG.csv')),
        ("Roofline plot", os.path.join(output_dir, 'counters.html')),
        ("Full console log", log_file_path)
    ]
    
    # Only print files that exist
    for description, file_path in output_files:
        if os.path.exists(file_path):
            logger.write(f"  - {description:<45}: {file_path}")
    
    logger.write(f"\nProfiling session ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.close()


if __name__ == "__main__":
    main()

