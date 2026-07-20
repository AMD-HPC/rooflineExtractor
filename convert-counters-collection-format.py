# Copyright (C) 2026 Advanced Micro Devices, Inc.
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file # or at https://opensource.org/licenses/MIT.

#!/usr/bin/env python3
import os
import re
import pandas as pd
import argparse
import logging


# rocprofv3 writes per-rank MPI output under ``.../rank_<N>/pmc_*/...`` (see
# profile_app.py's build_rocprof_command). Detect that rank id from the file
# path so each rank's counters can be qualified with an ``Application`` column,
# which keeps otherwise-identical Dispatch_Id values from different ranks from
# collapsing together downstream.
_RANK_DIR_RE = re.compile(r"(?:^|/)rank_(\d+)(?:/|$)")


def rank_from_path(file_path):
    match = _RANK_DIR_RE.search(file_path.replace(os.sep, "/"))
    if match:
        return f"rank_{match.group(1)}"
    return None


def read_csv(file_path):
    df = pd.DataFrame()
    skipped = []

    def on_bad_line(line):
        skipped.append(line)
        return None

    # rocprofv3 writes counter rows from multiple GPUs concurrently, which
    # can occasionally interleave two records onto a single physical line.
    # Skip those corrupt lines instead of failing the whole conversion.
    #
    # The API for handling bad lines differs by pandas version:
    #   - pandas >= 1.4 supports a callable passed to on_bad_lines
    #   - pandas 1.3.x supports on_bad_lines="skip"
    #   - pandas < 1.3 uses error_bad_lines/warn_bad_lines
    # Try each in turn so the conversion works across versions.
    try:
        df = pd.read_csv(file_path, engine="python", on_bad_lines=on_bad_line)
    except TypeError:
        try:
            df = pd.read_csv(file_path, engine="python", on_bad_lines="skip")
        except TypeError:
            df = pd.read_csv(
                file_path,
                engine="python",
                error_bad_lines=False,
                warn_bad_lines=False,
            )
    except Exception as e:
        logging.info(f"Error reading {file_path}: {e}")
        raise
    if skipped:
        logging.warning(
            f"Skipped {len(skipped)} malformed line(s) in {file_path}"
        )
    return df


def get_counter_collection_files(root_path):
    file_paths = []
    for root, _, files in os.walk(root_path):
        if "pmc_" in root:
            for file in files:
                if file.endswith("counter_collection.csv"):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
    return file_paths


def get_combined_df(args):
    files_list = []
    for input in args.input:
        if os.path.isfile(input):
            files_list.append(input)
        elif os.path.isdir(input):
            files_list.extend(get_counter_collection_files(input))
    if not files_list:
        raise ValueError("Valid Input files not found")
    logging.info(f"Processing files: {files_list}")
    combined_df = pd.DataFrame()
    for file in files_list:
        file_df = read_csv(file)
        # Qualify each row with its MPI rank (when the file lives under a
        # rank_<N> directory) so per-rank kernels stay distinct after merging.
        rank = rank_from_path(file)
        if rank is not None:
            file_df["Application"] = rank
        combined_df = pd.concat([combined_df, file_df], ignore_index=True)
    return combined_df


def write_to_file(df, args):
    logging.info(f"Saving output file to : {args.output}")
    directory, file_path = os.path.split(args.output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_csv(args.output, index=False)


def main(args):
    logging.basicConfig(level=args.loglevel)
    input_df = get_combined_df(args)
    # Validate
    columns = [
        "Correlation_Id",
        "Dispatch_Id",
        "Agent_Id",
        "Queue_Id",
        "Process_Id",
        "Thread_Id",
        "Grid_Size",
        "Kernel_Id",
        "Kernel_Name",
        "Workgroup_Size",
        "LDS_Block_Size",
        "Scratch_Size",
        "VGPR_Count",
        "SGPR_Count",
        "Counter_Name",
        "Counter_Value",
        "Start_Timestamp",
        "End_Timestamp",
    ]
    for col in input_df.columns:
        if col not in columns:
            logging.debug(f"Unexpected column {col} found in rocprofv3 input file")

    non_index_columns = [
        "Correlation_Id",
        "Start_Timestamp",
        "End_Timestamp",
        "Process_Id",
        "Thread_Id",
        "Kernel_Id",
    ]

    # Convert
    indexes = [
        "Dispatch_Id",
        "Agent_Id",
        "Grid_Size",
        "Kernel_Name",
        "LDS_Block_Size",
        "Queue_Id",
        "SGPR_Count",
        "Scratch_Size",
        "VGPR_Count",
        "Workgroup_Size",
    ]

    # rocprofv3 emits the numeric index columns with inconsistent dtypes across
    # PMC passes (e.g. Scratch_Size as int "0" in one pass and float "0.0" or
    # string in another). After concatenation those columns become object dtype
    # with mixed values that never compare equal, so the pivot would treat each
    # pass as a separate group and fail to merge counters for the same kernel.
    # Coerce them to a single nullable-integer type so identical kernels align.
    numeric_indexes = [
        "Dispatch_Id",
        "Grid_Size",
        "LDS_Block_Size",
        "Queue_Id",
        "SGPR_Count",
        "Scratch_Size",
        "VGPR_Count",
        "Workgroup_Size",
    ]
    for col in numeric_indexes:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce").astype(
                "Int64"
            )

    # Keep the per-rank Application tag (added in get_combined_df for MPI runs)
    # as part of the pivot index so it survives into the output and qualifies
    # each rank's Dispatch_Id downstream.
    if "Application" in input_df.columns:
        indexes = ["Application"] + indexes

    # Drop duplicate counters in multiple PMC lines
    input_df.drop_duplicates(
        subset=indexes + ["Counter_Name"], keep="first", inplace=True
    )

    pivoted_data = input_df.pivot_table(
        index=indexes, columns="Counter_Name", values="Counter_Value", aggfunc="sum"
    ).reset_index()

    # Save
    write_to_file(pivoted_data, args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Rocprofv3 Counter Collection input files and/or directories containing `*counter_collection.csv` files",
        nargs="+",
        default=[],
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Rocprofv1 formatted output file name",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Debug Logs",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose Logs",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
