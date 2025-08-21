import pandas as pd
from tabulate import tabulate
import numpy as np
import abc
import time
import pdb
import sys
import argparse
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy.stats import gmean

plt.rcParams.update({'font.size': 12})

sigRuntime = 10  # Default value, can be changed from command line
numCUs = 228  # total compute units on MI300A
peakBandwidth = 3700  # peak bandwidth estimate

# Get filenames from input args
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--counter', required=True, help='Provide roof_counters.csv filename from rocprof -i')
parser.add_argument('-r', '--results', required=True, help='Provide results.csv filename from rocprof --stats')
parser.add_argument('-m', '--hw-metrics', help='Provide hw_metrics.csv filename from rocSTAR')
parser.add_argument('--sig-runtime', required=False, default=10, type=float, help='Provide the percentage of runtime considered "significant" for analysis (kernels under that percentage will be omitted)')
parser.add_argument('-p', '--plot', action='store_true', required=False, help='Generate plots')
parser.add_argument('-d', '--dump', action='store_true', required=False, help='Dump DataFrame to csv')

args = parser.parse_args()




roofCountFilename = ""
statFlename = ""
runtimeFilename = ""
hwFilename = ""


last_dot_index = args.counter.rfind('.')
roofCountFilename = args.counter[:last_dot_index] # everything before .csv
last_dot_index = args.results.rfind('.')
runtimeFilename = args.results.rfind('.')  # everything before .csv
if args.hw_metrics != None:
    last_dot_index = args.hw_metrics.rfind('.')
    hwFilename = args.hw_metrics[:last_dot_index]  # everything before .csv

sigRuntime = args.sig_runtime

# Load roof counters into pandas df
df_roof = pd.read_csv(args.counter)


# Check if empty
if df_roof.empty:
  print('Input roof counters file is empty')
  quit() 
# Check for wrong file
if "CompleteNs" in df_roof.columns:
    print('Error: "results.csv" log file submitted with "-c" flag, which is for "roof-counters.csv" log file')
    quit()

# Check for 'None' values and remove them
total_none_kernels = max(df_roof.isnull().any(axis=1).sum(), (df_roof == 'None').any(axis=1).sum())

if total_none_kernels > 0:
    print(f'{total_none_kernels} kernels had None values for some of the counters, which is a sign that runs of the application are non-deterministic. Removing these kernels and attempting to continue')
    df_roof = df_roof[~(df_roof == 'None').any(axis=1)]
    df_roof = df_roof[~(df_roof.isnull()).any(axis=1)]


# Function to convert columns with type mismatches to integers
def convert_columns_to_int(df):
    counters = ['SQ_INSTS_VALU_ADD_F16', 'SQ_INSTS_VALU_MUL_F16',       'SQ_INSTS_VALU_FMA_F16', 'SQ_INSTS_VALU_TRANS_F16',       'SQ_INSTS_VALU_ADD_F32', 'SQ_INSTS_VALU_MUL_F32',       'SQ_INSTS_VALU_FMA_F32', 'SQ_INSTS_VALU_TRANS_F32',       'SQ_INSTS_VALU_ADD_F64', 'SQ_INSTS_VALU_MUL_F64',       'SQ_INSTS_VALU_FMA_F64', 'SQ_INSTS_VALU_TRANS_F64',       'SQ_INSTS_VALU_MFMA_MOPS_F16', 'SQ_INSTS_VALU_MFMA_MOPS_BF16',       'SQ_INSTS_VALU_MFMA_MOPS_F32', 'SQ_INSTS_VALU_MFMA_MOPS_F64', 'SQ_LDS_IDX_ACTIVE', 'SQ_LDS_BANK_CONFLICT',       'TCP_TCC_READ_REQ_sum', 'TCP_TCC_WRITE_REQ_sum',       'TCP_TCC_ATOMIC_WITH_RET_REQ_sum', 'TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum',       'TCP_TOTAL_CACHE_ACCESSES_sum', 'SQ_INSTS_VALU_INT32',       'SQ_INSTS_VALU_INT64', 'SQ_INSTS_VALU_CVT', 'SQ_INSTS_SALU']
    if 'SQ_INSTS_VALU' in df.columns:
        counters.append('SQ_INSTS_VALU')
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

    for counter in counters:
        df[counter] = pd.to_numeric(df[counter], errors='coerce').astype(int)
    return df
df_roof = convert_columns_to_int(df_roof)

# Bandwidths (MI250, MI300A, MI300x)
caches = {
    "MI250":  {"HBM": 1340,  "L2": 5019,  "vL1d": 9217,  "LDS": 20816},
    "MI300A": {"HBM": 3688,  "L2": 20343, "vL1d": 26092, "LDS": 57574},
    "MI300x": {"HBM": 4224,  "L2": 26655, "vL1d": 34191, "LDS": 75657},
}

# Roofline numbers (peak bandwidth, peak compute)
compute_peaks = {
    "MI250":  36111,
    "MI300A": 78716,
    "MI300x": 94602,
}

# Compute total flops, AI's
def computeFlops(df):

    # Compute total achieved FLOPs for each datatype (FP16, FP32, FP64)
    ## Scalar Ops
    df['TOTAL_SALU'] = df['SQ_INSTS_SALU']

    ## Vector Ops
    df['TOTAL_VALU_F16'] = 64 * (df['SQ_INSTS_VALU_ADD_F16'] + df['SQ_INSTS_VALU_MUL_F16'] + df['SQ_INSTS_VALU_TRANS_F16'] + 2 * df['SQ_INSTS_VALU_FMA_F16'])
    df['TOTAL_VALU_F32'] = 64 * (df['SQ_INSTS_VALU_ADD_F32'] + df['SQ_INSTS_VALU_MUL_F32'] + df['SQ_INSTS_VALU_TRANS_F32'] + 2 * df['SQ_INSTS_VALU_FMA_F32'])
    df['TOTAL_VALU_F64'] = 64 * (df['SQ_INSTS_VALU_ADD_F64'] + df['SQ_INSTS_VALU_MUL_F64'] + df['SQ_INSTS_VALU_TRANS_F64'] + 2 * df['SQ_INSTS_VALU_FMA_F64'])
    df['TOTAL_VALU_I32'] = 64 * df['SQ_INSTS_VALU_INT32']
    df['TOTAL_VALU_I64'] = 64 * df['SQ_INSTS_VALU_INT64']

    ## Matrix Ops
    df['TOTAL_MOPS_F16'] = 512 * df['SQ_INSTS_VALU_MFMA_MOPS_F16']
    df['TOTAL_MOPS_BF16'] = 512 * df['SQ_INSTS_VALU_MFMA_MOPS_BF16']
    df['TOTAL_MOPS_F32'] = 512 * df['SQ_INSTS_VALU_MFMA_MOPS_F32']
    df['TOTAL_MOPS_F64'] = 512 * df['SQ_INSTS_VALU_MFMA_MOPS_F64']

    # Other VALU Ops (e.g. Int16, Int8)
    if 'SQ_INSTS_VALU' in df.columns:
        df['TOTAL_VALU_OTHER'] = 64 * (df['SQ_INSTS_VALU'] - (df['SQ_INSTS_VALU_ADD_F16'] + df['SQ_INSTS_VALU_MUL_F16'] + df['SQ_INSTS_VALU_TRANS_F16'] + df['SQ_INSTS_VALU_FMA_F16'] + df['SQ_INSTS_VALU_ADD_F32'] + df['SQ_INSTS_VALU_MUL_F32'] + df['SQ_INSTS_VALU_TRANS_F32'] + df['SQ_INSTS_VALU_FMA_F32'] + df['SQ_INSTS_VALU_ADD_F64'] + df['SQ_INSTS_VALU_MUL_F64'] + df['SQ_INSTS_VALU_TRANS_F64'] + df['SQ_INSTS_VALU_FMA_F64'] + df['SQ_INSTS_VALU_INT32'] + df['SQ_INSTS_VALU_INT64'] + df['SQ_INSTS_VALU_MFMA_MOPS_F16'] + df['SQ_INSTS_VALU_MFMA_MOPS_BF16'] + df['SQ_INSTS_VALU_MFMA_MOPS_F32'] + df['SQ_INSTS_VALU_MFMA_MOPS_F64']))


    ## Total
    df['TOTAL_OPS'] = df['TOTAL_SALU'] + df['TOTAL_VALU_F16'] + df['TOTAL_VALU_F32'] + df['TOTAL_VALU_F64'] + df['TOTAL_VALU_I32'] + df['TOTAL_VALU_I64'] + df['TOTAL_MOPS_F16'] + df['TOTAL_MOPS_BF16'] + df['TOTAL_MOPS_F32'] + df['TOTAL_MOPS_F64']
    if 'SQ_INSTS_VALU' in df.columns:
        df['TOTAL_OPS'] = df['TOTAL_OPS'] + df['TOTAL_VALU_OTHER']

    # Compute Bandwidths
    ## LDS
    df['BW_LDS'] = 32 * 4 * (df['SQ_LDS_IDX_ACTIVE'] - df['SQ_LDS_BANK_CONFLICT'])
    df['BW_LDS_ATOMICS'] = 64 * (df['TCP_TCC_ATOMIC_WITH_RET_REQ_sum'] + df['TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum'])

    ## L2
    df['BW_L2'] = 64 * df['TCP_TCC_READ_REQ_sum'] + 64 * df['TCP_TCC_WRITE_REQ_sum'] + df['BW_LDS_ATOMICS']

    ## vL1D
    df['BW_vL1d'] = 64 * df['TCP_TOTAL_CACHE_ACCESSES_sum']

    arch = ""

    ## HBM
    ### Check architecture
    if df.keys().str.contains('TCC_BUBBLE').sum() > 0:
        # We have a gfx942 or gfx950 arch counter file
        arch = "MI300x"
        df['BW_HBM'] = 128 * df['TCC_BUBBLE_sum'] + 32 * df['TCC_EA0_RDREQ_32B_sum'] + 64 * (df['TCC_EA0_RDREQ_sum'] - df['TCC_BUBBLE_sum'] - df['TCC_EA0_RDREQ_32B_sum']) + 32 * (df['TCC_EA0_WRREQ_sum'] - df['TCC_EA0_WRREQ_64B_sum']) + 64 * df['TCC_EA0_WRREQ_64B_sum']
    else:
        # Assuming gfx90a
        arch = "MI250"
        df['BW_HBM'] = 32 * df['TCC_EA_RDREQ_32B_sum'] + 64 * (df['TCC_EA_RDREQ_sum'] - df['TCC_EA_RDREQ_32B_sum']) + 32 * (df['TCC_EA_WRREQ_sum'] - df['TCC_EA_WRREQ_64B_sum']) + 64 * df['TCC_EA_WRREQ_64B_sum']

    # Compute AI for each part of memory hierarchy (HBM, L2, L1)
    ## LDS
    df['AI_LDS_TOT'] = df['TOTAL_OPS'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_LDS_SALU'] = df['TOTAL_SALU'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_LDS_VALU_F16'] = df['TOTAL_VALU_F16'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_LDS_VALU_F32'] = df['TOTAL_VALU_F32'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_LDS_VALU_F64'] = df['TOTAL_VALU_F64'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_LDS_VALU_I32'] = df['TOTAL_VALU_I32'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_LDS_VALU_I64'] = df['TOTAL_VALU_I64'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_LDS_MOPS_F16'] = df['TOTAL_MOPS_F16'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_LDS_MOPS_BF16'] = df['TOTAL_MOPS_BF16'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_LDS_MOPS_F32'] = df['TOTAL_MOPS_F32'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_LDS_MOPS_F64'] = df['TOTAL_MOPS_F64'].divide(df['BW_LDS']).replace(np.inf, 0).replace(np.nan, 0)
    ## L2
    df['AI_L2_TOT'] = df['TOTAL_OPS'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_L2_SALU'] = df['TOTAL_SALU'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_L2_VALU_F16'] = df['TOTAL_VALU_F16'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_L2_VALU_F32'] = df['TOTAL_VALU_F32'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_L2_VALU_F64'] = df['TOTAL_VALU_F64'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_L2_VALU_I32'] = df['TOTAL_VALU_I32'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_L2_VALU_I64'] = df['TOTAL_VALU_I64'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_L2_MOPS_F16'] = df['TOTAL_MOPS_F16'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_L2_MOPS_BF16'] = df['TOTAL_MOPS_BF16'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_L2_MOPS_F32'] = df['TOTAL_MOPS_F32'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_L2_MOPS_F64'] = df['TOTAL_MOPS_F64'].divide(df['BW_L2']).replace(np.inf, 0).replace(np.nan, 0)
    ## vL1D
    df['AI_vL1d_TOT'] = df['TOTAL_OPS'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_vL1d_SALU'] = df['TOTAL_SALU'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_vL1d_VALU_F16'] = df['TOTAL_VALU_F16'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_vL1d_VALU_F32'] = df['TOTAL_VALU_F32'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_vL1d_VALU_F64'] = df['TOTAL_VALU_F64'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_vL1d_VALU_I32'] = df['TOTAL_VALU_I32'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_vL1d_VALU_I64'] = df['TOTAL_VALU_I64'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_vL1d_MOPS_F16'] = df['TOTAL_MOPS_F16'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_vL1d_MOPS_BF16'] = df['TOTAL_MOPS_BF16'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_vL1d_MOPS_F32'] = df['TOTAL_MOPS_F32'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_vL1d_MOPS_F64'] = df['TOTAL_MOPS_F64'].divide(df['BW_vL1d']).replace(np.inf, 0).replace(np.nan, 0)
    ## HBM
    df['AI_HBM_TOT'] = df['TOTAL_OPS'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_HBM_SALU'] = df['TOTAL_SALU'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_HBM_VALU_F16'] = df['TOTAL_VALU_F16'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_HBM_VALU_F32'] = df['TOTAL_VALU_F32'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_HBM_VALU_F64'] = df['TOTAL_VALU_F64'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_HBM_VALU_I32'] = df['TOTAL_VALU_I32'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_HBM_VALU_I64'] = df['TOTAL_VALU_I64'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_HBM_MOPS_F16'] = df['TOTAL_MOPS_F16'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_HBM_MOPS_BF16'] = df['TOTAL_MOPS_BF16'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_HBM_MOPS_F32'] = df['TOTAL_MOPS_F32'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)
    df['AI_HBM_MOPS_F64'] = df['TOTAL_MOPS_F64'].divide(df['BW_HBM']).replace(np.inf, 0).replace(np.nan, 0)

    # Add columns for peaks
    df['HBM_BW_PEAK'] = df['AI_HBM_TOT'] * caches[arch]['HBM']
    df['L2_BW_PEAK'] = df['AI_L2_TOT'] * caches[arch]['L2']
    df['vL1d_BW_PEAK'] = df['AI_vL1d_TOT'] * caches[arch]['vL1d']
    df['LDS_BW_PEAK'] = df['AI_LDS_TOT'] * caches[arch]['LDS']
    df = pd.concat([df, pd.DataFrame({'COMPUTE_PEAK': [compute_peaks[arch]] * len(df)})], axis=1)

    # Determine performance peak/limiter
    df_peaks = df[['HBM_BW_PEAK','L2_BW_PEAK','vL1d_BW_PEAK','LDS_BW_PEAK','COMPUTE_PEAK']]
    peaks = df_peaks.where(df_peaks > 0).min(axis=1)
    limiters = df_peaks.where(df_peaks > 0).idxmin(axis=1).str[:-5] + f" ({arch})"

    df = pd.concat([df, pd.DataFrame({'PEAK': peaks, 'LIMITER': limiters})], axis=1)


    return df


df_roof = computeFlops(df_roof)  # Compute AI's for each kernel dispatch

# Check for rocprofv3 and convert to v1 format
if 'Agent_Id' in df_roof.columns:
    df_roof = df_roof.drop(columns=['Agent_Id'])
    df_roof = df_roof.rename(columns={'Dispatch_Id':'Index'})
    df_roof = df_roof.rename(columns={'Kernel_Name':'KernelName'})

# Aggregate kernels
df = df_roof.groupby('KernelName', sort=False).mean(numeric_only=True).reset_index()


# Get runtime stats
df_runtime = pd.read_csv(args.results)

if 'Kernel_Name' in df_runtime.columns:
    df_runtime = df_runtime.rename(columns={'Kernel_Name':'KernelName'})
    df_runtime['DurationNs'] = df_runtime['End_Timestamp'] - df_runtime['Start_Timestamp']
    df_runtime = df_runtime.rename(columns={'Dispatch_Id':'Index'})

totalRuntimes = df_runtime.groupby('KernelName')['DurationNs'].sum()
averageRuntimes = df_runtime.groupby('KernelName')['DurationNs'].mean()
percentRuntimes = df_runtime.groupby('KernelName').sum()['DurationNs']/df_runtime['DurationNs'].sum()*100

df = pd.merge(df, totalRuntimes, on='KernelName').rename(columns={"DurationNs":"RuntimeNs"})
df = pd.merge(df, averageRuntimes, on='KernelName').rename(columns={"DurationNs":"AverageNs"})
df = pd.merge(df, percentRuntimes, on='KernelName').rename(columns={"DurationNs":"Percentage"})
# Add column for number of kernels
df = df.merge(df_roof.groupby('KernelName', sort=False).size().reset_index(name='Count'), on='KernelName')

insigKernels = len(df[df['Percentage'] < sigRuntime])  # save this value for guided analysis

df_roof = df_roof.merge(df_runtime, on='Index')

# Plot the AIs
if args.plot:
    ## Extract/format the data for clustering
    columns = [c for c in df.columns if c.startswith('AI_')]
    p_df = df.filter(['KernelName'] + ['Percentage'] + columns, axis=1)
    p_df = p_df.sort_values(by='Percentage').tail(5)
    p_df.index = p_df['KernelName']
    p_df.drop('KernelName', axis=1, inplace=True)
    p_df = p_df.transpose()
    ax = p_df.plot(kind="bar")
    fig = ax.get_figure()
    fig.set_size_inches(24,12)
    fig.tight_layout(pad=3, rect=[0,0,0.7,0.98])
    #fig.subplots_adjust(bottom=0.15, left=0.05)
    ax.set_xlabel("<AI_MEM_COMPUTE_DATATYPE>")
    ax.set_ylabel("Arithmetic Intensity (Ops/Byte)")
    ax.set_title(roofCountFilename)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yscale('log')
    ax.legend(loc='upper left', bbox_to_anchor=(1.04,1))
    fig.savefig(roofCountFilename + '_plot.png')




# Get telemetry data
if args.hw_metrics != None:
    df_hw = pd.read_csv(args.hw_metrics)
    if df_hw.empty:
        print('Input hw metrics file is empty')
        quit()

    # Trim whitespace from the ends of column labels
    df_hw = df_hw.rename(columns={col: col.strip() for col in df_hw.columns})
    # Make name consistent with other dataframes
    df_hw.rename(columns={'Label':'KernelName'}, inplace=True)
    df_hw['KernelName'] = df_hw['KernelName'].str.strip()

    # Names of labels from hw_metrics.csv
    maxSocketTemp = 'Max Socket Temp (C)'
    maxVRTemp = 'Max VR Temp (C)'
    maxHBMTemp = 'Max HBM Temp (C)'
    socketPower = 'Socket Power (W)'
    uclk = 'Effective UCLK Frequency (Mhz)'
    gfx2 = 'XCC2 Effective GFXCLK Frequency (MHz)'
    gfx3 = 'XCC3 Effective GFXCLK Frequency (MHz)'
    gfx4 = 'XCC4 Effective GFXCLK Frequency (MHz)'
    gfx5 = 'XCC5 Effective GFXCLK Frequency (MHz)'
    gfx6 = 'XCC6 Effective GFXCLK Frequency (MHz)'
    gfx7 = 'XCC7 Effective GFXCLK Frequency (MHz)'
    cols=[maxSocketTemp, maxVRTemp, maxHBMTemp, socketPower, uclk, gfx2, gfx3, gfx4, gfx5, gfx6, gfx7]

    # Calculate graphics clock mean and max across all chiplets
    df_hw['GFX mean'] = df_hw[[gfx2, gfx3, gfx4, gfx5, gfx6, gfx7]].mean(axis=1)
    df_hw['GFX max'] = df_hw[[gfx2, gfx3, gfx4, gfx5, gfx6, gfx7]].max(axis=1)

    # Gather statistics
    mins = (df_hw.groupby('KernelName')[cols].min())
    mins.columns = [col + ' min' for col in mins.columns]
    mins.reset_index(inplace=True)
    maxes = (df_hw.groupby('KernelName')[cols].max())
    maxes.columns = [col + ' max' for col in maxes.columns]
    maxes.reset_index(inplace=True)
    means = (df_hw.groupby('KernelName')[cols].mean())
    means.columns = [col + ' mean' for col in means.columns]
    means.reset_index(inplace=True)
    stdevs = (df_hw.groupby('KernelName')[cols].std())
    stdevs.columns = [col + ' stdev' for col in stdevs.columns]
    stdevs.reset_index(inplace=True)

    # Merge telemetry statistics into main dataframe (keeping kernels that don't have telemetry data)
    df = pd.merge(df, mins, on='KernelName', how='left')
    df = pd.merge(df, maxes, on='KernelName', how='left')
    df = pd.merge(df, means, on='KernelName', how='left')
    df = pd.merge(df, stdevs, on='KernelName', how='left')

df.set_index('KernelName', inplace=True)



# Plot power, temperature, clocks
if args.plot and args.hw_metrics != None:
    ## Convert timestamp column to datetime format
    df_hw['Timestamp'] = pd.to_datetime(df_hw['Timestamp'], unit='ns')

    # Create two subplots stacked vertically, set size
    fig, host = plt.subplots(2, sharex=True, figsize=(21,12))
    fig.suptitle('Achieved/Peak Throughput and Telemetry Data', fontsize=24)  # Set title
    plt.grid(True)  # Turn on gridlines
    fig.subplots_adjust(right=0.75)  # New axes appear on the right

    ## Plot telemetry
    ### Clock axis
    host[1].scatter(df_hw['Timestamp'], df_hw[gfx2], color='blue', label="GFX Clock", s=2, marker='.')  # Create a scatter plot
    host[1].scatter(df_hw['Timestamp'], df_hw[uclk], color='cyan', label="Memory Clock", s=2, marker='.')  # Create a scatter plot
    host[1].set_xlabel('Timestamp')  # Set x-axis label
    host[1].set_ylabel('Clock Frequency (MHz)', color='blue')  # Set y-axis label
    host[1].tick_params(axis='y', colors='blue')

    ### Power axis
    power_axis = host[1].twinx()
    power_axis.scatter(df_hw['Timestamp'], df_hw[socketPower], color='green', label="Power", s=2, marker='.')  # Create a scatter plot
    power_axis.set_xlabel('Timestamp')  # Set x-axis label
    power_axis.set_ylabel('Socket Power (W)', color='green')  # Set y-axis label
    power_axis.tick_params(axis='y', colors='green')
    plt.axhline(550, color='green', linestyle='dashed')

    ### Temperature axis
    temp_axis = host[1].twinx()
    temp_axis.spines["right"].set_position(("axes", 1.04))
    temp_axis.scatter(df_hw['Timestamp'], df_hw[maxSocketTemp], color='red', label="Socket Temp", s=2, marker='.')  # Create a scatter plot
    temp_axis.set_xlabel('Timestamp')  # Set x-axis label
    temp_axis.set_ylabel('Temperature (C)', color='red')  # Set y-axis label
    temp_axis.tick_params(axis='y', colors='red')


    ## Plot achieved vs. peak throughput
    ### Calculate totals of operation types
    df_roof['TOT_ADD'] = df_roof['SQ_INSTS_VALU_ADD_F16'] + df_roof['SQ_INSTS_VALU_ADD_F32'] + df_roof['SQ_INSTS_VALU_ADD_F64']
    df_roof['TOT_MUL'] = df_roof['SQ_INSTS_VALU_MUL_F16'] + df_roof['SQ_INSTS_VALU_MUL_F32'] + df_roof['SQ_INSTS_VALU_MUL_F64']
    df_roof['TOT_TRANS'] = df_roof['SQ_INSTS_VALU_TRANS_F16'] + df_roof['SQ_INSTS_VALU_TRANS_F32'] + df_roof['SQ_INSTS_VALU_TRANS_F64']
    df_roof['TOT_FMA'] = df_roof['SQ_INSTS_VALU_FMA_F16'] + df_roof['SQ_INSTS_VALU_FMA_F32'] + df_roof['SQ_INSTS_VALU_FMA_F64']
    df_roof['TOT_VALU'] = df_roof['TOT_ADD'] + df_roof['TOT_MUL'] + df_roof['TOT_FMA'] + df_roof['TOT_TRANS']
    df_roof['FLOP_FACTOR'] = ((128 * (df_roof['TOT_ADD'] + df_roof['TOT_MUL'] + df_roof['TOT_TRANS'])) + (256 * df_roof['TOT_FMA'])) / df_roof['TOT_VALU']

    ### Get timing info from rocprof
    df_roof = pd.merge(df_roof, df_runtime[['Index', 'DurationNs']], on='Index')

    ### Calculate achieved and peak
    df_roof['ACHIEVED'] = df_roof['TOTAL_OPS'] / df_roof['DurationNs']
    df_roof.rename(columns={'Index': 'Instance Index'}, inplace=True)
    df_roof = pd.merge(df_roof, df_hw.groupby('Instance Index')['GFX max'].max(), on='Instance Index')
    df_roof['PEAK'] = pd.DataFrame({"compute":df_roof['GFX max']/1000*numCUs*df_roof['FLOP_FACTOR'], "memory":peakBandwidth*df_roof['AI_HBM_TOT']}).min(axis=1)
    df_hw = pd.merge(df_hw, df_roof[['Instance Index', 'ACHIEVED']], on='Instance Index')
    df_hw = pd.merge(df_hw, df_roof[['Instance Index', 'PEAK']], on='Instance Index')

    ### Plot
    host[0].scatter(df_hw['Timestamp'], df_hw['PEAK'], label="Peak Achievable Flops", marker='.', color='b')
    host[0].scatter(df_hw['Timestamp'], df_hw['ACHIEVED'], label="Flops Achieved", marker='.', color='orange')
    host[0].set_ylabel('Throughput (GFlops/s)')
    host[0].grid(True)
    host[0].set_ylim(bottom=0)

    ### Add legend
    legend_elements = [Line2D([0], [0], color='b', lw=1, label='GFX Clock'),
                       Line2D([0], [0], color='c', lw=1, label='Memory Clock'),
                       Line2D([0], [0], color='g', lw=1, label='Power'),
                       Line2D([0], [0], color='g', lw=1, linestyle='dashed', label='Power Limit'),
                       Line2D([0], [0], color='r', lw=1, label='Socket Temp')]
    host[1].legend(handles=legend_elements, loc='best')

    legend_elements = [Line2D([0], [0], color='b', lw=2, label='Theoretical Peak Flops'),
                       Line2D([0], [0], color='orange', lw=2, label='Flops Achieved')]
    host[0].legend(handles=legend_elements, loc='best')

    ## Save and close plot
    fig.tight_layout()
    plt.savefig(hwFilename + '_plot.png')
    plt.close()


# Prepare for analysis
df = df.sort_values(by='Percentage', ascending=False)
df_peaks = df[['HBM_BW_PEAK','L2_BW_PEAK','vL1d_BW_PEAK','LDS_BW_PEAK','COMPUTE_PEAK']]
if df.keys().str.contains('TCC_BUBBLE').sum() > 0:
    arch = "MI300x"
else:
    arch = "MI250"
df['LIMITER'] = df_peaks.where(df_peaks > 0).idxmin(axis=1).str[:-5] + " (" + arch + ")"
df['Throughput'] = df['TOTAL_OPS'] / df['AverageNs']
df['PercentAchieved'] = df['Throughput'] / df['PEAK'] * 100

# Guided analysis
print(f"Total unique kernels: {len(df)}")
print(f"Total kernel dispatches: {len(df_roof)}")
for index, row in df.iterrows():  # loop through each kernel 
    if row['Percentage'] < sigRuntime:
        continue
    print("\n" + index)
    print(f"  Total operations per dispatch:", "{:e}".format(row['TOTAL_OPS']), "ops")
    print(f"  Average runtime per dispatch: ", "{:e}".format(row['AverageNs']), "ns")
    print(f"  Total contribution to runtime: {round(row['Percentage'], 3)} %")
    print(f"  Total dispatches:              {int(row['Count'])}")

    print(f"\n  Arithmetic intensity (HBM):  ", round(row['AI_HBM_TOT'], 4))
    print(f"  Arithmetic intensity (L2):   ", round(row['AI_L2_TOT'], 4))
    print(f"  Arithmetic intensity (L1):   ", round(row['AI_vL1d_TOT'], 4))
    print(f"  Arithmetic intensity (LDS):  ", round(row['AI_LDS_TOT'], 4))

    print(f"\n  Performance limiter:         ", row['LIMITER'])
    print(f"  Peak achievable throughput:  ", round(row['PEAK'], 2), "GFLOPS/s")
    print(f"  Percent of peak achieved:    ", round(row['PercentAchieved'], 4), "%")

    # Calculate achieved
    print(f"\n  Achieved throughput:         ", round(row['Throughput'], 2), "GFLOPS/s")

    # Calulate peak
    if args.hw_metrics != None:
        totalAdd = row['SQ_INSTS_VALU_ADD_F16'] + row['SQ_INSTS_VALU_ADD_F32'] + row['SQ_INSTS_VALU_ADD_F64']
        totalMul = row['SQ_INSTS_VALU_MUL_F16'] + row['SQ_INSTS_VALU_MUL_F32'] + row['SQ_INSTS_VALU_MUL_F64']
        totalTrans = row['SQ_INSTS_VALU_TRANS_F16'] + row['SQ_INSTS_VALU_TRANS_F32'] + row['SQ_INSTS_VALU_TRANS_F64']
        totalFMA = row['SQ_INSTS_VALU_FMA_F16'] + row['SQ_INSTS_VALU_FMA_F32'] + row['SQ_INSTS_VALU_FMA_F64']
        totalOps = totalAdd + totalMul + totalTrans + totalFMA
        flopFactor = ((128 * (totalAdd + totalMul + totalTrans)) + (256 * totalFMA)) / totalOps

        if row[gfx2 + ' mean']/1000*numCUs*flopFactor < peakBandwidth*row['AI_HBM_TOT']:  # Compute bound
            print("\n  This kernel is compute bound.")
            peak = row[gfx2 + ' mean']/1000*numCUs*flopFactor
            print(f"  Peak achievable (using mean graphics counter):", round(peak, 2), "GFLOPS/s")
            percentAchieved = achieved / peak * 100
            print(f"  Percent achieved (using mean graphics counter):", round(percentAchieved, 3), "%")
            print(f"  Peak achievable (using max graphics counter): ", round(row[gfx2 + ' max']/1000*numCUs*flopFactor, 2), "GFLOPS/s")
            print(f"  Percent achieved (using max graphics counter):", round(100 * achieved / (row[(gfx2 + " max")]/1000*numCUs*flopFactor), 3), "%")
        else:  # Memory bound
            print("\n  This kernel is memory bound.")
            peak = peakBandwidth*row['AI_HBM_TOT']
            print(f"  Peak achievable:    ", round(peak, 2), "GFLOPS/s")
            percentAchieved = achieved / peak * 100
            print(f"  Percent achieved:   ", round(percentAchieved, 3), "%")

    # Troubleshooting bad kernels
    if args.hw_metrics != None:
        #if percentAchieved < 90:  # 90% threshold
        # Check if graphics clock is being throttled
        print(f"\n  Graphics counter peak:", round(row[gfx2 + ' max'], 2), "MHz")
        print(f"  Graphics counter mean:", round(row[gfx2 + ' mean'], 2), "MHz")
        if (row[gfx2 + " max"] < 2000):
            print("  Graphics counter is not reaching its peak (~2,000 MHz)")

        # Check power first
        print(f"\n  Power max: ", round(row[socketPower + " max"], 2), "W")
        print(f"  Power mean:", round(row[socketPower + " mean"], 2), "W")
        if row[socketPower + " max"] > 549:
            print("  The graphics counter is likely being throttled by power.")
        else:
            print("  Not throttled by power")

        # Check socket temp
        print(f"\n  Socket temp mean:", round(row[maxSocketTemp + " mean"],2), "C")
        print(f"  Socket temp max: ", round(row[maxSocketTemp + " max"],2), "C")

        # Check HBM temp
        print(f"\n  HBM temp mean:", round(row[maxHBMTemp + " mean"], 2), "C")
        print(f"  HBM temp max: ", round(row[maxHBMTemp + " max"], 2), "C")


print(f"\n{insigKernels} kernels omitted for having less than {sigRuntime} percent runtime (use --sig-runtime to change threshold)")
print()

df_roof['Percentage'] = df_roof['DurationNs']/sum(totalRuntimes) * 100
df_roof['Throughput'] = df_roof['TOTAL_OPS'] / df_roof['DurationNs']
df_roof['PercentAchieved'] = df_roof['Throughput'] / df_roof['PEAK'] * 100
df_roof = df_roof.rename(columns={'KernelName_x':'KernelName'}).merge(df['Percentage'],on='KernelName')
df_roof = df_roof.rename(columns={'Percentage_x':'Percentage'})
df_roof = df_roof.rename(columns={'Percentage_y':'PercentageAggregate'})

# Interactive roofline plot
if args.plot:
    df_plot = df_roof
    df_plot['TotalKernels'] = len(df_plot)  # Saving in this format to pass to tooltip

    # Maximum number of kernel dispatches to plot (browser slows down with more)
    n_samples = 50000

    if len(df_plot) > n_samples:
        df_plot.sort_values('Index')
        df_plot = df_plot.iloc[::len(df_plot) // n_samples]
        n_samples = len(df_plot)  # Adjust for remainder

    # Sort the data by significance
    df_plot = df_plot.sort_values(by='PercentageAggregate', ascending=False)

    # Assign a unique color to each name
    color_map = px.colors.qualitative.Plotly
    name_to_color = {name: color_map[i % len(color_map)] for i, name in enumerate(df.index)}
    df_plot['color'] = [name_to_color[name] for name in df_plot['KernelName']]
    df['color'] = [name_to_color[name] for name in df.index]

    # Compute the range for the plot
    x_min = 0.001
    x_max = 100000
    y_min = 0.5
    y_max = 200000

    # Get x-values for lines
    x_vals = np.logspace(np.log10(x_min), np.log10(x_max), 200)
    # Add a point for each spot a bandwidth line intersects the compute line
    x_vals = np.sort(np.append(x_vals, [compute_peaks[arch] / caches[arch][cache] for cache in caches[arch] for arch in caches]))

    # MI250 Roofline
    mi250x_lines = [
        go.Scatter(
            x=x_vals,
            y=np.minimum(caches['MI250'][key] * x_vals, compute_peaks['MI250']),
            visible=True,
            mode='lines',
            name=f'MI250X {key} Achievable Peak',
            line=dict(color=color_map[0], dash='solid')
        )
        for key in caches['MI250'].keys()
    ]
    # MI300a Roofline
    mi300a_lines = [go.Scatter(
            x=x_vals,
            y=np.minimum(caches['MI300A'][key] * x_vals, compute_peaks['MI300A']),
            visible=True,
            mode='lines',
            name=f'MI300A {key} Achievable Peak',
            line=dict(color=color_map[0], dash='solid')
        )
        for key in caches['MI300A'].keys()
    ]
    # MI300x Roofline
    mi300x_lines = [go.Scatter(
            x=x_vals,
            y=np.minimum(caches['MI300x'][key] * x_vals, compute_peaks['MI300x']),
            visible=True,
            mode='lines',
            name=f'MI300x {key} Achievable Peak',
            line=dict(color=color_map[1], dash='solid')
        )
        for key in caches['MI300x'].keys()
    ]

    # Truncate long kernel names (looking at you, rocblas)
    df['short_name'] = np.where(
        df.index.str.len() > 50,
        df.index.str.slice(0, 47) + '…',  # 47 + 1 char ellipsis = 48 visible chars
        df.index
    )
    # Truncate long kernel names (looking at you, rocblas)
    df_plot['short_name'] = np.where(
        df_plot['KernelName'].str.len() > 50,
        df_plot['KernelName'].str.slice(0, 47) + '…',  # 47 + 1 char ellipsis = 48 visible chars
        df_plot['KernelName']
    )

    # Layout with log-log axes and slider
    layout = go.Layout(
        title=f'Roofline Plot for kernels in {roofCountFilename}',
        xaxis=dict(
            type='log',
            title=f'Arithmetic Intensity (Flops per Byte Accessed)',
            range=[np.log10(x_min), np.log10(x_max)],
            autorange=False
        ),
        yaxis=dict(
            type='log',
            title='Throughput (GFLOPs/s)',
            range=[np.log10(y_min), np.log10(y_max)],
            autorange=False
        )
    )

    def interactive_plot(cache):

        # Add scatter plot
        scatter_items = []
        for name in df_plot['KernelName'].unique():
            kernels = df_plot[df_plot['KernelName'] == name]
            short_name = kernels.iloc[0]['short_name']
            short_name += f'\t{round(kernels.iloc[0]["PercentageAggregate"], 3)}% runtime'
            # Pass name, percent for tooltip
            customdata=np.stack([kernels['short_name'], kernels['Percentage'], kernels['Index'], kernels['TotalKernels'], kernels['PercentageAggregate'], kernels['PEAK'], kernels['LIMITER']], axis=-1)
            scatter_items.append(go.Scatter(
                x=kernels[f'AI_{cache}_TOT'],
                y=kernels['Throughput'],
                mode='markers',
                name=short_name,
                marker=dict(color=kernels['color']),
                visible=False,
                customdata = customdata,
                hovertemplate=
                    'Name: %{customdata[0]}<br>' +
                    'Index: %{customdata[2]} / %{customdata[3]}<br>' +
                    'AI: %{x}<br>' +
                    'Achieved throughput: %{y:.3f} GFLOPs/s<br>' +
                    'Peak throughput: %{customdata[5]:.3f} GFLOPs/s<br>' +
                    'Performance limiter: %{customdata[6]}<br>' +
                    'Aggregate percent runtime: %{customdata[4]:.5f} %<br>' +
                    'Individual percent runtime: %{customdata[1]:.5f} %<extra></extra>'
            ))

        return scatter_items

    def interactive_plot_agg(cache):

        # Add scatter plot
        scatter_items = []
        for index, kernel in df.iterrows():
            short_name = kernel['short_name']
            short_name += f'\t{round(kernel["Percentage"], 3)}% runtime'
            # Pass name, percent for tooltip
            customdata=np.stack([[kernel['short_name']], [kernel['Percentage']], [kernel['PEAK']], [kernel['LIMITER']], [kernel['Count']]], axis=-1)
            scatter_items.append(go.Scatter(
                x=[kernel[f'AI_{cache}_TOT']],
                y=[kernel['Throughput']],
                mode='markers',
                name=short_name,
                marker=dict(color=kernel['color']),
                visible=False,
                customdata = customdata,
                hovertemplate=
                    'Name: %{customdata[0]}<br>' +
                    'AI: %{x}<br>' +
                    'Achieved throughput: %{y:.3f} GFLOPs/s<br>' +
                    'Peak throughput: %{customdata[2]:.3f} GFLOPs/s<br>' +
                    'Performance limiter: %{customdata[3]}<br>' +
                    'Total dispatches: %{customdata[4]}<br>' +
                    'Aggregate percent runtime: %{customdata[1]:.5f} %<br>' +
                    '<extra></extra>'
            ))

        return scatter_items

    scatter_hbm = interactive_plot('HBM')
    scatter_l2 = interactive_plot('L2')
    scatter_l1 = interactive_plot('vL1d')
    scatter_lds = interactive_plot('LDS')
    scatter_hbm_agg = interactive_plot_agg('HBM')
    scatter_l2_agg = interactive_plot_agg('L2')
    scatter_l1_agg = interactive_plot_agg('vL1d')
    scatter_lds_agg = interactive_plot_agg('LDS')


    # Check for MI200/300
    if df_plot.keys().str.contains('TCC_BUBBLE').sum() > 0:
        rooflines = mi300a_lines + mi300x_lines
    else:
        rooflines = mi250x_lines

    # Set HBM aggregate as default visibility
    for scatter in scatter_hbm_agg:
        scatter.visible = True
    for roofline in rooflines:
        roofline.line.width = 1
    for roofline in rooflines[::4]:
        roofline.line.width = 3

    # Create slider steps based on percentage runtime thresholds
    thresholds = df['Percentage'].tolist()
    # Set max number of thresholds at 40, also include 0
    thresholds.sort()
    thresholds = [0] + thresholds[-40:]

    # Create separate slider for each cache level
    sliders = []
    # Individual kernel dispatches
    for c in range(4):
        steps = []
        for threshold in thresholds:
            # Make the rooflines visible
            visible = [True] * 4
            visible = visible * int(len(rooflines)/4)

            # Filter the scatter plots for the correct cache
            visible = visible + [False] * (c * len(df_plot.groupby('KernelName')))
            visible = visible + (pd.Series(df_plot.groupby('KernelName')['PercentageAggregate'].first().tolist()).sort_values() >= threshold).tolist()[::-1]
            visible = visible + [False] * ((3 - c) * len(df_plot.groupby('KernelName')))
            # Aggregates
            visible = visible + [False] * (4 * len(df))
            steps.append(dict(
                method="update",
                args=[{"visible": visible}],
                label=f"{threshold:.3f}%"
            ))
        sliders.append(dict(
            active=0,
            currentvalue={"prefix": "Minimum Percent Runtime: "},
            pad={"t": 50},
            steps=steps
        ))
    # Aggregate kernel dispatches
    for c in range(4):
        steps = []
        for threshold in thresholds:
            # Make the rooflines visible
            visible = [True] * 4
            visible = visible * int(len(rooflines)/4)

            # Filter the scatter plots for the correct cache
            visible = visible + [False] * (4 * len(df_plot.groupby('KernelName')))
            # Aggregates
            visible = visible + [False] * (c * len(df))
            visible = visible + (pd.Series(df['Percentage'].tolist()).sort_values() >= threshold).tolist()[::-1]
            visible = visible + [False] * ((3 - c) * len(df))
            steps.append(dict(
                method="update",
                args=[{"visible": visible}],
                label=f"{threshold:.3f}%"
            ))
        sliders.append(dict(
            active=0,
            currentvalue={"prefix": "Minimum Percent Runtime: "},
            pad={"t": 50},
            steps=steps
        ))

    # Add disclaimer if kernel dispatches need to be filtered
    disclaimer = None
    if df_plot.iloc[0]['TotalKernels'] > n_samples:
        disclaimer = dict(
            text=f"Depicting {n_samples} kernel dispatches out of {df_plot.iloc[0]['TotalKernels']}",
            xref="paper", yref="paper",
            x=1, y=0.05,
            showarrow=False,
            font=dict(size=12),
            xanchor='left',
            yanchor='top'
        )

    # Combine scatters and rooflines
    fig = go.Figure(data=rooflines + scatter_hbm + scatter_l2 + scatter_l1 + scatter_lds + scatter_hbm_agg + scatter_l2_agg + scatter_l1_agg + scatter_lds_agg, layout=layout)
    fig.update_layout(
        sliders=[sliders[4]],
        plot_bgcolor="white",
        xaxis=dict(
            gridcolor="lightgray",
            zerolinecolor="lightgray",
            linecolor="black",
        ),
        yaxis=dict(
            gridcolor="lightgray",
            zerolinecolor="lightgray",
            linecolor="black",
        ),
        updatemenus=[
            # Add dark mode toggle
            dict(
                type="buttons",
                direction="right",
                buttons=list([
                    dict(label="Light Mode",
                        method="relayout",
                        args=[{
                            "plot_bgcolor": "white",
                            "paper_bgcolor": "white",
                            "font.color": "black",
                            "xaxis.gridcolor": "lightgray",
                            "xaxis.zerolinecolor": "lightgray",
                            "xaxis.linecolor": "black",
                            "yaxis.gridcolor": "lightgray",
                            "yaxis.zerolinecolor": "lightgray",
                            "yaxis.linecolor": "black"
                        }]),
                    dict(label="Dark Mode",
                        method="relayout",
                        args=[{
                            "plot_bgcolor": "black",
                            "paper_bgcolor": "black",
                            "font.color": "white",
                            "xaxis.gridcolor": "gray",
                            "xaxis.zerolinecolor": "gray",
                            "xaxis.linecolor": "white",
                            "yaxis.gridcolor": "gray",
                            "yaxis.zerolinecolor": "gray",
                            "yaxis.linecolor": "white"
                        }])
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1,
                xanchor="right",
                y=1.1,
                yanchor="top"
            ),
            # Add memory hierarchy toggle
            dict(
                type="buttons",
                direction="down",
                buttons=list([
                    dict(label="HBM Agg",
                        method="update",
                        args=[
                            {
                                "line": [
                                    {"width": 3 if j == 0 else 1, "color": color_map[i]}
                                    for i in range(int(len(rooflines) / 4))
                                    for j in range(4)
                                ],
                                "visible": (
                                    [True] * len(rooflines) +
                                    [False] * len(scatter_hbm) * 4 +
                                    [True] * len(scatter_hbm_agg) +
                                    [False] * len(scatter_hbm_agg) * 3
                                ),
                            }, {
                                "sliders": [sliders[4]],
                                "annotations": [None]
                            }
                        ]),
                    dict(label="HBM",
                        method="update",
                        args=[
                            {
                                "line": [
                                    {"width": 3 if j == 0 else 1, "color": color_map[i]}
                                    for i in range(int(len(rooflines) / 4))
                                    for j in range(4)
                                ],
                                "visible": (
                                    [True] * len(rooflines) +
                                    [True] * len(scatter_hbm) +
                                    [False] * len(scatter_hbm) * 3 +
                                    [False] * len(scatter_hbm_agg) * 4
                                ),
                            }, {
                                "sliders": [sliders[0]],
                                "annotations": [disclaimer]
                            }
                        ]),
                    dict(label="L2 Agg",
                        method="update",
                        args=[{
                                "line": [
                                    {"width": 3 if j == 1 else 1, "color": color_map[i]}
                                    for i in range(int(len(rooflines) / 4))
                                    for j in range(4)
                                ],
                                "visible": (
                                    [True] * len(rooflines) +
                                    [False] * len(scatter_l2) * 4 +
                                    [False] * len(scatter_l2_agg) +
                                    [True] * len(scatter_l2_agg) +
                                    [False] * len(scatter_l2_agg) * 2
                                ),
                            }, {
                                "sliders": [sliders[5]],
                                "annotations": [None]
                            }
                        ]),
                    dict(label="L2",
                        method="update",
                        args=[{
                                "line": [
                                    {"width": 3 if j == 1 else 1, "color": color_map[i]}
                                    for i in range(int(len(rooflines) / 4))
                                    for j in range(4)
                                ],
                                "visible": (
                                    [True] * len(rooflines) +
                                    [False] * len(scatter_l2) +
                                    [True] * len(scatter_l2) +
                                    [False] * len(scatter_l2) * 2 +
                                    [False] * len(scatter_l2_agg) * 4
                                ),
                            }, {
                                "sliders": [sliders[1]],
                                "annotations": [disclaimer]
                            }
                        ]),
                    dict(label="vL1d Agg",
                        method="update",
                        args=[{
                                "line": [
                                    {"width": 3 if j == 2 else 1, "color": color_map[i]}
                                    for i in range(int(len(rooflines) / 4))
                                    for j in range(4)
                                ],
                                "visible": (
                                    [True] * len(rooflines) +
                                    [False] * len(scatter_l1) * 4 +
                                    [False] * len(scatter_l1_agg) * 2 +
                                    [True] * len(scatter_l1_agg) +
                                    [False] * len(scatter_l1_agg)
                                ),
                            }, {
                                "sliders": [sliders[6]],
                                "annotations": [None]
                            }
                        ]),
                    dict(label="vL1d",
                        method="update",
                        args=[{
                                "line": [
                                    {"width": 3 if j == 2 else 1, "color": color_map[i]}
                                    for i in range(int(len(rooflines) / 4))
                                    for j in range(4)
                                ],
                                "visible": (
                                    [True] * len(rooflines) +
                                    [False] * len(scatter_l1) * 2 +
                                    [True] * len(scatter_l1) +
                                    [False] * len(scatter_l1) +
                                    [False] * len(scatter_l1_agg) * 4
                                ),
                            }, {
                                "sliders": [sliders[2]],
                                "annotations": [disclaimer]
                            }
                        ]),
                    dict(label="LDS Agg",
                        method="update",
                        args=[{
                                "line": [
                                    {"width": 3 if j == 3 else 1, "color": color_map[i]}
                                    for i in range(int(len(rooflines) / 4))
                                    for j in range(4)
                                ],
                                "visible": (
                                    [True] * len(rooflines) +
                                    [False] * len(scatter_lds) * 4 +
                                    [False] * len(scatter_lds_agg) * 3 +
                                    [True] * len(scatter_lds_agg)
                                ),
                            }, {
                                "sliders": [sliders[7]],
                                "annotations": [None]
                            }
                        ]),
                    dict(label="LDS",
                        method="update",
                        args=[{
                                "line": [
                                    {"width": 3 if j == 3 else 1, "color": color_map[i]}
                                    for i in range(int(len(rooflines) / 4))
                                    for j in range(4)
                                ],
                                "visible": (
                                    [True] * len(rooflines) +
                                    [False] * len(scatter_lds) * 3 +
                                    [True] * len(scatter_lds) +
                                    [False] * len(scatter_lds_agg) * 4
                                ),
                            }, {
                                "sliders": [sliders[3]],
                                "annotations": [disclaimer]
                            }
                        ]),
                ]),
                showactive=True,
            ),

        ]
    )

    fig.write_html(f'{roofCountFilename}.html')
    print(f"Roofline plot saved to                  {roofCountFilename}.html")

    print(f"Arithmetic intensity bar plot saved to  {roofCountFilename}_plot.png")
    if args.hw_metrics != None:
        print(f"Clocks and telemetry data plot saved to {hwFilename}_plot.png")

# Output Results to CSV
if args.dump:
    df.to_csv(roofCountFilename + '_EXTRACTED_AGG.csv')
    df_roof.to_csv(roofCountFilename + '_EXTRACTED.csv')
    print(f"Full dataframe dumped to                {roofCountFilename}_EXTRACTED.csv")
    print(f"Aggregate kernels dataframe dumped to   {roofCountFilename}_EXTRACTED_AGG.csv")

