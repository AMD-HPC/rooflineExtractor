# Roofline Extractor
Roofline Extractor is a tool that calculates the percent of the empirical peak performance that an application is achieving on a per-kernel basis. The script also can pull in telemetry data (power, temperature, clocks) from [rocSTAR](https://github.com/AMD-HPC/rocSTAR) to help explain any shortcomings in performance.

[Example Roofline Plot for rocHPL](http://canofcorn.amd.com/Rooflines/hpl-mi300x-roof-counters.html)

## Data Generation
The following two runs of rocprof are needed to use rooflineExtractor:
* Counters
  * This run gathers counters for the application. Pick the input file from this directory that is appropriate for your architecture.
  * `rocprof -i roof-counters-<arch>.txt <app exe>` or `rocprofv3 -i roof-counters-<arch>.txt -- <app exe>`
    * If using rocprofv3, another command is needed to consolidate its output into a single file: `python3 convert-conters-collection-format.py -i <path to rocprofv3 output files> -o <singular output file>`
* Runtime stats
  * This run gathers timing information for the application
  * `rocprof --stats <exe>` or `rocprofv3 --kernel-trace -- <exe>`

## Install Python Packages
Run this to install all necessary packages:
```
pip install -r requirements.txt
```

## Run
Simplest run:

`python3 rooflineExtractor.py -c [roof-counter.csv filename] -r [results.csv filename]`

Additional optional flags:
* `--plot`: Generate plots
* `--dump`: Dump pandas dataframe to `logs/<app>_roof_counters_EXTRACTED.csv`
* `--sig-runtime [% runtime]`: Specify what's the minimum runtime for a kernel to be considered "significant" and be included in analysis. Defaults to 10%.
* `-m [hw_metrics filename]`: Include rocSTAR telemetry data, which is used for calculating peak throughputs and plotting

## Optional: RocSTAR Setup
This step is optional. To generate the telemetry data, the application must be instrumented with rocSTAR. Examples of instrumented kernels can be found [here](https://github.com/AMD-HPC/rocSTAR/tree/main/examples).
* Clone the repository: `git clone git@github.com:AMD-HPC/rocSTAR.git`
* Run build.sh to generate the rocSTAR/esmi library: `./build.sh`
* Find the source code for the application. Add rocStarInit() and rocStarFinalize() to the beginning and end of the code. Then add rocStarStart("kernel\_name") and rocStarStop() around around each call for the kernels that you want to profile. 
* Make sure the label provided in rocStarStart **exactly** matches the name of the kernel being profiled, so the Python script can correctly identify the kernel.
* Compile and run the app. When compiling, be sure the include path includes rocSTAR/build/esmi\_ib\_library/include and the linker path includes rocSTAR/build/esmi\_ib\_library/build

## Example:
Here is an example using nbody-nvidia-mini.
```
# Collect kernel counters
rocprof -i roof-counters-gfx942.txt ./nbody-orig 1048576
# Collect runtime stats
rocprof --stats ./nbody-orig 1048576
# Optional: Set up rocSTAR (edit nbody-orig.cu)
#           Run the app, instrumented with rocSTAR, to generate telemetry data
./nbody-orig 1048576

# Run roofline extractor to generate plots and dataframes
python3 rooflineExtractor.py -c roof-counters-gfx942.csv -r results.csv --plot --dump
```
## Output:
* A guided analysis via the terminal to check which kernels aren't achieving a performance threshold. If a kernel is below the threshold, the script will check if the graphics clock is being throttled by power or temperature.
* An HTML file with an interactive roofline plot with the performance and arithmetic intensity of each kernel instance
* A CSV file with all of the per-kernel throughput and arithmetic intensity information calculated
* A `hw_metrics_plot.png` file that shows the power consumption, socket temperature, graphics clock, and memory clock over time. The graph also shows the flops achieved vs. empirical peak for each instance of each kernel tracked, on the same timeline.
* A `<app>_roof_counters_plot.png` file that plots Arithmetic Intensities (The one true AI) for each of the different parts of the memory hierarchy for the the different compute pipes using the different datatypes.
  * x-axis is a list of different arithmetic intensities of the format `AI_memRegion_computeRegion_datatype` where
    * `memRegion` is the location in the memory hierarchy for which we are counting bytes moved
    * `computeRegion` is the compute pipe for which we are counting OPS performed
    * `datatype` is the datatype being performed in the `computeRegion`
  * y-axis is log scale showing the achieved arithmetic intensity (ops/byte)


![nbody_plot](https://github.com/AMD-HPC/rocShore/assets/170367005/aa3589a3-3f1b-4f02-b2c3-c191a4b0d4ee)
