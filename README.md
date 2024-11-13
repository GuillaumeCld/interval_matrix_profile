
# Interval Matrix Profile

This repository contains the implementation of the Interval Matrix Profile algorithm, and Seasonal Matrix Profile. The project includes various components such as tests, experiments, and Python bindings using Pybind11. The repository also provides case studies and experiments to demonstrate the application of the algorithm.

## Building the project

The project have been tested on gcc 10.3.0 / cmake 3.21.3 and gcc 13.1.0 / cmake 3.19.0


### Building the python depanencies (Required for notebooks)

Creating the python environment with conda
```sh
 conda env create -f environment.yml
```
Add the environment to jupyter kernels
```sh
python -m ipykernel install --user --name imp --display-name "Python (imp)"
```


### Build with CMake

1. Create a build directory:
```sh
mkdir build
cd build
```
2.  Setup the CMake
    With ccmake or directly in the CMakeLists.txt, you can steup which compentent to compile:
    - tests
    - experiments
    - Python binding with Pybind11

3. Run CMake to configure the project:
```sh
cmake ..
```

4. Build the project:
```sh
make
```

## Running the experiments

The experiments are in the ~experiments/~ folder. 
The scripts to run the  experiments are in ~scripts/~, the ~all.sh~ run the scripts with the paper configuration.
```sh
bash scripts/all.sh
```


The numerical precision experiment can be run as follows,
```sh
cd build
./experiments/numerical_error
```



## Case studies with the notebooks 

The Climate Pulse is available in ~Data/climate_pulse~ or directly downloable from [Climate Pulse](https://pulse.climate.copernicus.eu/).

The Mediterannean sea surface temperature data is available in ~Data/SST~.

The daily and hourly data from ERA5_land are availlables in ~Data/ERA5_land~.

The case studies of the paper can be found in the folder ~notebooks/~.

- [Climate Pulse SST](notebooks/climate_pulse_SST_2024.ipynb)
- [Climate Pulse T2M (TAS)](notebooks/climate_pulse_T2M_2023.ipynb)
- [Mediterranean SST](medi_sst_pybind.ipynb)