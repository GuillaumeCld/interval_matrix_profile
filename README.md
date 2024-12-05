
# Interval Matrix Profile

This repository contains the implementation of the Interval Matrix Profile algorithm, and Seasonal Matrix Profile. The project includes various components such as tests, experiments, and Python bindings using Pybind11. The repository also provides case studies and experiments to demonstrate the application of the algorithm.

## Build the project

The project have been tested on gcc 13.1.0 / cmake 3.19.0 and  gcc 10.3.0 / cmake 3.21.3.


### Build the python dependencies (Required for the notebooks)

Creating the python environment with conda
```sh
 conda env create -f environment.yml
 conda activate imp
```
Add the environment to jupyter kernels
```sh
python -m ipykernel install --user --name imp --display-name "Python (imp)"
```


### Build the sources

1. Create a build directory:
```sh
mkdir build
cd build
```
2.  Setup the CMake
    With ccmake or directly in the CMakeLists.txt, you can set which components to compile:
    - tests
    - experiments
    - Python binding with [Pybind11](https://github.com/pybind/pybind11)

If building the python binding be sure to be in the python environment.

3. Run CMake to configure the project:
```sh
cmake ..
```

4. Build the project:
```sh
make
```

## Run the experiments

The experiments are in the /experiments/ folder. 
The scripts to run the experiments are in /scripts/, the /all.sh/ script run the scripts with the paper configuration.
```sh
bash scripts/all.sh
```


The numerical precision experiment can be run as follows,
```sh
cd build
./experiments/numerical_error
```

## Case studies with the notebooks 

The Climate Pulse data is available in /Data/climate_pulse/ or directly downloadable from [Climate Pulse](https://pulse.climate.copernicus.eu/).

The Mediterranean sea surface temperature data is available in /Data/SST/.

The daily and hourly data from ERA5_land are availlable in /Data/ERA5_land/.

The case studies of the paper can be found in the folder /notebooks//.

- [Climate Pulse SST](notebooks/climate_pulse_SST_2024.ipynb)
- [Climate Pulse T2M (TAS)](notebooks/climate_pulse_T2M_2023.ipynb)
- [Mediterranean SST](notebooks/medi_sst_pybind.ipynb)
