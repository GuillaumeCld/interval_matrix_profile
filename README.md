# Interval Matrix Profile




## Building the C++ project

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


### Running the experiments


### Case studies with the notebooks 