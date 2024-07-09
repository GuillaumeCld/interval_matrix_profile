# Go to the root of the repository

echo $(pwd)
topo --output-format png -v --no-io > cpu.png
cd Devel/MP_CPP/
# Build the project
if ![ -d build ]; then
    mkdir build
fi
echo "Building the project..."
cd build
cmake ..
make -j 16

# Run the project
cd ..
echo $(pwd)
echo "Running the experiments..."
# bash scripts/speed_experiments.sh 524288 64 para_stomp.txt
bash scripts/speedup.sh para_stomp.txt
gnuplot -e "ARG1='para_stomp_speedup.txt'; ARG2='para_stomp.png'" scripts/speedup.gp
echo "Experiments completed!"
