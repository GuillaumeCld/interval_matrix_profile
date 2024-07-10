#Example: bash scripts/k_nn_exclusion_var.sh 1048576 15 k_nn_exclusion_var

# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <vector_size> <k> <output_file>"
    exit 1
fi

# Assign input parameters to variables
vector_size=$1
k=$2
output_file=$3

# Create the output files
sweep_file="./scripts/output_files/${output_file}_sweep.txt"
heap_file="./scripts/output_files/${output_file}_heap.txt"
echo "exclusion_size,Time" > $sweep_file
echo "exclusion_size,Time" > $heap_file

# Loop to run the experiment 20 times with different thread numbers
for exclusion_size in $(seq 32 64 1024)
do
    echo "Running experiment $exclusion_size"
    taskset -c 1 ./build/experiments/k_nn $vector_size $k $exclusion_size > res_tmp
    cat res_tmp
    TIME=$(grep "Sweep" res_tmp | awk '{print $2}')
    echo "$exclusion_size,${TIME}" >> $sweep_file
    TIME=$(grep "Heap" res_tmp | awk '{print $2}')
    echo "$exclusion_size,${TIME}" >> $heap_file
    rm res_tmp
done

# Generate the plots
gnuplot -c scripts/gnuplots/knn_exclusion_var.gp $sweep_file $heap_file