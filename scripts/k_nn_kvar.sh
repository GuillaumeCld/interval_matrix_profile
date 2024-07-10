# Example: ash scripts/k_nn_kvar.sh 1048576 128 k_nn_kvar

# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <vector_size> <exclusion_size> <output_file>"
    exit 1
fi

# Assign input parameters to variables
vector_size=$1
exclusion_size=$2
output_file=$3

# Create the output files
sweep_file="./scripts/output_files/${output_file}_sweep.txt"
heap_file="./scripts/output_files/${output_file}_heap.txt"

echo "k,Time" > $sweep_file
echo "k,Time" > $heap_file

# Loop to run the experiment 20 times with different thread numbers
for k in $(seq 4 4 64)
do
    echo "Running experiment $k"
    taskset -c 1 ./build/experiments/k_nn $vector_size $k $exclusion_size > res_tmp
    cat res_tmp
    TIME=$(grep "Sweep" res_tmp | awk '{print $2}')
    echo "$k,${TIME}" >> $sweep_file
    TIME=$(grep "Heap" res_tmp | awk '{print $2}')
    echo "$k,${TIME}" >> $heap_file
    rm res_tmp
done

# Generate the plots
gnuplot -c scripts/gnuplots/knn_var.gp $sweep_file $heap_file
