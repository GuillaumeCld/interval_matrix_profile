# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <vector_size> <window_size> <output_file>"
    exit 1
fi

# Assign input parameters to variables
vector_size=$1
window_size=$2
output_file=$3

# Loop to run the experiment 20 times with different thread numbers
for i in $(seq 1 20)
do
    echo "Running experiment $i"
    export OMP_NUM_THREADS=$i
    ./build/experiments/experiments $vector_size $window_size > res_tmp
    cat res_tmp
    TIME=$(grep "BLOCK" res_tmp | awk '{print $4}')
    echo "$i,${TIME}" >> $output_file
    rm res_tmp
done