# Usage: bash scripts/imp_k_impact.sh imp_k_impact

# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0  <output_file>"
    exit 1
fi
export OMP_NUM_THREADS=1
# Create the output files
block_file="./scripts/output_files/${$3}_block.txt"

echo "k,Time" > $block_file

# Loop to run the experiment 20 times with different thread numbers
for k in 1 5 10 20
do
    echo "Running experiment $k"
    taskset -c 1 ./build/experiments/imp_time 5 $k > res_tmp
    cat res_tmp
    TIME=$(grep "BLOCK" res_tmp | awk '{print $2}')
    echo "$k,${TIME}" >> $block_file
    rm res_tmp
done

