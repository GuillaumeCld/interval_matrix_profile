# Example: bash scripts/imp_interval_impact.sh 50 500 imp_interval_impact

# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <min_size> <max_size> <output_file>"
    exit 1
fi
export OMP_NUM_THREADS=1
# Assign input parameters to variables
min_size=$1
max_size=$2
output_file=$3

# Create the output files
bf_file="./scripts/output_files/${output_file}_bf.txt"
aamp_file="./scripts/output_files/${output_file}_aamp.txt"
block_file="./scripts/output_files/${output_file}_block.txt"

echo "L,Time" > $bf_file
echo "L,Time" > $aamp_file
echo "L,Time" > $block_file

# Loop to run the experiment 20 times with different thread numbers
for size in $(seq $min_size 50 $max_size)
do
    echo "Running experiment $size"
    taskset -c 1 ./build/experiments/imp_time 3 $size > res_tmp
    cat res_tmp
    TIME=$(grep "BF" res_tmp | awk '{print $2}')
    echo "$size,${TIME}" >> $bf_file
    TIME=$(grep "AAMP" res_tmp | awk '{print $2}')
    echo "$size,${TIME}" >> $aamp_file
    TIME=$(grep "BLOCK" res_tmp | awk '{print $2}')
    echo "$size,${TIME}" >> $block_file
    rm res_tmp
done

# Generate the plots
gnuplot -c scripts/gnuplots/imp_interval_impact.gp $bf_file $aamp_file $block_file 
