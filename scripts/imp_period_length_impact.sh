# Example: bash scripts/imp_period_length_impact.sh 100 1000 imp_period_length_impact

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
stomp_file="./scripts/output_files/${output_file}_stomp.txt"
block_file="./scripts/output_files/${output_file}_block.txt"

echo "L,Time" > $bf_file
echo "L,Time" > $stomp_file
echo "L,Time" > $block_file

# Loop to run the experiment 20 times with different thread numbers
for size in $(seq $min_size 100 $max_size)
do
    echo "Running experiment $size"
    taskset -c 1 ./build/experiments/imp_time 6 $size > res_tmp
    cat res_tmp
    TIME=$(grep "BF" res_tmp | awk '{print $2}')
    echo "$size,${TIME}" >> $bf_file
    TIME=$(grep "AAMP" res_tmp | awk '{print $2}')
    echo "$size,${TIME}" >> $stomp_file
    TIME=$(grep "BLOCK" res_tmp | awk '{print $2}')
    echo "$size,${TIME}" >> $block_file
    rm res_tmp
done

# Generate the plots
gnuplot -c scripts/gnuplots/imp_period_length_impact.gp $bf_file $stomp_file $block_file 
