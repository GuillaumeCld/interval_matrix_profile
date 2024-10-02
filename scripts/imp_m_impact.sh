# Example: ash scripts/imp_m_impact.sh 1 45 imp_m_impact

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


echo "m,Time" > $bf_file
echo "m,Time" > $stomp_file
echo "m,Time" > $block_file

# Loop to run the experiment 20 times with different thread numbers
for m in $(seq $min_size 5 $max_size)
do
    echo "Running experiment $m"
    taskset -c 1 ./build/experiments/imp_time 1 $m > res_tmp
    cat res_tmp
    TIME=$(grep "BF" res_tmp | awk '{print $2}')
    echo "$m,${TIME}" >> $bf_file
    TIME=$(grep "STOMP" res_tmp | awk '{print $2}')
    echo "$m,${TIME}" >> $stomp_file
    TIME=$(grep "BLOCK" res_tmp | awk '{print $2}')
    echo "$m,${TIME}" >> $block_file
    rm res_tmp
done

# Generate the plots
gnuplot -c scripts/gnuplots/imp_m_impact.gp $bf_file $stomp_file $block_file 
