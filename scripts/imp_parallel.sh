# Usage: bash scripts/imp_parallel.sh 1 20 imp_parallel
# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <min_threads> <max_threads> <output_file>"
    exit 1
fi

# Assign input parameters to variables
min_threads=$1
max_threads=$2
output_file=$3

#> $output_file
output_file="./scripts/output_files/${output_file}.txt"

TIME1=0

export OMP_PROC_BIND=close
# Loop to run the experiment 20 times with different thread numbers
for i in $(seq $min_threads 2 $max_threads)
do
    echo "Running experiment $i"
    export OMP_NUM_THREADS=$i
    echo $OMP_NUM_THREADS
    ./build/experiments/imp_time 4 $i > res_tmp
    cat res_tmp
    TIME=$(grep "BLOCK" res_tmp | awk '{print $2}')

    if [ $i -eq $min_threads ]; then
        SPEEDUP=1
        TIME1=$TIME
    else
        SPEEDUP=$(echo "scale=2; $TIME1/$TIME" | bc)
    fi
    echo "$i,${TIME},${SPEEDUP}" >> $output_file
    rm res_tmp
done

gnuplot -c scripts/gnuplots/imp_parallel.gp $output_file 
