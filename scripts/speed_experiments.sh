#!bin/bash
vector_size=100000
window_size=64

for i in $(seq 1 20)
do
    echo "Running experiment $i"
    export OMP_NUM_THREADS=$i
    ../build/experiments/experiments $vector_size $window_size > res_tmp
    cat res_tmp
    TIME=`grep "Brute Force" res_tmp | awk '{print $5}'`
    echo "$i,${TIME}" >> bf_100k.txt
    rm res_tmp
done