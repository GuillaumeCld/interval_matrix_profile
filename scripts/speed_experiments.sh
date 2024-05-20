#!bin/bash
vector_size=1000000
window_size=64

for i in $(seq 1 3 16)
do
    echo "Running experiment $i"
    export OMP_NUM_THREADS=$i
    ../build/experiments/experiments $vector_size $window_size > res_tmp
    cat res_tmp
    TIME=`grep "STOMP" res_tmp | awk '{print $4}'`
    echo "$i,${TIME}" >> stomp_1m.txt
    rm res_tmp
done