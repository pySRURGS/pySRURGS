for i in $(seq 0 19)
do
    for j in $(seq 0 9)
    do 
        winpty python pySRURGS.py -run_ID $j ./benchmarks/${i}_train.csv 1000
    done
done

for i in $(seq 20 99)
do 
    winpty python pySRURGS.py ./benchmarks/${i}_train.csv 10000
done
