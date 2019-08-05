EXP_PER_RUN=10
let "TIME_PER_RUN=$EXP_PER_RUN * 3"
while [[ 1 -eq 1 ]]
do
    timeout ${TIME_PER_RUN}s winpty python pySRURGS.py ./csvs/x1_squared_minus_five_x3.csv $EXP_PER_RUN single_process
done
    