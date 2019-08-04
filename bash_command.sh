EXP_PER_RUN=1000
let "TIME_PER_RUN=$EXP_PER_RUN * 3"
while [[ 1 -eq 1 ]]
do
    timeout ${TIME_PER_RUN}s winpty python pySRURGS.py ./Generalized_Formula_for_Compressibility_Factor_Z.csv $EXP_PER_RUN
done
    