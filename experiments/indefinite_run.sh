while [[ 1 -eq 1 ]]
do
winpty python pySRURGS.py -plotting -max_num_fit_params 2 -funcs_arity_one sin,cos,exp ./csvs/cos_x_times_exp_neg_0_pt_01_x.csv 100
done