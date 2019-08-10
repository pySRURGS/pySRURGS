import sys
import os 
sys.path.append('./..')
import pySRURGS

try:
    import sh
except ImportError:
    # fallback: emulate the sh API with pbs
    import pbs
    class Sh(object):
        def __getattr__(self, attr):
            return pbs.Command(attr)
    sh = Sh()

SR_config1, SR_config2 = pySRURGS.generate_benchmarks_SRconfigs()

def run_SRGP(csv_path, path_to_db, numgens, popsize, int_max_params, 
             n_functions, f_functions=None):
    if f_functions is None or f_functions == '':
        sh.python('SRGP.py', csv_path, path_to_db, numgens, popsize,
                  int_max_params, n_functions)
    else:
        sh.python('SRGP.py', csv_path, path_to_db, numgens, popsize,
                  int_max_params, n_functions, '-f_functions', f_functions)

def give_db_path(path_to_csv, run_ID):
    return './../db/'+os.path.basename(path_to_csv)[:-3]+run_ID+'.SRGP.db'

def run_experiments(SRconfig, start_index, count_experiments, n_runs):
    funcs_arity_two = ','.join(SR_config._n_functions)
    funcs_arity_one = ','.join(SR_config._f_functions)
    max_num_fit_params = SR_config._max_num_fit_params
    max_size_trees = SR_config._max_size_trees
    popsize = 500
    numgens = 20
    # first twenty problems
    for z in range(start_index,count_experiments):
        print("SRGP experiment number:", z)
        train = './benchmarks/'+str(z)+'_train.csv'
        for j in range(0,n_runs):
            print("Run number:", j, "out of", n_runs)
            run_ID = str(j)
            path_to_db = give_db_path(train, run_ID) 
            run_SRGP(train, path_to_db, numgens, popsize, max_num_fit_params, 
                     funcs_arity_two, funcs_arity_one)

run_experiments(SR_config1, 0, 20, 10)
run_experiments(SR_config2, 20, 80, 10)
