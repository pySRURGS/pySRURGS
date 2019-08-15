import sys
import os 
sys.path.append('./..')
import pySRURGS
import database

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
        arguments = ' '.join(['$PYSRURGSDIR/experiments/SRGP.py', 
                              csv_path, path_to_db, str(numgens), str(popsize), 
                              str(int_max_params), n_functions])
    else:
        arguments = ' '.join(['$PYSRURGSDIR/experiments/SRGP.py', 
                              csv_path, path_to_db, str(numgens), str(popsize), 
                              str(int_max_params), n_functions, '-f_functions', 
                              f_functions])
    return arguments 
    
def give_db_path(path_to_csv, run_ID):
    return '$PYSRURGSDIR/db/'+os.path.basename(path_to_csv)[:-3]+run_ID+'.SRGP.db'

def generate_list_of_experiments(SR_config, start_index, count_experiments, n_runs):
    list_of_jobs = []
    funcs_arity_two = ','.join(SR_config._n_functions)
    funcs_arity_one = ','.join(SR_config._f_functions)
    max_num_fit_params = SR_config._max_num_fit_params
    max_size_trees = SR_config._max_permitted_trees
    popsize = 500
    numgens = 40
    # first twenty problems
    for z in range(start_index,count_experiments):
        train = '$PYSRURGSDIR/csvs/benchmarks/'+str(z)+'_train.csv'
        for j in range(0,n_runs):
            run_ID = str(j)
            path_to_db = give_db_path(train, run_ID)             
            algorithm = 'SRGP'
            arguments = run_SRGP(train, path_to_db, numgens, popsize, max_num_fit_params, 
                                 funcs_arity_two, funcs_arity_one)
            list_of_jobs.append([algorithm, arguments])
    return list_of_jobs

if __name__ == '__main__':
    jobs1 = generate_list_of_experiments(SR_config1, 0, 20, 10)
    jobs2 = generate_list_of_experiments(SR_config2, 20, 80, 10)
    #database.submit_job_to_db(jobs1)
    database.submit_job_to_db(jobs2)
    
