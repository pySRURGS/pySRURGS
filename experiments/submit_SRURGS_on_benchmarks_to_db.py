import sys
import os 
sys.path.append('./..')
import pySRURGS
import database


# TODO - change the code so that 'iters' reflects the number of iterations 
# run by the corresponding SRGP run

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

def run_SRURGS(SRconfig, start_index, count_experiments):
    # nruns determined by the number of runs in the corresponding SRGP 
    funcs_arity_two = ','.join(SR_config._n_functions)
    funcs_arity_one = ','.join(SR_config._f_functions)
    max_num_fit_params = SR_config._max_num_fit_params
    max_size_trees = SR_config._max_size_trees
    # first twenty problems
    for z in range(start_index,count_experiments):
        print("Experiment number:", z)
        train = '$PYSRURGS/benchmarks/'+str(z)+'_train.csv'
        iters = find_matching_SRGP_job_n_evals(train)[0]
        for j in range(0,n_runs):
            print("Run number:", j)
            run_ID = str(j)        
            command = ['python', 'pySRURGS.py',
                       '-run_ID', run_ID,
                       '-funcs_arity_two', funcs_arity_two,
                       '-max_num_fit_params', str(max_num_fit_params),
                       '-max_size_trees', str(max_size_trees),
                       str(train), str(iters)]

def give_db_path(path_to_csv, run_ID):
    return '$PYSRURGSDIR/db/'+os.path.basename(path_to_csv)[:-3]+run_ID+'.SRURGS.db'

def generate_list_of_experiments(SR_config, start_index, count_experiments, n_runs):
    list_of_jobs = []
    funcs_arity_two = ','.join(SR_config._n_functions)
    funcs_arity_one = ','.join(SR_config._f_functions)
    max_num_fit_params = SR_config._max_num_fit_params
    max_size_trees = SR_config._max_permitted_trees
    for z in range(start_index,count_experiments):
        train = '$PYSRURGSDIR/csvs/benchmarks/'+str(z)+'_train.csv'
        for j in range(0,n_runs):
            run_ID = str(j)
            path_to_db = give_db_path(train, run_ID)             
            algorithm = 'SRURGS'
            arguments = run_SRURGS(SR_config, start_index, count_experiments)
            list_of_jobs.append([algorithm, arguments])
    return list_of_jobs


if __name__ == '__main__':
    jobs1 = generate_list_of_experiments(SR_config1, 0, 20, 10)
    jobs2 = generate_list_of_experiments(SR_config2, 20, 80, 10)
    #database.submit_job_to_db(jobs1)
    database.submit_job_to_db(jobs2)
    
