import sys
import os 
sys.path.append('./..')
import pySRURGS
import database
import pdb

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

def run_SRURGS(n_evals, csv_path, path_to_db, max_num_fit_params, max_permitted_trees, 
               funcs_arity_two, funcs_arity_one=None):
    if funcs_arity_one is not None or funcs_arity_one != '':
        arguments = ' '.join(['$PYSRURGSDIR/pySRURGS.py',
                              '-funcs_arity_two', funcs_arity_two,
                              '-max_num_fit_params', str(max_num_fit_params),
                              '-max_permitted_trees', str(max_permitted_trees),
                              '-path_to_db', path_to_db,
                              csv_path, str(n_evals)])
    else:
        arguments = ' '.join(['$PYSRURGSDIR/pySRURGS.py', 
                              '-funcs_arity_two', funcs_arity_two,
                              '-funcs_arity_one', funcs_arity_one,
                              '-max_num_fit_params', str(max_num_fit_params),
                              '-max_permitted_trees', str(max_permitted_trees),
                              '-path_to_db', path_to_db,
                              csv_path, str(n_evals)])
    return arguments 

def give_db_path(path_to_csv, run_ID):
    return '$PYSRURGSDIR/db/'+os.path.basename(path_to_csv)[:-3]+run_ID+'.SRURGS.db'

def generate_list_of_experiments(SR_config, start_index, count_experiments, n_runs):
    list_of_jobs = []
    funcs_arity_two = ','.join(SR_config._n_functions)
    funcs_arity_one = ','.join(SR_config._f_functions)
    max_num_fit_params = SR_config._max_num_fit_params
    max_permitted_trees = SR_config._max_permitted_trees
    for z in range(start_index,start_index+count_experiments):
        train = '$PYSRURGSDIR/csvs/benchmarks/'+str(z)+'_train.csv'
        for j in range(0,n_runs):
            run_ID = str(j)
            path_to_db = give_db_path(train, run_ID)
            SRGP_db = path_to_db.replace("SRURGS", "SRGP")
            SRGP_db = SRGP_db.split('/')[-1]
            n_evals = database.find_matching_SRGP_job_n_evals(SRGP_db)
            if n_evals == -1:
                continue
            SRURGS_db = path_to_db.split('/')[-1]            
            SRURGS_check = database.find_matching_SRURGS_job(path_to_db)
            if SRURGS_check is None:
                pass
            else: # job already exists, don't resubmit 
                continue 
            algorithm = 'SRURGS'               
            arguments = run_SRURGS(n_evals, train, path_to_db, 
                                   max_num_fit_params, max_permitted_trees, 
                                   funcs_arity_two, funcs_arity_one)
            list_of_jobs.append([algorithm, arguments])
    return list_of_jobs


if __name__ == '__main__':
    jobs1 = generate_list_of_experiments(SR_config1, 0, 20, 10)
    jobs2 = generate_list_of_experiments(SR_config2, 20, 80, 10)    
    database.submit_job_to_db(jobs1)
    database.submit_job_to_db(jobs2)
    
