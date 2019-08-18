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

def generate_list_of_experiments(SR_config1, SR_config_2):    
    # select all the jobs that are SRGP and n_eval != -1
    sql = 'SELECT job_ID,n_evals from jobs WHERE algorithm = "SRGP" and n_evals != -1;'
    results = database.run_select_qry(sql)
    list_of_jobs = []
    # use the appropriate SR_config based on the number
    for job in results:
        job_ID = job[0]
        n_evals = job[1]
        exp_num = int(job_ID.split('_')[0])
        if int(exp_num) < 20:
            SR_config = SR_config1
        else:
            SR_config = SR_config2
        # assign the SRURGS parameter values
        train = '$PYSRURGSDIR/csvs/benchmarks/'+str(exp_num)+'_train.csv'
        # INSERT with IGNORE so just insert all the finished SRGP jobs 
        run_ID = job_ID.split('.')[1]
        path_to_db = give_db_path(train, run_ID)
        algorithm = 'SRURGS'         
        funcs_arity_two = ','.join(SR_config._n_functions)
        funcs_arity_one = ','.join(SR_config._f_functions)
        max_num_fit_params = SR_config._max_num_fit_params
        max_permitted_trees = SR_config._max_permitted_trees
        arguments = run_SRURGS(n_evals, train, path_to_db, 
                                   max_num_fit_params, max_permitted_trees, 
                                   funcs_arity_two, funcs_arity_one)
        try:
            n_eval2 = int(arguments.split(' ')[-1])
            if n_eval2 < 20000:
                pdb.set_trace()
        except:
            pdb.set_trace()
        list_of_jobs.append([algorithm, arguments])
    return list_of_jobs


if __name__ == '__main__':
    print('about to generate jobs')
    jobs = generate_list_of_experiments(SR_config1, SR_config2)
    print('about to submit jobs')
    database.submit_job_to_db(jobs)
    
