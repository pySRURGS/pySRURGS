# I am using sh through python so as to bypass issue with python 
# multiprocessing on windows and to avoid writing too much bash code
import sys
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
iters = 100

def run_experiments(SRconfig, start_index, count_experiments, n_runs):
    funcs_arity_two = ','.join(SR_config._n_functions)
    funcs_arity_one = ','.join(SR_config._f_functions)
    max_num_fit_params = SR_config._max_num_fit_params
    max_size_trees = SR_config._max_size_trees
    # first twenty problems
    for z in range(start_index,count_experiments):
        print("Experiment number:", z)
        train = './benchmarks/'+str(z)+'_train.csv'
        for j in range(0,n_runs):
            print("Run number:", j)
            run_ID = str(j)        
            sh.python('pySRURGS.py',
                  '-run_ID', run_ID,
                  '-funcs_arity_two', funcs_arity_two,
                  '-max_num_fit_params', max_num_fit_params,
                  '-max_size_trees', max_size_trees,
                  train, iters)

run_experiments(SR_config1, 0, 20)
run_experiments(SR_config1, 20, 80)
