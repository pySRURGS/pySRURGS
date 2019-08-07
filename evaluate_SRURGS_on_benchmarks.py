# I am using sh through python so as to bypass issue with python 
# multiprocessing on windows and to avoid writing too much bash code
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
iters = 1000 
# first twenty problems
funcs_arity_two = ','.join(SR_config1._n_functions)
funcs_arity_one = ','.join(SR_config1._f_functions)
max_num_fit_params = SR_config1._max_num_fit_params
max_size_trees = SR_config1._max_size_trees
for z in range(0,20):
    print("Experiment number:", z)
    train = './benchmarks/'+str(z)+'_train.csv'
    for j in range(0,10):
        print("Run number:", j)
        run_ID = str(j)        
        sh.python('pySRURGS.py',
                  '-run_ID', run_ID,
                  '-funcs_arity_two', funcs_arity_two,
                  '-max_num_fit_params', max_num_fit_params,
                  '-max_size_trees', max_size_trees,
                  train, iters)

# remaining 80 cases 
# first twenty problems
funcs_arity_two = ','.join(SR_config2._n_functions)
funcs_arity_one = ','.join(SR_config2._f_functions)
max_num_fit_params = SR_config2._max_num_fit_params
max_size_trees = SR_config2._max_size_trees
for z in range(20,99):
    print("Experiment number:", z)
    train = './benchmarks/'+str(z)+'_train.csv'    
    for j in range(0,10):
        print("Run number:", j)
        run_ID = str(j)
        sh.python('pySRURGS.py', 
                  '-run_ID', run_ID,
                  '-funcs_arity_two', funcs_arity_two,
                  '-funcs_arity_one', funcs_arity_one, 
                  '-max_num_fit_params', max_num_fit_params,
                  '-max_size_trees', max_size_trees,
                  train, iters)
