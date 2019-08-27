'''
    The test script for pySRURGS - runs using both the command line interface 
    and the python interface
    
'''
import os
import sys 
import glob
import pdb
import pySRURGS
import mpmath
import tqdm
from pySRURGS import Result
try:
    import sh
except ImportError:
    # fallback: emulate the sh API with pbs
    import pbs
    class Sh(object):
        def __getattr__(self, attr):
            return pbs.Command(attr)
    sh = Sh()

dbs_dir = './db'
qrtic_polynml_csv = './csvs/quartic_polynomial.csv'
qrtic_polynml_db = './db/quartic_polynomial.db'
x1sqrd_csv = './csvs/x1_squared_minus_five_x3.csv'
x1sqrd_db = './db/x1_squared_minus_five_x3.db'
benchmarks_dir = './csvs/benchmarks'

def test_command_line_code():
    print('Started run_command_line_tests')
    # Command line interface
    sh.python('pySRURGS.py', qrtic_polynml_csv, 10)
    print('Finished basic run')
    sh.python('pySRURGS.py', '-single', qrtic_polynml_csv, 10)
    print('Finished single cpu run')
    sh.python('pySRURGS.py', '-count', qrtic_polynml_csv, 10)
    print('Finished count equations run')
    sh.python('pySRURGS.py', '-max_num_fit_params', 0, qrtic_polynml_csv, 10)
    print('Finished zero fit params run')
    sh.python('pySRURGS.py', '-max_num_fit_params', 5, qrtic_polynml_csv, 10)
    print('Finished five fit params run')
    sh.python('pySRURGS.py', '-funcs_arity_two', 'add,sub,div', '-max_num_fit_params', 5, qrtic_polynml_csv, 10)
    print('Finished add sub div funcs arity two run')
    sh.python('pySRURGS.py', '-funcs_arity_one', 'tan,exp', '-max_num_fit_params', 5, qrtic_polynml_csv, 10)
    print('Finished tan,exp funcs arity one run')
    sh.python('pySRURGS.py', '-max_permitted_trees', 10, '-max_num_fit_params', 5, qrtic_polynml_csv, 10)
    print('Finished run_command_line_tests')

def test_python_code():
    print('Started run_python_tests')
    # Python level code
    # load the default command line values
    defaults = pySRURGS.defaults_dict
    n_funcs = defaults['funcs_arity_two']
    n_funcs = n_funcs.split(',')
    n_funcs = pySRURGS.check_validity_suggested_functions(n_funcs, 2)     
    f_funcs = None
    if f_funcs is None or f_funcs == '':
        f_funcs = []
    else:
        f_funcs = f_funcs.split(',')
        f_funcs = pySRURGS.check_validity_suggested_functions(f_funcs, 1)
    max_num_fit_params = defaults['max_num_fit_params']
    max_permitted_trees = defaults['max_permitted_trees']
    # assign the arguments used for later assessment of the algorithm
    path_to_db = qrtic_polynml_db
    path_to_csv = qrtic_polynml_csv
    try:
        os.remove(path_to_db)
    except:
        pass
    SRconfig = pySRURGS.SymbolicRegressionConfig(n_funcs, f_funcs, max_num_fit_params, max_permitted_trees)
    # the the -count functionality
    num_equations = pySRURGS.count_number_equations(path_to_csv, SRconfig)
    max_attempts = 15
    # test the basic functionality
    for i in range(0,max_attempts):
        pySRURGS.uniform_random_global_search_once(path_to_db, path_to_csv, SRconfig)
    assert type(num_equations) == mpmath.ctx_mp_python.mpf
    print(num_equations)
    # get the MSE of the first run 
    result_list = pySRURGS.compile_results(path_to_db, path_to_csv, SRconfig)
    MSE_1st_run = result_list._results[0]._MSE
    # test the multiprocessing functionality and that MSE decreases with 1000 runs
    max_attempts = 100
    for i in tqdm.tqdm(range(0,max_attempts)):
        pySRURGS.uniform_random_global_search_once(path_to_db, path_to_csv, SRconfig)
    result_list = pySRURGS.compile_results(path_to_db, path_to_csv, SRconfig)
    MSE_2nd_run = result_list._results[0]._MSE
    assert MSE_2nd_run <= MSE_1st_run
    print(num_equations)    
    # first remove all the benchmark files from the repository 
    benchmarks = glob.glob(os.path.join(benchmarks_dir, '*'))
    for benchmark_file in benchmarks:
        os.remove(benchmark_file)
    # test max_num_fit_params 0 
    max_attempts = 20
    SRconfig = pySRURGS.SymbolicRegressionConfig(n_funcs, f_funcs, 0, max_permitted_trees)
    for i in tqdm.tqdm(range(0,max_attempts)):
        pySRURGS.uniform_random_global_search_once(path_to_db, path_to_csv, SRconfig)
    # test max_num_fit_params 5
    SRconfig = pySRURGS.SymbolicRegressionConfig(n_funcs, f_funcs, 5, max_permitted_trees)
    for i in tqdm.tqdm(range(0,max_attempts)):
        pySRURGS.uniform_random_global_search_once(path_to_db, path_to_csv, SRconfig)
    # test funcs_arity_two = 'add,sub,div'
    test_n_funcs = 'add,sub,div'
    test_n_funcs = test_n_funcs.split(',')
    test_n_funcs = pySRURGS.check_validity_suggested_functions(test_n_funcs, 2)    
    SRconfig_test_n_funcs = pySRURGS.SymbolicRegressionConfig(test_n_funcs, f_funcs, 5, max_permitted_trees)
    for i in tqdm.tqdm(range(0,max_attempts)):
        pySRURGS.uniform_random_global_search_once(path_to_db, path_to_csv, SRconfig)
    # test funcs_arity_one = 'tan,exp'
    test_f_funcs = 'tan,exp'
    if test_f_funcs is None or test_f_funcs == '':
        test_f_funcs = []
    else:
        test_f_funcs = test_f_funcs.split(',')
        test_f_funcs = pySRURGS.check_validity_suggested_functions(test_f_funcs, 1)
    SRconfig_test_f_funcs = pySRURGS.SymbolicRegressionConfig(n_funcs, test_f_funcs, 5, max_permitted_trees)
    for i in tqdm.tqdm(range(0,max_attempts)):
        pySRURGS.uniform_random_global_search_once(path_to_db, path_to_csv, SRconfig_test_f_funcs)
    # test max_permitted_trees = 10
    test_max_permitted_trees = 10
    SRconfig_test_permitted_trees = pySRURGS.SymbolicRegressionConfig(n_funcs, f_funcs, 5, test_max_permitted_trees)
    for i in tqdm.tqdm(range(0,max_attempts)):
        pySRURGS.uniform_random_global_search_once(path_to_db, path_to_csv, SRconfig_test_permitted_trees)
    # plot results 
    pySRURGS.plot_results(path_to_db, path_to_csv, SRconfig)
    # generate benchmarks
    pySRURGS.generate_benchmarks()
    print('Finished run_python_tests')

if __name__ == '__main__':
    test_command_line_code()
    test_python_code()

