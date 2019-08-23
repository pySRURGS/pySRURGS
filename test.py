import sh
import os
import sys 
import glob
import pySRURGS

dbs_dir = './db'
qrtic_polynml_csv = './csvs/quartic_polynomial.csv'
benchmarks_dir = './csvs/benchmarks'

def run_command_line_tests():
    # Command line interface
    sh.python('pySRURGS.py', qrtic_polynml_csv, 10)
    sh.python('pySRURGS.py', '-single', qrtic_polynml_csv, 10)
    sh.python('pySRURGS.py', '-count', qrtic_polynml_csv, 10)
    sh.python('pySRURGS.py', '-benchmarks', qrtic_polynml_csv, 10)
    sh.python('pySRURGS.py', '-max_num_fit_params', 0, qrtic_polynml_csv, 10)
    sh.python('pySRURGS.py', '-max_num_fit_params', 5, qrtic_polynml_csv, 10)
    sh.python('pySRURGS.py', '-funcs_arity_two', 'add,sub,div', '-max_num_fit_params', 5, qrtic_polynml_csv, 10)
    sh.python('pySRURGS.py', '-funcs_arity_one', 'tan,exp', '-max_num_fit_params', 5, qrtic_polynml_csv, 10)
    sh.python('pySRURGS.py', '-max_permitted_trees', 10, '-max_num_fit_params', 5, qrtic_polynml_csv, 10)

def run_python_tests():
    # Python level code
    # load the default command line values
    defaults = pySRURGS.defaults_dict
    n_funcs = defaults['funcs_arity_two']
    f_funcs = None
    max_num_fit_params = defaults['max_num_fit_params']
    max_permitted_trees = defaults['max_permitted_trees']
    # assign the arguments used for later assessment of the algorithm
    path_to_db = os.path.join(dbs_dir, 'quartic_polynomial.db')
    path_to_csv = qrtic_polynml_csv
    SRconfig = pySRURGS.SymbolicRegressionConfig(n_funcs, f_funcs, max_num_fit_params, max_permitted_trees)
    # the the -count functionality
    num_equations = pySRURGS.count_number_equations(path_to_csv, SRconfig)
    max_attempts = 5
    # test the basic functionality
    for i in range(0,max_attempts):
        pySRURGS.uniform_random_global_search_once(path_to_db, path_to_csv, SRconfig)
    assert type(num_equations) == float
    print(num_equations)
    # get the MSE of the first run 
    result_list = pySRURGS.compile_results(path_to_db, path_to_csv, SRconfig)
    MSE_1st_run = result_list._results[0]._MSE
    # test the multiprocessing functionality and that MSE decreases with 1000 runs
    max_attempts = 1000
    for i in tqdm.tqdm(range(0,max_attempts)):
        pySRURGS.uniform_random_global_search_once(path_to_db, path_to_csv, SRconfig)
    result_list = pySRURGS.compile_results(path_to_db, path_to_csv, SRconfig)
    MSE_2nd_run = result_list._results[0]._MSE
    assert MSE_2nd_run < MSE_1st_run
    print(num_equations)    
    # first remove all the benchmark files from the repository 
    benchmarks = glob.glob(os.path.join(benchmarks_dir, '*'))
    for benchmark_file in benchmarks:
        os.remove(benchmark_file)
    # test the generation of benchmarks 
    pySRURGS.generate_benchmarks()
    # test max_num_fit_params 0 
    SRconfig = pySRURGS.SymbolicRegressionConfig(n_funcs, f_funcs, 0, max_permitted_trees)
    for i in range(0,max_attempts):
        pySRURGS.uniform_random_global_search_once(path_to_db, path_to_csv, SRconfig)
    # test max_num_fit_params 5
    SRconfig = pySRURGS.SymbolicRegressionConfig(n_funcs, f_funcs, 5, max_permitted_trees)
    for i in range(0,max_attempts):
        pySRURGS.uniform_random_global_search_once(path_to_db, path_to_csv, SRconfig)
    # test funcs_arity_two = 'add,sub,div'
    test_n_funcs = 'add,sub,div'
    SRconfig_test_n_funcs = pySRURGS.SymbolicRegressionConfig(test_n_funcs, f_funcs, 5, max_permitted_trees)
    for i in range(0,max_attempts):
        pySRURGS.uniform_random_global_search_once(path_to_db, path_to_csv, SRconfig)
    # test funcs_arity_one = 'tan,exp'
    test_f_funcs = 'tan,exp'
    SRconfig_test_n_funcs = pySRURGS.SymbolicRegressionConfig(n_funcs, test_f_funcs, 5, max_permitted_trees)
    for i in range(0,max_attempts):
        pySRURGS.uniform_random_global_search_once(path_to_db, path_to_csv, SRconfig)
    # test max_permitted_trees = 10
    test_max_permitted_trees = 10
    SRconfig_test_n_funcs = pySRURGS.SymbolicRegressionConfig(n_funcs, test_f_funcs, 5, test_max_permitted_trees)
    for i in range(0,max_attempts):
        pySRURGS.uniform_random_global_search_once(path_to_db, path_to_csv, SRconfig)
    

if __name__ == '__main__':
    run_command_line_tests()
    run_python_tests()

'''
usage: pySRURGS.py [-h] [-run_ID RUN_ID] [-single] [-count]
                   [-benchmarks] [-funcs_arity_two FUNCS_ARITY_TWO]
                   [-funcs_arity_one FUNCS_ARITY_ONE]
                   [-max_num_fit_params MAX_NUM_FIT_PARAMS]
                   [-max_size_trees MAX_SIZE_TREES]
                   train iters
                   
optional arguments:
  -h, --help            show this help message and exit
  -run_ID RUN_ID        some text that uniquely identifies this run (default:
                        None)
  -single               run in single processing mode (default: False)
  -count                Instead of doing symbolic regression, just count out
                        how many possible equations for this configuration. No
                        other processing performed. (default: False)
  -benchmarks           Instead of doing symbolic regression, generate the 100
                        benchmark problems. No other processing performed.
                        (default: False)
  -funcs_arity_two FUNCS_ARITY_TWO
                        a comma separated string listing the functions of
                        arity two you want to be considered.
                        Permitted:add,sub,mul,div,pow (default:
                        add,sub,mul,div,pow)
  -funcs_arity_one FUNCS_ARITY_ONE
                        a comma separated string listing the functions of
                        arity one you want to be considered.
                        Permitted:sin,cos,tan,exp,log,sinh,cosh,tanh (default:
                        None)
  -max_num_fit_params MAX_NUM_FIT_PARAMS
                        the maximum number of fitting parameters permitted in
                        the generated models (default: 3)
  -max_permitted_trees MAX_PERMITTED_TREES
                        the number of unique binary trees that are permitted
                        in the generated models - binary trees define the form
                        of the equation, increasing this number tends to
                        increase the complexity of generated equations
                        (default: 1000)
                   
'''
