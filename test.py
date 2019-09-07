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
import numpy as np
from pySRURGS import Result
from sqlitedict import SqliteDict
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
qrtic_polynml_csv = './csv/quartic_polynomial.csv'
qrtic_polynml_db = './db/quartic_polynomial.db'
x1sqrd_csv = './csv/x1_squared_minus_five_x3.csv'
x1sqrd_db = './db/x1_squared_minus_five_x3.db'
benchmarks_dir = './csv/benchmarks'

##### CLI ARGUMENT STRINGS ######

x = [None]*10
x[0] = '-deterministic -single ./csv/quartic_polynomial.csv 10'
x[1] = '-deterministic -max_num_fit_params 0 ./csv/quartic_polynomial.csv 10'
x[2] = '-deterministic -max_num_fit_params 5 ./csv/quartic_polynomial.csv 10'
x[3] = '-deterministic -funcs_arity_two add,sub,div -max_num_fit_params 5 ./csv/quartic_polynomial.csv 10'
x[4] = '-deterministic -funcs_arity_one tan,exp -max_num_fit_params 5 ./csv/quartic_polynomial.csv 10'
x[5] = '-deterministic -max_permitted_trees 10 -max_num_fit_params 5 ./csv/quartic_polynomial.csv 10'
x[6] = '-exhaustive -funcs_arity_two add,sub -funcs_arity_one sin -max_permitted_trees 3 -max_num_fit_params 1 ./csv/quartic_polynomial 10'
x[7] = '-exhaustive -funcs_arity_two add,sub -max_permitted_trees 3 -max_num_fit_params 1 ./csv/quartic_polynomial.csv 10'
x[8] = '-plotting -max_permitted_trees 10 -max_num_fit_params 5 ./csv/quartic_polynomial.csv 10'

##### CLI TEST INPUTS ######


test_inputs = dict()
test_inputs[x[0]] = ['pySRURGS.py', '-single', '-deterministic', 
                     qrtic_polynml_csv, 10]
test_inputs[x[1]] = ['pySRURGS.py', '-deterministic', '-max_num_fit_params', 0, 
                     qrtic_polynml_csv, 10]
test_inputs[x[2]] = ['pySRURGS.py', '-deterministic', '-max_num_fit_params', 5, 
                     qrtic_polynml_csv, 10]
test_inputs[x[3]] = ['pySRURGS.py', '-deterministic', '-funcs_arity_two', 
                     'add,sub,div', '-max_num_fit_params', 5, qrtic_polynml_csv, 
                     10]
test_inputs[x[4]] = ['pySRURGS.py', '-deterministic', '-funcs_arity_one', 
                     'tan,exp', '-max_num_fit_params', 5, qrtic_polynml_csv, 10]
test_inputs[x[5]] = ['pySRURGS.py', '-deterministic', '-max_permitted_trees', 
                     10, '-max_num_fit_params', 5, qrtic_polynml_csv, 10]
test_inputs[x[6]] = ['pySRURGS.py', '-exhaustive', '-funcs_arity_two', 
                     'add,sub', '-funcs_arity_one', 'sin', 
                     '-max_permitted_trees', 3, '-max_num_fit_params', 1, 
                     qrtic_polynml_csv, 10]
test_inputs[x[7]] = ['pySRURGS.py', '-exhaustive', '-funcs_arity_two', 
                     'add,sub', '-max_permitted_trees', 3, 
                     '-max_num_fit_params', 1, qrtic_polynml_csv, 10]
test_inputs[x[8]] = ['pySRURGS.py', '-plotting', '-max_permitted_trees', 10, 
                     '-max_num_fit_params', 5, qrtic_polynml_csv, 10]

##### CLI TEST OUTPUTS ######

test_outputs = dict()
test_outputs[x[0]] = "0.000283258      0.999993  (p0**2 + p2**x + (p0 - 2*p2)*(p0 + (p2/(p0 + x))**p1))/(p0 - 2*p2)                   1.62E+01,-4.18E+01,1.65E+01"
test_outputs[x[1]] = "46.2653   0.299924  3*x**2*(x**2 - x + 1)"
test_outputs[x[2]] = "0.235363    0.994111     (-p3 + p4*(-p1 + (p4 + x)**(p0 + p1) + (p1 + p2**p1 + x)**(p1**p4 - p3)))/p4    2.79E+00,2.44E-08,1.10E+00,3.52E-01,7.12E-01"
test_outputs[x[3]] = " 2.33541   0.938364  (p0*p3*(p0 + p4) + p0*(p2 + p4 - x) + (p0 + p4)*(-p1 + p4 + x))/(p0*(p0 + p4))  2.21E-01,4.86E-01,-7.01E-02,-3.81E+00,1.27E+00"
test_outputs[x[4]] = " 5.89909e-05   0.999999     -p0**p2*(p0*p2)**(p2**x)*(p0 + p2)*(p2 - p3) + (p0 + x)**(p2*p3)*(2*p1**(x + 1) - p2)       1.25E+00,1.03E+00,2.04E+00,2.11E+00,1.00E+00"
test_outputs[x[5]] = "2.33541   0.938364  p2**p3 - p4*x + x        1.00E+00,1.00E+00,5.29E-01,1.00E+00,-2.85E+00"
test_outputs[x[6]] = "Number possible equations: 12.0"
test_outputs[x[7]] = "Number possible equations: 42.0"
test_outputs[x[8]] = " 0.6804   0.981226     (p0**(p4 + 1))**(p4**x)  1.03E+00,1.00E+00,1.00E+00,1.00E+00,7.06E+0"

def refresh_db(path_to_db):
    try:
        os.remove(path_to_db)
    except:
        pass

def test_command_line_code():
    path_to_db = qrtic_polynml_db
    print('Started run_command_line_tests')
    # Command line interface
    for arguments_string in test_inputs.keys():
        refresh_db(path_to_db)
        input_args = test_inputs[arguments_string]
        output_string = sh.python(*input_args)
        print(output_string)
        assert test_outputs[arguments_string] in output_string, print(
            arguments_string)
        print('Finished', *input_args)

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
    refresh_db(path_to_db)
    SRconfig = pySRURGS.SymbolicRegressionConfig(
        n_funcs, f_funcs, max_num_fit_params, max_permitted_trees)
    # the the -count functionality
    test_f_funcs = 'tan,exp,cos,sin,log,sinh,cosh,tanh'
    if test_f_funcs is None or test_f_funcs == '':
        test_f_funcs = []
    else:
        test_f_funcs = test_f_funcs.split(',')
        test_f_funcs = pySRURGS.check_validity_suggested_functions(
            test_f_funcs, 1)
    SRconfigtest = pySRURGS.SymbolicRegressionConfig(
        n_funcs, f_funcs, max_num_fit_params, max_permitted_trees)
    num_equations = pySRURGS.count_number_equations(path_to_csv, SRconfigtest)
    num_equations = int(num_equations)
    max_attempts = 15
    # test the basic functionality
    for i in range(0, max_attempts):
        pySRURGS.uniform_random_global_search_once(
            path_to_db, path_to_csv, SRconfig)
    assert type(num_equations) == mpmath.ctx_mp_python.mpf
    print(num_equations)
    # get the MSE of the first run
    result_list = pySRURGS.compile_results(path_to_db, path_to_csv, SRconfig)
    (_, _, _, _, _, dataset, _, _, _) = pySRURGS.setup(path_to_csv, SRconfig)
    MSE_calc = np.sum((result_list._results[0].predict(
        dataset) - dataset._y_data)**2/len(dataset._y_data))
    MSE_1st_run = result_list._results[0]._MSE
    print(MSE_calc, MSE_1st_run)
    # test the multiprocessing functionality and that MSE decreases with 1000 runs
    max_attempts = 100
    for i in tqdm.tqdm(range(0, max_attempts)):
        pySRURGS.uniform_random_global_search_once(
            path_to_db, path_to_csv, SRconfig)
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
    SRconfig = pySRURGS.SymbolicRegressionConfig(
        n_funcs, f_funcs, 0, max_permitted_trees)
    for i in tqdm.tqdm(range(0, max_attempts)):
        pySRURGS.uniform_random_global_search_once(
            path_to_db, path_to_csv, SRconfig)
    # test max_num_fit_params 5
    SRconfig = pySRURGS.SymbolicRegressionConfig(
        n_funcs, f_funcs, 5, max_permitted_trees)
    for i in tqdm.tqdm(range(0, max_attempts)):
        pySRURGS.uniform_random_global_search_once(
            path_to_db, path_to_csv, SRconfig)
    # test funcs_arity_two = 'add,sub,div'
    test_n_funcs = 'add,sub,div'
    test_n_funcs = test_n_funcs.split(',')
    test_n_funcs = pySRURGS.check_validity_suggested_functions(test_n_funcs, 2)
    SRconfig_test_n_funcs = pySRURGS.SymbolicRegressionConfig(
        test_n_funcs, f_funcs, 5, max_permitted_trees)
    for i in tqdm.tqdm(range(0, max_attempts)):
        pySRURGS.uniform_random_global_search_once(
            path_to_db, path_to_csv, SRconfig)
    # test funcs_arity_one = 'tan,exp'
    SRconfig_test_f_funcs = pySRURGS.SymbolicRegressionConfig(
        n_funcs, test_f_funcs, 5, max_permitted_trees)
    for i in tqdm.tqdm(range(0, max_attempts)):
        pySRURGS.uniform_random_global_search_once(
            path_to_db, path_to_csv, SRconfig_test_f_funcs)
    # test max_permitted_trees = 10
    test_max_permitted_trees = 10
    SRconfig_test_permitted_trees = pySRURGS.SymbolicRegressionConfig(
        n_funcs, f_funcs, 5, test_max_permitted_trees)
    for i in tqdm.tqdm(range(0, max_attempts)):
        pySRURGS.uniform_random_global_search_once(
            path_to_db, path_to_csv, SRconfig_test_permitted_trees)
    # plot results
    pySRURGS.plot_results(path_to_db, path_to_csv, SRconfig)
    # generate benchmarks
    pySRURGS.generate_benchmarks()
    print('Finished run_python_tests')
    # print DB inspection code
    SR_config = pySRURGS.SymbolicRegressionConfig()
    path_to_csv = './csv/quartic_polynomial.csv'
    path_to_db = './db/quartic_polynomial.db'
    pySRURGS.assign_n_evals(path_to_db)
    with SqliteDict(path_to_db, autocommit=True) as results_dict:
        best_result = results_dict['best_result']
        number_equations = results_dict['n_evals']
    result_list, dataset = pySRURGS.get_resultlist(
        path_to_db, path_to_csv, SR_config)
    result_list.sort()
    # after running sort, zero^th element is the best result
    best_result = result_list._results[0]
    print("R^2:", best_result._R2, "Equation:", best_result._simple_equation,
          "Unsimplified Equation:", best_result._equation)
    result_list.print(dataset._y_data)
    # run tests for exhaustive search
    refresh_db(path_to_db)
    SRconfig = pySRURGS.SymbolicRegressionConfig(['add', 'sub'], ['sin'], 1, 3)
    pySRURGS.exhaustive_search(path_to_db, path_to_csv, SRconfig, mode='multi')
    refresh_db(path_to_db)
    SRconfig = pySRURGS.SymbolicRegressionConfig(['add', 'sub'], [], 1, 3)
    pySRURGS.exhaustive_search(
        path_to_db, path_to_csv, SRconfig, mode='single')


if __name__ == '__main__':
    test_command_line_code()
    exit(0)
    test_python_code()
