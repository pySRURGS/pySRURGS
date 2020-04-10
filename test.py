'''
    The test script for pySRURGS - runs using both the command line interface 
    and the python interface
    
'''
import os
import sys
import glob
import pdb
import pySRURGS
from pySRURGS import Result
import mpmath
import tqdm
import numpy as np
import unittest
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

dbs_dir = './db/'
working_db = dbs_dir + 'working.db'
qrtic_polynml_csv = './csv/quartic_polynomial.csv'
qrtic_polynml_db = './db/quartic_polynomial.db'
benchmarks_dir = './csv/benchmarks'
num_iters = 10

def refresh_db(path_to_db):
    try:
        os.remove(path_to_db)
    except:
        pass

test_CLI = True

if test_CLI == True:
    class TestCommandLineInterface(unittest.TestCase):
        
        def setUp(self):
            refresh_db(working_db)

        def test_single_processor_deterministic(self):        
            output = sh.python3('pySRURGS.py', '-single', '-deterministic', 
                                '-path_to_db', working_db, qrtic_polynml_csv, 
                                num_iters)
            output = output.strip()
            print(output)
            n_results = pySRURGS.count_results(working_db)        
            self.assertGreater(n_results, 0.95*num_iters)
            
        def test_zero_fit_params(self):
            output = sh.python3('pySRURGS.py', '-deterministic', 
                                '-max_num_fit_params', 0, '-path_to_db', working_db, 
                                qrtic_polynml_csv, num_iters)
            output = output.strip()
            print(output)
            result_list = pySRURGS.get_resultlist(working_db)
            n_results = pySRURGS.count_results(working_db)
            self.assertGreater(n_results, 0.9*num_iters)
            for i in range(0, n_results):
                self.assertEqual(len(result_list._results[i]._params), 0)
            
        def test_five_fit_params(self):
            output = sh.python3('pySRURGS.py', '-deterministic', 
                                '-max_num_fit_params', 5, '-path_to_db', working_db, 
                                qrtic_polynml_csv, num_iters)
            output = output.strip()
            print(output)
            n_results = pySRURGS.count_results(working_db)
            self.assertGreater(n_results, 0.98*num_iters)

        def test_funcs_arity_two(self):
            output = output = sh.python3('pySRURGS.py', '-deterministic', 
                                        '-funcs_arity_two', 'add,sub,div', 
                                        '-max_num_fit_params', 5, '-path_to_db', 
                                        working_db, qrtic_polynml_csv, num_iters)
            output = output.strip()
            print(output)
            n_results = pySRURGS.count_results(working_db)
            result_list = pySRURGS.get_resultlist(working_db)
            self.assertGreater(n_results, 0.98*num_iters)
            for i in range(0, n_results):
                my_result = result_list._results[i]
                simple_eqn = my_result._simple_equation
                self.assertEqual(('mul' in simple_eqn), False)
                self.assertEqual(('exp' in simple_eqn), False)

        def test_funcs_arity_one(self):
            output = sh.python3('pySRURGS.py', '-deterministic', '-funcs_arity_one', 
                                'tan,exp', '-max_num_fit_params', 5, '-path_to_db', 
                                 working_db, qrtic_polynml_csv, num_iters)
            output = output.strip()
            print(output)
            n_results = pySRURGS.count_results(working_db)
            result_list = pySRURGS.get_resultlist(working_db)
            self.assertGreater(n_results, 0.98*num_iters)
            for i in range(0, n_results):
                my_result = result_list._results[i]
                simple_eqn = my_result._simple_equation
                self.assertEqual(('cos' in simple_eqn), False)

        def test_max_permitted_trees(self):
            output = sh.python3('pySRURGS.py', '-deterministic', 
                                '-max_permitted_trees', 10, '-max_num_fit_params', 5, 
                                '-path_to_db', working_db, qrtic_polynml_csv, 
                                num_iters)
            output = output.strip()
            print(output)
            n_results = pySRURGS.count_results(working_db)
            result_list = pySRURGS.get_resultlist(working_db)
            self.assertGreater(n_results, 0.98*num_iters)    

        def test_combined_parameters_1(self):        
            output = sh.python3('pySRURGS.py', '-exhaustive', '-funcs_arity_two', 
                                'add,sub', '-funcs_arity_one', 'sin', 
                                '-max_permitted_trees', 3, '-max_num_fit_params', 1, 
                                '-path_to_db', working_db, qrtic_polynml_csv, 
                                num_iters)
            output = output.strip()
            print(output)
            self.assertEqual('12.0' in output, True)
            n_results = pySRURGS.count_results(working_db)
            result_list = pySRURGS.get_resultlist(working_db)
            self.assertGreater(n_results, 0.98*num_iters)

        def test_combined_parameters_2(self):
            output = sh.python3('pySRURGS.py', '-exhaustive', '-funcs_arity_two', 
                                'add,sub', '-max_permitted_trees', 3, 
                                '-max_num_fit_params', 1, 
                                '-path_to_db', working_db, qrtic_polynml_csv, 10)
            output = output.strip()
            print(output)
            self.assertEqual('42.0' in output, True)
            n_results = pySRURGS.count_results(working_db)
            result_list = pySRURGS.get_resultlist(working_db)
            self.assertGreater(n_results, 0.98*num_iters)


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
    max_attempts = 15
    # test the basic functionality
    for i in range(0, max_attempts):
        pySRURGS.uniform_random_global_search_once_to_db(None,
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
        pySRURGS.uniform_random_global_search_once_to_db(None,
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
        pySRURGS.uniform_random_global_search_once_to_db(None,
            path_to_db, path_to_csv, SRconfig)
    # test max_num_fit_params 5
    SRconfig = pySRURGS.SymbolicRegressionConfig(
        n_funcs, f_funcs, 5, max_permitted_trees)
    for i in tqdm.tqdm(range(0, max_attempts)):
        pySRURGS.uniform_random_global_search_once_to_db(None,
            path_to_db, path_to_csv, SRconfig)
    # test funcs_arity_two = 'add,sub,div'
    test_n_funcs = 'add,sub,div'
    test_n_funcs = test_n_funcs.split(',')
    test_n_funcs = pySRURGS.check_validity_suggested_functions(test_n_funcs, 2)
    SRconfig_test_n_funcs = pySRURGS.SymbolicRegressionConfig(
        test_n_funcs, f_funcs, 5, max_permitted_trees)
    for i in tqdm.tqdm(range(0, max_attempts)):
        pySRURGS.uniform_random_global_search_once_to_db(None,
            path_to_db, path_to_csv, SRconfig)
    # test funcs_arity_one = 'tan,exp'
    SRconfig_test_f_funcs = pySRURGS.SymbolicRegressionConfig(
        n_funcs, test_f_funcs, 5, max_permitted_trees)
    for i in tqdm.tqdm(range(0, max_attempts)):
        pySRURGS.uniform_random_global_search_once_to_db(None,
            path_to_db, path_to_csv, SRconfig_test_f_funcs)
    # test max_permitted_trees = 10
    test_max_permitted_trees = 10
    SRconfig_test_permitted_trees = pySRURGS.SymbolicRegressionConfig(
        n_funcs, f_funcs, 5, test_max_permitted_trees)
    for i in tqdm.tqdm(range(0, max_attempts)):
        pySRURGS.uniform_random_global_search_once_to_db(None,
            path_to_db, path_to_csv, SRconfig_test_permitted_trees)
    # plot results
    pySRURGS.plot_results(path_to_db, path_to_csv, SRconfig)
    # generate benchmarks
    pySRURGS.generate_benchmarks()    
    # print DB inspection code
    SR_config = pySRURGS.SymbolicRegressionConfig()
    path_to_csv = './csv/quartic_polynomial.csv'
    path_to_db = './db/quartic_polynomial.db'    
    with SqliteDict(path_to_db, autocommit=True) as results_dict:
        best_result = results_dict['best_result']
    result_list = pySRURGS.get_resultlist(path_to_db)
    dataset = pySRURGS.get_dataset(path_to_csv, SRconfig)
    result_list.sort()
    # after running sort, zero^th element is the best result
    best_result = result_list._results[0]
    print("R^2:", best_result._R2, "Equation:", best_result._simple_equation,
          "Unsimplified Equation:", best_result._equation)
    result_list.print(dataset._y_data)
    # run tests for exhaustive search
    refresh_db(path_to_db)
    SRconfig = pySRURGS.SymbolicRegressionConfig(['add', 'sub'], ['sin'], 1, 3)
    pySRURGS.exhaustive_search(path_to_db, path_to_csv, SRconfig)
    refresh_db(path_to_db)
    SRconfig = pySRURGS.SymbolicRegressionConfig(['add', 'sub'], [], 1, 3)
    pySRURGS.exhaustive_search(
        path_to_db, path_to_csv, SRconfig)
    refresh_db(path_to_db)
    print('Finished run_python_tests')

if __name__ == '__main__':
    test_python_code()
    unittest.main(verbosity=2)
