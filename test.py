'''
    The test script for pySRURGS - runs using both the command line interface 
    and the python interface. Mostly useful to ensure I do not break anything in 
    the course of further software development.
    
    Sohrab Towfighi (C) 2020
'''
import os
import sys
import glob
import pdb
from pySRURGS import * 
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
benchmarks_dir = './csv/benchmarks'
num_iters = 10

def refresh_db(path_to_db):
    try:
        os.remove(path_to_db)
    except:
        pass


class TestCommandLineInterface(unittest.TestCase):
    
    def setUp(self):
        refresh_db(working_db)

    def test_cli_single_processor_deterministic(self):            
        output = sh.python3('pySRURGS.py', '-single', '-deterministic', 
                            '-path_to_db', working_db, qrtic_polynml_csv, 
                            num_iters)
        output = output.strip()
        print(output)
        n_results = count_results(working_db)        
        self.assertGreater(n_results, 0.95*num_iters)
        
    def test_cli_zero_fit_params(self):
        output = sh.python3('pySRURGS.py', '-deterministic', 
                            '-max_num_fit_params', 0, '-path_to_db', working_db, 
                            qrtic_polynml_csv, num_iters)
        output = output.strip()
        print(output)
        result_list = get_resultlist(working_db)
        n_results = count_results(working_db)
        self.assertGreater(n_results, 0.9*num_iters)
        for i in range(0, n_results):
            self.assertEqual(len(result_list._results[i]._params), 0)
        
    def test_cli_five_fit_params(self):
        output = sh.python3('pySRURGS.py', '-deterministic', 
                            '-max_num_fit_params', 5, '-path_to_db', working_db, 
                            qrtic_polynml_csv, num_iters)
        output = output.strip()
        print(output)
        n_results = count_results(working_db)
        self.assertGreater(n_results, 0.98*num_iters)

    def test_cli_funcs_arity_two(self):
        output = output = sh.python3('pySRURGS.py', '-deterministic', 
                                    '-funcs_arity_two', 'add,sub,div', 
                                    '-max_num_fit_params', 1, '-path_to_db', 
                                    working_db, qrtic_polynml_csv, num_iters*10)
        output = output.strip()
        print(output)
        n_results = count_results(working_db)
        result_list = get_resultlist(working_db)
        self.assertGreater(n_results, 0.98*num_iters)
        flag1 = False
        flag2 = False
        flag3 = False
        for i in range(0, n_results):
            my_result = result_list._results[i]
            eqn = my_result._equation
            self.assertEqual(('mul' in eqn), False)
            self.assertEqual(('exp' in eqn), False)
            if 'add' in eqn:
                flag1 = True
            if 'sub' in eqn:
                flag2 = True 
            if 'div' in eqn:
                flag3 = True 
        self.assertEqual(flag1, True)
        self.assertEqual(flag2, True)
        self.assertEqual(flag3, True)        

    def test_cli_funcs_arity_one(self):
        output = sh.python3('pySRURGS.py', '-deterministic', '-funcs_arity_one', 
                            'tan,exp,sinh,cosh', '-funcs_arity_two', 'add',
                            '-max_num_fit_params', 1, '-path_to_db', 
                             working_db, qrtic_polynml_csv, num_iters*20)
        output = output.strip()
        print(output)
        n_results = count_results(working_db)
        result_list = get_resultlist(working_db)
        self.assertGreater(n_results, 0.98*num_iters)
        flag1 = False
        flag2 = False
        flag3 = False
        flag4 = False
        for i in range(0, n_results):
            my_result = result_list._results[i]
            simple_eqn = my_result._simple_equation
            if 'cosh' in simple_eqn:
                flag1 = True
            if 'tan' in simple_eqn:
                flag2 = True
            if 'sinh' in simple_eqn:
                flag3 = True
            if 'exp' in simple_eqn:
                flag4 = True
        self.assertEqual(flag1, True)
        self.assertEqual(flag2, True)
        self.assertEqual(flag3, True)
        self.assertEqual(flag4, True)

    def test_cli_max_permitted_trees(self):
        output = sh.python3('pySRURGS.py', '-deterministic', 
                            '-max_permitted_trees', 10, '-max_num_fit_params', 
                            5, '-path_to_db', working_db, qrtic_polynml_csv, 
                            num_iters)
        output = output.strip()
        print(output)
        n_results = count_results(working_db)
        result_list = get_resultlist(working_db)
        self.assertGreater(n_results, 0.98*num_iters)    

    def test_cli_combo_params_exhaustive_1(self):        
        output = sh.python3('pySRURGS.py', '-exhaustive', '-funcs_arity_two', 
                            'add,sub', '-funcs_arity_one', 'sin', 
                            '-max_permitted_trees', 3, '-max_num_fit_params', 1, 
                            '-path_to_db', working_db, qrtic_polynml_csv, 
                            num_iters)
        output = output.strip()
        print(output)
        self.assertEqual('12.0' in output, True)
        n_results = count_results(working_db)
        result_list = get_resultlist(working_db)
        self.assertEqual(n_results, 10) # 2 trees must simplify

    def test_cli_combo_params_exhaustive_2(self):
        output = sh.python3('pySRURGS.py', '-exhaustive', '-funcs_arity_two', 
                            'add,sub', '-max_permitted_trees', 3, 
                            '-max_num_fit_params', 1, 
                            '-path_to_db', working_db, qrtic_polynml_csv, 10)
        output = output.strip()
        print(output)
        self.assertEqual('42.0' in output, True)
        n_results = count_results(working_db)
        result_list = get_resultlist(working_db)
        self.assertEqual(n_results, 18) # 24 trees must simplify


class TestPython(unittest.TestCase):
    
    def setUp(self):      
        self._defaults = defaults_dict
        self._max_attempts = 15
        self._n_funcs = self._defaults['funcs_arity_two']
        self._n_funcs = self._n_funcs.split(',')
        self._n_funcs = check_validity_suggested_functions(self._n_funcs, 2)
        self._f_funcs = None
        if self._f_funcs is None or self._f_funcs == '':
            self._f_funcs = []
        else:
            self._f_funcs = self._f_funcs.split(',')
            self._f_funcs = check_validity_suggested_functions(self._f_funcs, 1)
        self._max_num_fit_params = self._defaults['max_num_fit_params']
        self._max_permitted_trees = self._defaults['max_permitted_trees']
        self._path_to_db = working_db
        self._path_to_csv = qrtic_polynml_csv            
        self._SRconfig = SymbolicRegressionConfig(self._n_funcs, 
                                                  self._f_funcs, 
                                                  self._max_num_fit_params, 
                                                  self._max_permitted_trees)
        refresh_db(self._path_to_db)



    def test_py_count_equations(self):            
        test_f_funcs = 'tan,exp,cos,sin,log,sinh,cosh,tanh'
        test_f_funcs = test_f_funcs.split(',')
        test_f_funcs = check_validity_suggested_functions(test_f_funcs, 1)
        SRconfig = SymbolicRegressionConfig(self._n_funcs, test_f_funcs, 0, 
                                            self._max_permitted_trees)
        num_equations = count_number_equations(self._path_to_csv, SRconfig)
        self.assertEqual(type(num_equations), mpmath.ctx_mp_python.mpf)

    def test_py_basic_functionality(self):
        for i in range(0, int(self._max_attempts / 10)):
            uniform_random_global_search_once_to_db(None,
                                          self._path_to_db, 
                                          self._path_to_csv, 
                                          self._SRconfig)
        result_list = compile_results(self._path_to_db, 
                                      self._path_to_csv, 
                                      self._SRconfig)
        MSE_1st_run = result_list._results[0]._MSE
        max_attempts2 = self._max_attempts * 10
        for i in tqdm.tqdm(range(0, max_attempts2)):
            uniform_random_global_search_once_to_db(None,
                self._path_to_db, self._path_to_csv, self._SRconfig)
        result_list = compile_results(self._path_to_db, 
                                      self._path_to_csv, 
                                      self._SRconfig)
        MSE_2nd_run = result_list._results[0]._MSE
        self.assertLess(MSE_2nd_run, MSE_1st_run)
        
    def test_py_max_fit_params_zero(self):
        # test max_num_fit_params 0
        SRconfig = SymbolicRegressionConfig(self._n_funcs, self._f_funcs, 0, 
                                            self._max_permitted_trees)
        for i in tqdm.tqdm(range(0, self._max_attempts)):
            uniform_random_global_search_once_to_db(None, 
                                                    self._path_to_db, 
                                                    self._path_to_csv, 
                                                    SRconfig)
        result_list = get_resultlist(working_db)
        n_results = count_results(working_db)
        for i in range(0, n_results):
            self.assertEqual(len(result_list._results[i]._params), 0)

    def test_py_max_fit_params_five(self):
        SRconfig = SymbolicRegressionConfig(self._n_funcs, self._f_funcs, 5, 
                                            self._max_permitted_trees)
        for i in tqdm.tqdm(range(0, self._max_attempts)):
            uniform_random_global_search_once_to_db(None,
                self._path_to_db, self._path_to_csv, SRconfig)
        result_list = get_resultlist(working_db)
        n_results = count_results(working_db)            
        for i in range(0, n_results):
            self.assertLess(len(result_list._results[i]._params), 6)


    def test_py_funcs_arity_two(self):
        test_n_funcs = 'add,sub,div'
        test_n_funcs = test_n_funcs.split(',')
        test_n_funcs = check_validity_suggested_functions(test_n_funcs, 2)
        SRconfig = SymbolicRegressionConfig(test_n_funcs, self._f_funcs, 5, 
                                            self._max_permitted_trees)
        for i in tqdm.tqdm(range(0, self._max_attempts)):
            uniform_random_global_search_once_to_db(None, self._path_to_db, 
                                                    self._path_to_csv, 
                                                    SRconfig)
        flag1 = False
        flag2 = False 
        flag3 = False
        result_list = get_resultlist(working_db)
        n_results = count_results(working_db)            
        for i in range(0, n_results):
            my_result = result_list._results[i]
            eqn = my_result._equation
            self.assertEqual(('mul' in eqn), False)
            self.assertEqual(('exp' in eqn), False)
            if 'add' in eqn:
                flag1 = True
            if 'sub' in eqn:
                flag2 = True 
            if 'div' in eqn:
                flag3 = True 
        self.assertEqual(flag1, True)
        self.assertEqual(flag2, True)
        self.assertEqual(flag3, True)
            
    def test_py_funcs_arity_one(self):
        test_n_funcs = []
        test_f_funcs = 'tan,exp,sinh,cosh'
        test_f_funcs = test_f_funcs.split(',')
        test_f_funcs = check_validity_suggested_functions(test_f_funcs, 1)        
        SRconfig = SymbolicRegressionConfig(test_n_funcs, test_f_funcs, 5, 
                                            self._max_permitted_trees)
        for i in tqdm.tqdm(range(0, self._max_attempts)):
            uniform_random_global_search_once_to_db(None, self._path_to_db, 
                                                    self._path_to_csv, 
                                                    SRconfig)
        flag1 = False 
        flag2 = False 
        flag3 = True
        result_list = get_resultlist(working_db)
        n_results = count_results(working_db)            
        for i in range(0, n_results):
            my_result = result_list._results[i]
            eqn = my_result._equation
            self.assertEqual(('add' in eqn), False)
            self.assertEqual(('mul' in eqn), False)
            if 'tan' in eqn:
                flag1 = True
            if 'exp' in eqn:
                flag2 = True 
            if 'add' in eqn:
                flag3 = False 
        self.assertEqual(flag1, True)
        self.assertEqual(flag2, True)
        self.assertEqual(flag3, True)        

    def test_py_db_inspection_code(self):
        # DB inspection code
        import pySRURGS
        from result_class import Result # Result needs to be in the namespace.
        from sqlitedict import SqliteDict
        SRconfig = pySRURGS.SymbolicRegressionConfig()
        path_to_csv = './csv/quartic_polynomial.csv'
        path_to_db = './db/quartic_polynomial.db'
        refresh_db(path_to_db)
        
        # create test db        
        for i in tqdm.tqdm(range(0, self._max_attempts)):
            uniform_random_global_search_once_to_db(None, 
                                                    path_to_db, 
                                                    path_to_csv, 
                                                    self._SRconfig)

        result_list = pySRURGS.get_resultlist(path_to_db)
        result_list.sort()
        # after running sort, zero^th element is the best result
        best_result = result_list._results[0]
        print("R^2:", best_result._R2, 
              "Equation:", best_result._simple_equation, 
              "Unsimplified Equation:", best_result._equation)
        # will raise exception if unable to cast R2 as float
        float(best_result._R2)        
        # plot results
        plot_results(path_to_db, path_to_csv, SRconfig)
        self.assertEqual(os.path.isfile('./image/plot.png'), True)

    def test_py_combo_params_exhaustive_1(self):
        test_n_funcs = 'add,sub'
        test_n_funcs = test_n_funcs.split(',')
        test_n_funcs = check_validity_suggested_functions(test_n_funcs, 2)
        test_f_funcs = 'sin'
        test_f_funcs = test_f_funcs.split(',')
        test_f_funcs = check_validity_suggested_functions(test_f_funcs, 1)
        test_max_num_fit_params = 1
        test_max_num_trees = 3
        SRconfig = SymbolicRegressionConfig(test_n_funcs, test_f_funcs, 
                                            test_max_num_fit_params, 
                                            test_max_num_trees)
        exhaustive_search(self._path_to_db, self._path_to_csv, SRconfig)
        n_results = count_results(working_db)
        result_list = get_resultlist(working_db)
        self.assertEqual(n_results, 10)

    def test_py_combo_params_exhaustive_2(self):
        test_n_funcs = 'add,sub'
        test_n_funcs = test_n_funcs.split(',')
        test_n_funcs = check_validity_suggested_functions(test_n_funcs, 2)
        test_max_num_fit_params = 1
        test_max_num_trees = 3
        SRconfig = SymbolicRegressionConfig(test_n_funcs, self._f_funcs, 
                                            test_max_num_fit_params, 
                                            test_max_num_trees)
        exhaustive_search(self._path_to_db, self._path_to_csv, SRconfig)
        n_results = count_results(working_db)
        result_list = get_resultlist(working_db)
        self.assertEqual(n_results, 18)

    def test_py_benchmarks(self):
        benchmarks = glob.glob(os.path.join(benchmarks_dir, '*'))
        for benchmark_file in benchmarks:
            os.remove(benchmark_file)
        generate_benchmarks()
        for benchmark_file in benchmarks:
            self.assertEqual(os.path.isfile(benchmark_file), True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
