# this script needs to be run in the same directory as pySRURGS.py
import os 
import sys
import time
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
    
def refresh_db(path_to_db):
    try:
        os.remove(path_to_db)
    except:
        pass

num_iters_in_test = 100
n_funcs = ['add','sub','mul','div']
f_funcs = []
n_par = 2
benchmark_name = str(1)
qrtic_polynml_csv = './csv/quartic_polynomial.csv'
qrtic_polynml_db = './db/quartic_polynomial.db'
'''
for n_tree in [10, 100, 1000, 10000]:
    SR_config = pySRURGS.SymbolicRegressionConfig(n_funcs, f_funcs, n_par, n_tree)
    init_time = time.time()
    for i in range(0, num_iters_in_test):
        pySRURGS.generate_benchmark(benchmark_name, SR_config)
    end_time = time.time()
    diff_time = end_time - init_time 
    print("n_tree", n_tree, "time", diff_time)
'''
for n_tree in [10, 100, 1000, 10000]:
    refresh_db(qrtic_polynml_db)
    SR_config = pySRURGS.SymbolicRegressionConfig(n_funcs, f_funcs, n_par, n_tree)
    init_time = time.time()
    for i in range(0, num_iters_in_test):
        pySRURGS.uniform_random_global_search_once(qrtic_polynml_db, qrtic_polynml_csv, SR_config)
    end_time = time.time()
    diff_time = end_time - init_time 
    print("n_tree", n_tree, "time", diff_time)
    