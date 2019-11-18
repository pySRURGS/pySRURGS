import os 
import sys
import time
import pySRURGS
num_iters_in_test = 100
n_funcs = ['add','sub','mul','div']
f_funcs = []
n_par = 2
benchmark_name = str(1)
for n_tree in [10, 100, 1000]:
    SR_config = pySRURGS.SymbolicRegressionConfig(n_funcs, f_funcs, n_par, n_tree)
    init_time = time.time()
    for i in range(0, num_iters_in_test):
        pySRURGS.generate_benchmark(benchmark_name, SR_config)
    end_time = time.time()
    diff_time = end_time - init_time 
    print("n_tree", n_tree, "time", diff_time)
