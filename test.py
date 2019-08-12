import sh
import os
import sys 

sh.python('pySRURGS.py', './csvs/quartic_polynomial.csv', 10)
sh.python('pySRURGS.py', '-single', './csvs/quartic_polynomial.csv', 10)
sh.python('pySRURGS.py', '-count', './csvs/quartic_polynomial.csv', 10)
sh.python('pySRURGS.py', '-benchmarks', './csvs/quartic_polynomial.csv', 10)
sh.python('pySRURGS.py', '-max_num_fit_params', 0, './csvs/quartic_polynomial.csv', 10)
sh.python('pySRURGS.py', '-max_num_fit_params', 5, './csvs/quartic_polynomial.csv', 10)
sh.python('pySRURGS.py', '-funcs_arity_two', 'add,sub,div', '-max_num_fit_params', 5, './csvs/quartic_polynomial.csv', 10)
sh.python('pySRURGS.py', '-funcs_arity_one', 'tan,exp', '-max_num_fit_params', 5, './csvs/quartic_polynomial.csv', 10)
sh.python('pySRURGS.py', '-max_permitted_trees', 10, '-max_num_fit_params', 5, './csvs/quartic_polynomial.csv', 10)

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
