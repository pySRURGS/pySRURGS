import sys
sys.path.append("./..")
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')
import pySRURGS
import pdb
import numpy as np

SR_config = pySRURGS.SymbolicRegressionConfig(['add','sub','mul','div', 'pow'],
                                    [],
                                    max_num_fit_params=5,
                                    max_permitted_trees=1000)

en = pySRURGS.Enumerator2()

for i in range(1,6):
    csv_path = './../csvs/quartic_polynomial.csv'  
    dataset = pySRURGS.Dataset(csv_path, 5)
    params = pySRURGS.create_fitting_parameters(dataset._int_max_params)
    i = 902
    r0 = 1000
    s0 = 1000
    n = SR_config._max_permitted_trees
    A = en.get_A(n, i)
    B = en.get_B(n, i)    
    for r in range(r0, A):
        x_data = []
        plt_data = []
        for s in range(s0, 10000):#B):
            print(s)
            eqn = pySRURGS.equation_generator2(i, r, s, dataset, en, 
                                               SR_config, simpler=False)
            eqn = pySRURGS.clean_funcstring(eqn)
            try:
                (sum_of_squared_residuals, 
                sum_of_squared_totals, 
                R2,
                params_dict_to_store, 
                residual) = pySRURGS.check_goodness_of_fit(eqn, params, dataset)
            except FloatingPointError:
                pass
            plt_data.append(sum_of_squared_residuals)
            x_data.append(s)
        
        plt.plot(np.array(x_data), np.array(plt_data), 'go')
        plt.show()