import sys
sys.path.append("./..")
import random
import matplotlib
matplotlib.use('TkAgg')
import pySRURGS
import pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sqlitedict import SqliteDict
from matplotlib.ticker import MaxNLocator
import os
import seaborn as sns

path_to_db = './stochasticity.db'

def main(max_permitted_trees, max_num_fit_params, i, r0, s0, figpath):
    SR_config = pySRURGS.SymbolicRegressionConfig(['add','sub','mul','div','pow'],
                                        [],
                                        max_num_fit_params=max_num_fit_params,
                                        max_permitted_trees=max_permitted_trees)
    en = pySRURGS.Enumerator2()
    csv_path = './../csv/quartic_polynomial.csv'  
    dataset = pySRURGS.Dataset(csv_path, SR_config._max_num_fit_params)
    params = pySRURGS.create_fitting_parameters(dataset._int_max_params)
    n = len(SR_config._n_functions)
    A = en.get_A(n, i)
    B = en.get_B(dataset._m_terminals, i)
    rows = 125
    cols = 81
    data = np.ones((rows,cols))
    x_data = np.ones((rows,cols))
    y_data = np.ones((rows,cols))
    print(A)
    print(B)
    with SqliteDict(path_to_db, autocommit=True) as results_dict:
        for r in range(r0, r0+rows):
            print(r, A)
            for s in range(s0, s0+cols):            
                try:
                    R2 = results_dict[str(i)+str(r)+str(s)]
                except KeyError:                               
                    try:
                        eqn = pySRURGS.equation_generator2(i, r, s, dataset, en, SR_config, simpler=False) 
                        eqn = pySRURGS.clean_funcstring(eqn) 
                        (sum_of_squared_residuals, 
                        sum_of_squared_totals, 
                        R2,
                        params_dict_to_store, 
                        residual) = pySRURGS.check_goodness_of_fit(eqn, params, dataset)
                    except FloatingPointError:
                        pass            
                    results_dict[str(i)+str(r)+str(s)] = R2
                if R2 > 1 or R2 < 0:
                    R2 = 0
                data[r-r0,s-s0] = R2
                x_data[r-r0,s-s0] = r
                y_data[r-r0,s-s0] = s
    fig = plt.figure(figsize=(3.14, 2))
    ax = sns.heatmap(data, vmin=0, vmax=1, linewidth=0.0, cmap=matplotlib.cm.viridis, yticklabels=False, xticklabels=False, cbar_kws={'label':'R^2', 'ticks':[0, 0.5, 1]})
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("r")
    plt.ylabel("s")
    plt.tight_layout()
    plt.show()
    plt.savefig(figpath, bbox_inches='tight')    

if __name__ == '__main__':    
    figpath = 'C:/Users/sohra/Google Drive (fischerproject2018@gmail.com)/pySRURGS/experiments/figures/Fig2.png'
    main(10, 2, 3, 0, 0, figpath)
    
