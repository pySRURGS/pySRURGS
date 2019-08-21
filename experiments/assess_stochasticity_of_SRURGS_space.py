import sys
sys.path.append("./..")
import random
import matplotlib
matplotlib.use('TkAgg')
import pySRURGS
import pdb
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

def main(max_permitted_trees, max_num_fit_params, i, r0, s0):
    SR_config = pySRURGS.SymbolicRegressionConfig(['add','sub','mul','div','pow'],
                                        [],
                                        max_num_fit_params=max_num_fit_params,
                                        max_permitted_trees=max_permitted_trees)
    en = pySRURGS.Enumerator2()
    csv_path = './../csvs/quartic_polynomial.csv'  
    dataset = pySRURGS.Dataset(csv_path, SR_config._max_num_fit_params)
    params = pySRURGS.create_fitting_parameters(dataset._int_max_params)
    n = len(SR_config._n_functions)
    A = en.get_A(n, i)
    B = en.get_B(dataset._m_terminals, i)
    rows = 125
    cols = 81
    data = np.ones((rows,cols))
    print(A)
    print(B)
    x_data = []
    y_data = []
    for r in range(r0, r0+rows):
        print(r, A)        
        for s in range(s0, s0+cols):           
            try:
                eqn = pySRURGS.equation_generator2(i, r, s, dataset, en, SR_config, simpler=False)
            except:
                pdb.set_trace()
            eqn = pySRURGS.clean_funcstring(eqn)
            try:
                (sum_of_squared_residuals, 
                sum_of_squared_totals, 
                R2,
                params_dict_to_store, 
                residual) = pySRURGS.check_goodness_of_fit(eqn, params, dataset)
            except FloatingPointError:
                pass            
            if R2 > 1 or R2 < 0:
                R2 = 0
            data[r-r0,s-s0] = R2
            x_data.append(r)
            y_data.append(s)
        


    fig = plt.figure()
    ax = fig.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    scat = ax.scatter(x_data, y_data, c=data.flatten())
    cbar = plt.colorbar(scat, ax=ax, boundaries=np.linspace(0,1,100))
    cbar.set_label('R^2')
    plt.xlabel("r")
    plt.ylabel("s")
    plt.tight_layout()
    plt.show()
    plt.clf()

if __name__ == '__main__':
    main(10, 2, 3, 0, 0)
    main(100, 2, 50, 1000, 1000)
'''
x_data = list(range(0,A))
y_data = list(range(0,B))
plt.contour(y_data, x_data, data, cmap='RdGy')
cbar = plt.colorbar()
cbar.set_label('R^2')
ax = fig.gca()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()'''