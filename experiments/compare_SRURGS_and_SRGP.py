# statistical comparison of results 
import scipy 
from scipy.stats import ttest_rel
import os 
import glob 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sqlitedict
import sys 
sys.path.append('./..')
import pySRURGS
from pySRURGS import Result
from sqlitedict import SqliteDict
import pdb
import numpy as np

srurgs_dir = 'C:/Users/sohra/srurgs_data/'
srgp_dir = 'C:/Users/sohra/Dropbox/Apps/pySRURGS/'
path_to_stats_db = './stats.db'
# go to srurgs dir 
# read file, find matching file in srgp dir 
# add to dictionary 
# add the arrays

SR_config1, SR_config2 = pySRURGS.generate_benchmarks_SRconfigs()

srgp_files = glob.glob(srgp_dir+'*SRGP*')
srurgs_files = glob.glob(srurgs_dir+'*SRURGS*')
a = []
b = []

for srurgs_result in srurgs_files:
    path_to_srurgs_db = srurgs_result    
    with SqliteDict(path_to_stats_db, autocommit=True) as results_dict: 
        try:
            MSE_SRGP, MSE_SRURGS = results_dict[path_to_srurgs_db]
        except KeyError:
            print(path_to_srurgs_db)
            problem_name_srurgs = os.path.basename(srurgs_result)
            problem_name_srgp = problem_name_srurgs.replace("SRURGS", "SRGP")
            for file in srgp_files:
                if problem_name_srgp == os.path.basename(file):
                    path_to_srgp_db = file
            print(path_to_srgp_db)
            problem_number = int(problem_name_srurgs.split('_')[0])
            path_to_csv = 'C:/Users/sohra/Google Drive (fischerproject2018@gmail.com)/pySRURGS/csvs/benchmarks/' + str(problem_number)+'_train.csv'
            if problem_number < 20:
                SR_config = SR_config1 
            else:
                SR_config = SR_config2
            SRURGS_result_list, dataset = pySRURGS.get_resultlist(path_to_srurgs_db, path_to_csv, SR_config)
            SRURGS_result_list.sort()
            SRGP_result_list, dataset = pySRURGS.get_resultlist(path_to_srgp_db, path_to_csv, SR_config)
            SRGP_result_list.sort()
            MSE_SRGP = SRGP_result_list._results[0]._MSE
            MSE_SRURGS = SRURGS_result_list._results[0]._MSE
            results_dict[path_to_srurgs_db] = (MSE_SRGP, MSE_SRURGS)
    a.append(MSE_SRGP)
    b.append(MSE_SRURGS)
stat, p_val = ttest_rel(a, b)
print(p_val)
if p_val < 0.05:
    print("Significant difference")
print("AVG SRGP MSE", np.average(a))
print("AVG SRURGS MSE", np.average(b))
print("MEDIAN SRGP MSE", np.median(a))
print("MEDIAN SRURGS MSE", np.median(b))
data = [a,b]    
plt.boxplot(data)

