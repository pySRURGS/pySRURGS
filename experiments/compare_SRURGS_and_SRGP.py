# statistical comparison of results 
import scipy 
import parmap
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
images_dir = 'C:/Users/sohra/Google Drive (fischerproject2018@gmail.com)/pySRURGS/experiments/figures'
# go to srurgs dir 
# read file, find matching file in srgp dir 
# add to dictionary 
# add the arrays

SR_config1, SR_config2 = pySRURGS.generate_benchmarks_SRconfigs()

def give_algo_files(algo='SRGP'):
    if algo == 'SRGP':
        return glob.glob(srgp_dir+'*SRGP*')

def give_srurgs_files():
    return glob.glob(srurgs_dir+'*SRURGS*')

def compare_SRURGS_with_algo(problem_number_given=None, 
                             algo_name='SRGP', 
                             benchmark_start_num=0, 
                             benchmark_end_num=100):
    a = []
    b = []
    algo_files = give_algo_files(algo_name)
    srurgs_files = give_srurgs_files()
    for path_to_srurgs_db in srurgs_files:        
        problem_ID_srurgs = os.path.basename(path_to_srurgs_db)
        problem_ID_algo = problem_ID_srurgs.replace("SRURGS", algo_name)
        problem_number = int(problem_ID_srurgs.split('_')[0])
        # if we set problem_number, only get data for that problem and add to db 
        if problem_number_given is not None:
            if problem_number != problem_number_given:
                continue
        # only consider problems within our domain 
        if problem_number < benchmark_start_num or problem_number >= benchmark_end_num:
            continue
        for file in algo_files:
            if problem_ID_algo == os.path.basename(file):
                path_to_algo_db = file
        path_to_algo_db # here to raise error if we do not find path_to_algo_db        
        print(path_to_srurgs_db)
        print(path_to_algo_db)
        with SqliteDict(path_to_stats_db, autocommit=True) as results_dict:
            try:        
                result_algo, result_SRURGS = results_dict[path_to_srurgs_db]
                best_algo_R2 = result_algo._R2
                best_SRURGS_R2 = result_SRURGS._R2
            except KeyError:            
                path_to_csv = 'C:/Users/sohra/Google Drive (fischerproject2018@gmail.com)/pySRURGS/csvs/benchmarks/' + str(problem_number)+'_train.csv'
                if problem_number < 20:
                    SR_config = SR_config1 
                else:
                    SR_config = SR_config2
                SRURGS_result_list, dataset = pySRURGS.get_resultlist(path_to_srurgs_db, path_to_csv, SR_config)
                SRURGS_result_list.sort()
                algo_result_list, dataset = pySRURGS.get_resultlist(path_to_algo_db, path_to_csv, SR_config)
                algo_result_list.sort()
                pdb.set_trace()
                best_algo = algo_result_list._results[0]
                best_SRURGS = SRURGS_result_list._results[0]
                best_algo_R2 = best_algo._R2
                best_SRURGS_R2 = best_SRURGS._R2
                results_dict[path_to_srurgs_db] = (best_algo, best_SRURGS)
        if best_algo_R2 > 1 or best_algo_R2 < 0:
            best_algo_R2 = 0
        if best_SRURGS_R2 > 1 or best_SRURGS_R2 < 0:
            best_SRURGS_R2 = 0
        a.append(best_algo_R2)
        b.append(best_SRURGS_R2)
    a = np.array(a)
    b = np.array(b)
    a = np.nan_to_num(a)
    b = np.nan_to_num(b)
    if problem_number_given is None:
        stat, p_val = ttest_rel(a, b)
        print(p_val)
        if p_val < 0.05:
            print("Significant difference")
        print("AVG SRGP R2", np.average(a))
        print("AVG SRURGS R2", np.average(b))
        print("MEDIAN SRGP R2", np.median(a))
        print("MEDIAN SRURGS R2", np.median(b))
        data = [a,b]
        plt.figure(figsize=(3.14, 2))        
        plt.hist(data, bins=(np.arange(12)-0.5)/10, label=[algo_name,'SRURGS'])
        plt.xlabel("R^2")
        plt.legend()
        plt.savefig(images_dir+'/histo_'+algo_name+str(benchmark_start_num)+'_'+str(benchmark_end_num)+'.eps')        
        medianprops = dict(linewidth=4, color='firebrick')
        plt.figure(figsize=(3.14, 2))
        plt.boxplot(data, labels=[algo_name, 'SRURGS'], medianprops=medianprops)
        plt.ylabel('R^2')
        plt.savefig(images_dir+'/boxplot_'+algo_name+str(benchmark_start_num)+'_'+str(benchmark_end_num)+'.eps')
        
    else:
        return
    
if __name__ == '__main__':
    #os.remove(path_to_stats_db)
    #results = parmap.map(compare_SRURGS_with_algo, list(range(0,20)), 
    #                     'SRGP', 0, 20, pm_pbar=True)
    compare_SRURGS_with_algo(problem_number_given=None, 
                             algo_name='SRGP', 
                             benchmark_start_num=0, 
                             benchmark_end_num=20)    
    #results = parmap.map(compare_SRURGS_with_algo, list(range(20,100)), 
    #                     'SRGP', 20, 100, pm_pbar=True)
    #compare_SRURGS_with_algo(problem_number_given=None, 
    #                         algo_name='SRGP', 
    #                         benchmark_start_num=20, 
    #                         benchmark_end_num=100)