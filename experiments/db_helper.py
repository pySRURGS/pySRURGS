from sqlitedict import SqliteDict
import pdb
import sys 
sys.path.append('./../')
import pySRURGS  

if __name__ == '__main__':
    args = sys.argv[1:]
    path_to_db = args[0]
    n = pySRURGS.assign_n_evals(path_to_db)
    print(n)