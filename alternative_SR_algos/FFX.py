import numpy as np
import ffx
import sys
sys.path.append('./..')
from pySRURGS import Dataset
import argparse
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train", help="absolute or relative file path to the csv file housing the training data")
    parser.add_argument("pickle", help="absolute or relative file path to the pickle file where we save result")
    arguments = parser.parse_args()

    path_to_csv = arguments.train
    picklefile = arguments.pickle
    dataset = Dataset(path_to_csv, 0):
    FFX = ffx.FFXRegressor()
    FFX.fit(dataset._x_data, dataset._y_data)

    with open(picklefile, 'wb') as handle:
        pickle.dump(FFX, handle, protocol=pickle.HIGHEST_PROTOCOL)
