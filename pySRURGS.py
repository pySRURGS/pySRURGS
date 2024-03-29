#!/usr/bin/env python

'''
pySRURGS - Symbolic Regression by Uniform Random Global Search
Sohrab Towfighi (C) 2019-2020
License: GPL 3.0
https://github.com/pySRURGS/pySRURGS
'''

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style
import multiprocessing as mp
from itertools import repeat
import collections
from sqlitedict import SqliteDict
import tabulate
import datetime
import sympy
from sympy import simplify, sympify, Symbol
import mpmath
import sys
import lmfit
import csv
import time
import pdb
import re
import os
import tqdm
import itertools
import parmap
import pandas
import argparse
import numpy as np
from result_class import Result
from math_funcs import (sympy_Sub, sympy_Div, sin, cos, tan, exp, log, sinh, 
                        cosh, tanh, sum, add, sub, mul, div, pow)
np.seterr(all='raise')
matplotlib.style.use('seaborn-colorblind')


''' GLOBALS '''
BIG_NUM = 1.79769313e+300
NUM_ITERS_LIMIT = (2**32-1)
ERASE_LINE = '\x1b[2K' # erase line command
fitting_param_prefix = 'begin_fitting_param_'
fitting_param_suffix = '_end_fitting_param'
variable_prefix = 'begin_variable_'
variable_suffix = '_end_variable'
path_to_toy_csv = './csv/toy_data_for_benchmark_gen.csv'
benchmarks_x_domain = [0, 10]
benchmarks_fit_param_domain = [-10, 10]
benchmarks_dir = './csv/benchmarks'
benchmarks_summary_tsv = './benchmarks_summary.tsv'
memoize_funcs = False
randgen = np.random.RandomState()
defaults_dict = {'funcs_arity_one': None,
                 'funcs_arity_two': 'add,sub,mul,div,pow',
                 'max_num_fit_params': 3,
                 'max_permitted_trees': 1000,
                 'path_to_db': None,
                 'path_to_weights': None}
''' END GLOBALS '''
                 

def check_validity_suggested_functions(suggested_funcs, arity):
    '''
    Takes a list of suggested functions to use in the search space and checks
    that they are valid.

    Parameters
    ----------
    suggested_funcs: list
        A list of strings.
        In case of `arity==1`, permitted values are ['sin','cos','tan','exp',
                                                     'log','tanh','sinh','cosh',
                                                     None]
        In case of `arity==2`, permitted values are ['add','sub','mul','div',
                                                     'pow', None]

    Returns
    -------
    suggested_funcs: list

    Raises
    ------
    Exception, if any of the suggested funcs is not in the permitted list
    '''
    valid_funcs_arity_1 = ['sin', 'cos', 'tan', 'exp', 'log', 'tanh', 'sinh', 
                           'cosh', None]
    valid_funcs_arity_2 = ['add', 'sub', 'mul', 'div', 'pow', None]
    if arity == 1:
        if suggested_funcs != [',']:
            for func in suggested_funcs:
                if func not in valid_funcs_arity_1:
                    msg = "Your suggested function of arity 1: " + func
                    msg += " is not in the list of valid functions"
                    msg += " " + str(valid_funcs_arity_1)
                    raise Exception(msg)
        else:
            suggested_funcs = []
    elif arity == 2:
        for func in suggested_funcs:
            if func not in valid_funcs_arity_2:
                msg = "Your suggested function of arity 2: " + func
                msg += " is not in the list of valid functions"
                msg += " " + str(valid_funcs_arity_2)
                raise Exception(msg)
    return suggested_funcs
                 
if defaults_dict['funcs_arity_two'] is None:
    default_n_funcs = []
else:
    default_n_funcs = check_validity_suggested_functions(
                                 defaults_dict['funcs_arity_two'].split(','), 2)
if defaults_dict['funcs_arity_one'] is None:
    default_f_funcs = []
else:
    default_f_funcs = check_validity_suggested_functions(
                                 defaults_dict['funcs_arity_one'].split(','), 1)


def is_csv_valid(filepath, check_header=False):
    try:
        with open(filepath, 'r') as csv_file:
            dialect = csv.Sniffer().sniff(csv_file.read(2048))
    except Exception as e:
        print("Error encountering while reading: ", filepath)
        print(e)
        exit(2)
    if check_header == True:        
        with open(filepath, 'r') as csv_file:
            sniffer = csv.Sniffer()
            has_header = sniffer.has_header(csv_file.read(2048))
        if has_header == False:
            print("File which must have header is missing header: ", filepath)
            exit(2)


class Dataset(object):
    """
    An object used to store the dataset of this symbolic regression
    problem.

    Parameters
    ----------
    path_to_csv_file: string
       Absolute or relative path to the CSV file for the numerical data. The
       rightmost column of the CSV file should be the dependent variable.
       The CSV file should have a header of column names and should NOT
       have a leftmost index column.

    int_max_params: int
        The maximum number of fitting parameters specified in the symbolic
        regression problem. Same as `max_num_fit_params`.

    path_to_weights: string 
        An absolute or relative path to the CSV for weights of the data points 
        in the CSV found in `path_to_csv`. If `None`, will assume all data 
        points are equally weighted.

    Returns
    -------
    self
        A pySRURGS.Dataset object, which houses a variety of attributes 
        including the numerical data, the sympy namespace, the data dict used in 
        evaluating the equation string, etc.
    """

    def __init__(self, 
                 path_to_csv_file, 
                 int_max_params=defaults_dict['max_num_fit_params'], 
                 path_to_weights=None):
        (dataframe, header_labels) = self.load_csv_data(path_to_csv_file)
        if path_to_weights is not None:
            (weights_df, empty_labels) = self.load_csv_data(path_to_weights, 
                                                            header=None)            
            self._data_weights = np.squeeze(weights_df.values)
        else: 
            self._data_weights = None
        self._int_max_params = int_max_params
        self._dataframe = dataframe
        self._header_labels = header_labels
        x_data, x_labels, x_properties = self.get_independent_data()
        y_data, y_label, y_properties = self.get_dependent_data()
        self._x_data = x_data
        self._x_labels = x_labels
        self._y_data = y_data
        self._y_label = y_label        
        if np.std(self._y_data) == 0:
            raise Exception("The data is invalid. All y values are the same.")
        self._param_names = [make_parameter_name(x) for x in
                             range(0, self._int_max_params)]
        self._data_properties = dict()
        self._data_properties.update(x_properties)
        self._data_properties.update(y_properties)
        self._data_dict = self.get_data_dict()
        self._num_variables = len(self._x_labels)
        self._m_terminals = self._num_variables + int_max_params
        self._terminals_list = (create_parameter_list(int_max_params) +
                                create_variable_list(path_to_csv_file))        

    def make_sympy_namespace(self):
        sympy_namespace = {}
        for variable_name in self._x_labels:
            sympy_namespace[variable_name] = sympy.Symbol(variable_name)
        for param_name in self._param_names:
            sympy_namespace[param_name] = sympy.Symbol(param_name)
        sympy_namespace['add'] = sympy.Add
        sympy_namespace['sub'] = sympy_Sub
        sympy_namespace['mul'] = sympy.Mul
        sympy_namespace['div'] = sympy_Div
        sympy_namespace['pow'] = sympy.Pow
        sympy_namespace['cos'] = sympy.Function('cos')
        sympy_namespace['sin'] = sympy.Function('sin')
        sympy_namespace['tan'] = sympy.Function('tan')
        sympy_namespace['cosh'] = sympy.Function('cosh')
        sympy_namespace['sinh'] = sympy.Function('sinh')
        sympy_namespace['tanh'] = sympy.Function('tanh')
        sympy_namespace['exp'] = sympy.Function('exp')
        sympy_namespace['log'] = sympy.Function('log')
        return sympy_namespace

    def load_csv_data(self, path_to_csv, header=True):
        if header is True:
            dataframe = pandas.read_csv(path_to_csv)
        else:
            dataframe = pandas.read_csv(path_to_csv, header=header)
        column_labels = dataframe.keys()
        return (dataframe, column_labels)

    def get_independent_data(self):
        '''
            Loads all data in self._dataframe except the rightmost column
        '''
        dataframe = self._dataframe
        header_labels = self._header_labels
        features = dataframe.iloc[:, :-1]
        features = np.array(features)
        labels = header_labels[:-1]
        properties = dict()
        for label in labels:
            properties.update(get_properties(dataframe[label], label))
        return (features, labels, properties)

    def get_dependent_data(self):
        '''
            Loads only the rightmost column from self._dataframe
        '''
        dataframe = self._dataframe
        header_labels = self._header_labels
        feature = dataframe.iloc[:, -1]
        feature = np.array(feature)
        label = header_labels[-1]
        properties = get_properties(dataframe[label], label)
        return (feature, label, properties)

    def get_data_dict(self):
        '''
            Creates a dictionary object which houses the values in the dataset 
            CSV. The variable names in the CSV become keys in this data_dict 
            dictionary.
        '''
        dataframe = self._dataframe
        data_dict = dict()
        for label in self._header_labels:
            data_dict[label] = np.array(dataframe[label].values).astype(float)
            check_for_nans(data_dict[label])
        return data_dict


class SymbolicRegressionConfig(object):
    """
    An object used to store the configuration of this symbolic regression
    problem.

    Parameters
    ----------

    path_to_csv: string
        An absolute or relative path to the dataset CSV file. Usually, this
        file ends in a '.csv' extension.

    path_to_db: string
        An absolute or relative path to where the code can save an output
        database file. Usually, this file ends in a '.db' extension.

    n_functions: list
       A list with elements from the set ['add','sub','mul','div','pow'].
       Defines the functions of arity two that are permitted in this symbolic
       regression run. Default: ['add','sub','mul','div', 'pow']

    f_functions: list
        A list with elements from the set ['cos','sin','tan','cosh','sinh',
        'tanh','exp','log']. Defines the functions of arity one that are
        permitted in this symbolic regression run.
        Default: []

    max_num_fit_params: int
        This specifies the length of the fitting parameters vector. Randomly
        generated equations can have up to `max_num_fit_params` independent
        fitting parameters. Default: 3

    max_permitted_trees: int
        This specifies the number of permitted unique binary trees, which
        determine the structure of random equations. pySRURGS will consider
        equations from [0 ... max_permitted_trees] during its search. Increasing
        this value increases the size of the search space. Default: 100

    path_to_weights: string 
        An absolute or relative path to the CSV for weights of the data points 
        in the CSV found in `path_to_csv`. If `None`, will assume all data 
        points are equally weighted.           
    
    Attributes
    ----------
    
    Most are simply the parameters which were passed in. Notably, there is the 
    dataset object, which is not a mere parameter.
    
    self._dataset
        A pySRURGS.Dataset object, which houses a variety of attributes 
        including the numerical data, the sympy namespace, the data dict used in 
        evaluating the equation string, etc.
    
    Returns
    -------
    self
        A pySRURGS.SymbolicRegressionConfig object, with attributes 
        self._path_to_csv, 
        self._path_to_db,
        self._n_functions, 
        self._f_functions, 
        self._max_num_fit_params, 
        self._max_permitted_trees,  
        self._path_to_weights, and 
        self._dataset.
    """

    def __init__(self,
                 path_to_csv,
                 path_to_db,
                 n_functions=default_n_funcs,
                 f_functions=default_f_funcs,
                 max_num_fit_params=defaults_dict['max_num_fit_params'],
                 max_permitted_trees=defaults_dict['max_permitted_trees'],
                 path_to_weights=None):  
        if path_to_db is None:
            path_to_db = create_db_name(path_to_csv)
        self._n_functions = n_functions
        self._f_functions = f_functions
        self._max_num_fit_params = max_num_fit_params
        self._max_permitted_trees = max_permitted_trees        
        self._path_to_csv = path_to_csv
        self._path_to_db = path_to_db
        is_csv_valid(path_to_csv, True)
        self._path_to_weights = path_to_weights
        if path_to_weights is not None:
            is_csv_valid(path_to_weights)
        self._dataset = Dataset(path_to_csv, 
                                max_num_fit_params, 
                                path_to_weights)


def memoize(func):
    """
    A memoize function that wraps other functions. Improves CPU performance at
    the cost of increased memory requirements.

    Parameters
    ----------
    func : function
        Will memoize the `func` provided `func` is deterministic in its outputs.

    Returns
    -------
    memoized_func: function
        A memoized wrapper around `func`
    """
    cache = dict()

    def memoized_func(*args):
        if args in cache and memoize_funcs:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return memoized_func


def has_nans(X):
    if np.any(np.isnan(X)):
        return True
    else:
        return False


def check_for_nans(X):
    if has_nans(X):
        raise Exception("Has NaNs")


def binary(num, pre='', length=16, spacer=0):
    ''' 
        formats a number into binary:
        https://stackoverflow.com/a/16926270/3549879 
    '''
    return '{0}{{:{1}>{2}}}'.format(pre, spacer, length).format(bin(num)[2:])


def make_variable_name(var):
    """
    Converts a variable name to pySRURGS safe variable names. Prevents string
    manipulations on variable names from affecting function names.

    Parameters
    ----------
    var : string
        A variable name.

    Returns
    -------
    var_name: string
        `var` wrapped in the pySRURGS variable prefix and suffix.
    """
    var_name = variable_prefix + str(var) + variable_suffix
    return var_name


def make_parameter_name(par):
    """
    Converts a fitting parameter name to pySRURGS safe parameter name. Prevents
    string manipulations on parameter names from affecting function names.

    Parameters
    ----------
    par : string
        A variable name.

    Returns
    -------
    par_name: string
        `par` wrapped in the pySRURGS parameter prefix and suffix.
    """
    par_name = fitting_param_prefix + str(par) + fitting_param_suffix
    return par_name


def remove_variable_tags(equation_string):
    """
    Removes the pySRURGS variable prefix/suffix from an equation string.

    Parameters
    ----------
    equation_string : string
        A pySRURGS generated equation string

    Returns
    -------
    equation_string: string
        `equation_string` with the variable prefix and suffix removed.
    """
    equation_string = equation_string.replace(variable_prefix, '')
    equation_string = equation_string.replace(variable_suffix, '')
    return equation_string


def remove_parameter_tags(equation_string):
    """
    Removes the pySRURGS fitting parameter prefix/suffix from an equation 
    string.

    Parameters
    ----------
    equation_string : string
        A pySRURGS generated equation string

    Returns
    -------
    equation_string: string
        `equation_string` with the fitting parameter prefix and suffix removed.
    """
    equation_string = equation_string.replace(fitting_param_prefix, '')
    equation_string = equation_string.replace(fitting_param_suffix, '')
    return equation_string


def remove_tags(equation_string):
    """
    Removes the pySRURGS variable and fitting parameter prefix/suffix from an
    equation string.

    Parameters
    ----------
    equation_string : string
        A pySRURGS generated equation string

    Returns
    -------
    equation_string: string
        `equation_string` with the variable and fitting parameter prefixes and
        suffixes removed.
    """
    equation_string = remove_parameter_tags(equation_string)
    equation_string = remove_variable_tags(equation_string)
    return equation_string


def remove_dict_tags(equation_string):
    """
    Prior to numerically evaluating an equation string, we replace the variable
    and fitting parameter suffix/prefix with code to access values housed in
    dictionaries. This function removes that dictionary code from the equation
    string.

    Parameters
    ----------
    equation_string : string
        A pySRURGS generated equation string just prior to being numerically
        evaluated.

    Returns
    -------
    equation_string: string
        `equation_string` without dictionary access formatting.
    """
    equation_string = equation_string.replace('df["', '')
    equation_string = equation_string.replace('params["', '')
    equation_string = equation_string.replace('"].value', '')
    equation_string = equation_string.replace('"]', '')
    return equation_string


def create_variable_list(m):
    """
    Creates a list of all the variable names.

    Parameters
    ----------
    m : string (1) or int (2)
        (1) Absolute or relative path to a CSV file with a header
        (2) The number of independent variables in the dataset

    Returns
    -------
    my_vars: list
        A list with dataset variable names as elements.
    """
    if type(m) == str:
        my_vars = pandas.read_csv(m).keys()[:-1].tolist()
        my_vars = [make_variable_name(x) for x in my_vars]
    if type(m) == int:
        my_vars = []
        for i in range(0, m):
            my_vars.append(make_variable_name('x' + str(i)))
    return my_vars


def create_parameter_list(m):
    """
    Creates a list of all the fitting parameter names.

    Parameters
    ----------
    m : int
        The number of fitting parameters in the symbolic regression problem

    Returns
    -------
    my_pars: list
        A list with fitting parameter names as elements.
    """
    my_pars = []
    for i in range(0, m):
        my_pars.append(make_parameter_name('p' + str(i)))
    return my_pars


@memoize
def get_bits(x):
    ''' Gets the odd and even bits of a binary number in string format. Used by
    `get_left_right_bits`.
    '''
    # Get all even bits of x
    even_bits = x[::2]
    # Get all odd bits of x
    odd_bits = x[1::2]
    return odd_bits, even_bits


@memoize
def get_left_right_bits(my_int):
    """
    Converts an integer to binary and returns two integers, representing the odd
    and even bits of its binary value.

    Parameters
    ----------
    my_int : integer

    Returns
    -------
    left_int: int
        An integer corresponding to the decimal representation of the odd bits
        of `my_int`'s binary representation

    right_int: int
        An integer corresponding to the decimal representation of the even bits
        of `my_int`'s binary representation
    """
    # splits an integer into its odd and even bits - AKA left and right bits
    int_as_bin = binary(my_int)
    left_bin, right_bin = get_bits(int_as_bin)
    left_int = int(left_bin, 2)
    right_int = int(right_bin, 2)
    return left_int, right_int


def get_properties(dataframe, label):
    """
    Returns a dictionary of statistical data around the data found in
    `dataframe[label]`

    Parameters
    ----------
    dataframe : pandas.DataFrame object

    label: a column name within `dataframe`

    Returns
    -------
    properties: dictionary
        A dictionary containing the mean, standard deviation, min, and max of
        the column.
    """
    properties = dict()
    properties[label + '_mean'] = dataframe.mean()
    properties[label + '_std'] = dataframe.std()
    properties[label + '_min'] = dataframe.min()
    properties[label + '_max'] = dataframe.max()
    return properties


@memoize
def mempower(a, b):
    """
    Same as pow, but able to handle extremely large values, and memoized.

    Parameters
    ----------
    a: int
    b: int

    Returns
    -------
    result: int
        `a ** b`
    """
    result = mpmath.power(a, b)
    return result


def get_element_of_cartesian_product(*args, repeat=1, index=0):
    """
    Access a specific element of a cartesian product, without needing to iterate
    through the entire product.

    Parameters
    ----------
    args: iterable
        A set of iterables whose cartesian product is being accessed

    repeat: int
        If `args` is only one object, `repeat` specifies the number of times
        to take the cartesian product with itself.

    index: int
        The index of the cartesian product which we want to access

    Returns
    -------
    ith_item: the `index`th element of the cartesian product
    """
    pools = [tuple(pool) for pool in args] * repeat
    if len(pools) == 0:
        return []
    len_product = len(pools[0])
    len_pools = len(pools)
    for j in range(1, len_pools):
        len_product = len_product * len(pools[j])
    if index >= len_product:
        raise Exception("index + 1 is bigger than the length of the product")
    index_list = []
    for j in range(0, len_pools):
        ith_pool_index = index
        denom = 1
        for k in range(j + 1, len_pools):
            denom = denom * len(pools[k])
        ith_pool_index = ith_pool_index // denom
        if j != 0:
            ith_pool_index = ith_pool_index % len(pools[j])
        index_list.append(ith_pool_index)
    ith_item = []
    for index in range(0, len_pools):
        ith_item.append(pools[index][index_list[index]])
    return ith_item


def simplify_equation_string(eqn_str, dataset):
    """
    Simplify a pySRURGS equation string into a more human readable format

    Parameters
    ----------
    eqn_str: string
        pySRURGS equation string

    dataset: pySRURGS.Dataset
        The dataset object used to generate the `eqn_str`

    Returns
    -------
    eqn_str: string
        A simpler, more human readable version of `eqn_str`

    Notes
    -------
    Uses sympy to perform simplification. The dataset object specifies the sympy
    namespace.
    """
    dataset._sympy_namespace = dataset.make_sympy_namespace()
    s = sympy.sympify(eqn_str, locals=dataset._sympy_namespace)
    try:
        eqn_str = str(sympy.simplify(s))
    except ValueError:
        pass
    if 'zoo' in eqn_str:  # zoo (complex infinity) in sympy
        raise FloatingPointError
    eqn_str = remove_variable_tags(eqn_str)
    eqn_str = remove_parameter_tags(eqn_str)
    return eqn_str


def equation_generator_binary_tree(i, q, r, s, enumerator, SRconfig):
    """
    Generates an equation string given the integers that specify an equation
    string in pySRURGS. Use `equation_generator_full_binary_tree` instead when 
    there are no functions of arity one permitted.

    Parameters
    ----------
    i: int
        Specifies the integer representation of the unique binary tree. Must be
        greater than 0.

    q: int
        Specifies the integer representation of the configuration of the
        functions of arity one. Must be greater than 0 and less than the number
        of possible configurations of functions of arity one, `G`: see
        `Enumerator.get_G` for details.

    r: int
        Specifies the integer representation of the configuration of the
        functions of arity two. Must be greater than 0 and less than the number
        of possible configurations of functions of arity two, `A`: see
        `Enumerator.get_A` for details.

    s: int
        Specifies the integer representation of the configuration of the
        terminals. Must be greater than 0 and less than the number
        of possible configurations of functions of arity two, `B`: see
        `Enumerator.get_B` for details.

    enumerator: pySRURGS.Enumerator
        The `Enumerator` object for the symbolic regression problem

    SRconfig: pySRURGS.SymbolicRegressionConfig
        The `SymbolicRegressionConfig` for the symbolic regression problem

    Returns
    -------
    tree: string
        The equation string, without any simplifications
    """
    en = enumerator
    dataset = SRconfig._dataset
    f = len(SRconfig._f_functions)
    n = len(SRconfig._n_functions)    
    m = dataset._m_terminals
    tree = ith_binary_tree(i)
    G = en.get_G(f, i)
    if q >= G and not G == 0:
        raise Exception("q is an index that must be smaller than G")
    A = en.get_A(n, i)
    if r >= A:
        raise Exception("r is an index that must be smaller than A")
    B = en.get_B(m, i)
    if s >= B:
        raise Exception("s is an index that must be smaller than B")
    l_i = en.get_l_i(i)
    k_i = en.get_k_i(i)
    j_i = en.get_j_i(i)
    # of all the possible configurations of arity 1 functions, we pick the
    # configuration at index q
    f_func_config = get_element_of_cartesian_product(SRconfig._f_functions,
                                                     repeat=l_i, index=q)
    # of all the possible configurations of arity 2 functions, we pick the
    # configuration at index r
    n_func_config = get_element_of_cartesian_product(SRconfig._n_functions,
                                                     repeat=k_i, index=r)
    # of all the possible configurations of terminals, we pick the
    # configuration at index s
    term_config = get_element_of_cartesian_product(dataset._terminals_list,
                                                   repeat=j_i, index=s)
    orig_tree = tree
    # the trees are generated in the form [. , .] where . denotes a leaf,
    # and the square brackets indicate a function
    # we do some string replacements here, according to the determined
    # configurations to build the equation as a string
    for z in range(0, len(n_func_config)):
        func = n_func_config[z]
        tree = tree.replace('[', func + '(', 1)
        tree = tree.replace(']', ')', 1)    
    for z in range(0, len(f_func_config)):
        func = f_func_config[z]
        tree = tree.replace('|', func + '(', 1)
        tree = tree.replace('|', ')', 1)
    func_tree = tree
    for z in range(0, len(term_config)):
        term = term_config[z]
        tree = tree.replace('.', term, 1)
    return tree


def equation_generator_full_binary_tree(i, r, s, enumerator, SRconfig):
    """
    Generates an equation string given the integers that specify an equation
    string in pySRURGS. Use `equation_generator_binary_tree` instead when there 
    are functions of arity one permitted.

    Parameters
    ----------
    i: int
        Specifies the integer representation of the unique binary tree. Must be
        greater than 0.

    r: int
        Specifies the integer representation of the configuration of the
        functions of arity two. Must be greater than 0 and less than the number
        of possible configurations of functions of arity two, `A`: see
        `Enumerator.get_A` for details.

    s: int
        Specifies the integer representation of the configuration of the
        terminals. Must be greater than 0 and less than the number
        of possible configurations of functions of arity two, `B`: see
        `Enumerator.get_B` for details.

    enumerator: pySRURGS.EnumeratorFullBinaryTree
        The `EnumeratorFullBinaryTree` object for the symbolic regression 
        problem

    SRconfig: pySRURGS.SymbolicRegressionConfig
        The `SymbolicRegressionConfig` for the symbolic regression problem

    Returns
    -------
    tree: string
        The equation string, without any simplifications
    """
    en = enumerator
    dataset = SRconfig._dataset
    n = len(SRconfig._n_functions)    
    m = dataset._m_terminals
    tree = ith_full_binary_tree(i)
    A = en.get_A(n, i)
    if r >= A:
        raise Exception("r is an index that must be smaller than A")
    B = en.get_B(m, i)
    if s >= B:
        raise Exception("s is an index that must be smaller than B")
    k_i = en.get_k_i(i)
    j_i = en.get_j_i(i)
    n_func_config = get_element_of_cartesian_product(SRconfig._n_functions,
                                                     repeat=k_i,
                                                     index=r)
    term_config = get_element_of_cartesian_product(dataset._terminals_list,
                                                   repeat=j_i, index=s)
    orig_tree = tree
    for z in range(0, len(n_func_config)):
        func = n_func_config[z]
        tree = tree.replace('[', func + '(', 1)
        tree = tree.replace(']', ')', 1)
    func_tree = tree
    for z in range(0, len(term_config)):
        term = term_config[z]
        tree = tree.replace('.', term, 1)
    return tree


def random_equation_binary_tree(N, cum_weights, enumerator, SRconfig,
                                details=False, i=None):
    """
    Generates a random equation string. Generating the random numbers which
    specify the equation, then passes those as arguments to equation_generator.
    Use `random_equation_full_binary_tree` instead when there are no functions 
    of arity one permitted.

    Parameters
    ----------
    N: int
        Specifies the number of unique binary trees permitted in our search. The
        search considers trees mapping from the integer domain [0 ... `N`-1].

    cum_weights: array like
        Specifies the probability that an integer in [0 ... `N`-1] will be
        randomly selected. Should add to 1. pySRURGS gives preference to larger
        values of `N` because they result in more possible equations and we want
        to give each equation uniform probability of selection.

    enumerator: pySRURGS.Enumerator
        The `EnumeratorFullBinaryTree` object for the symbolic regression 
        problem

    SRconfig: pySRURGS.SymbolicRegressionConfig
        The `SymbolicRegressionConfig` for the symbolic regression problem

    details: Boolean
        Determines the output type
        
    i: int (default: None)
        If not none, specifies which binary tree to use

    Returns
    -------
    if `details` == False:
        equation_string: string
            A randomly generated pySRURGS equation string
    if `details` == True:
        [equation_string, N, n, f, m, i, q, r, s]
    """
    dataset = SRconfig._dataset
    n = len(SRconfig._n_functions) # number of functions of arity 2
    f = len(SRconfig._f_functions) # number of functions of arity 1
    m = dataset._m_terminals # number of different terminals    
    if i is None:
        i = randgen.choice(range(0, N), p=cum_weights)
    q = enumerator.get_q(f, i) # get the q^th configuration of arity 1 functions
    r = enumerator.get_r(n, i) # get the r^th configuration of arity 2 functions
    s = enumerator.get_s(m, i) # get the s^th configuration of terminals
    equation_string = equation_generator_binary_tree(i, q, r, s, enumerator, 
                                                     SRconfig)
    if not details:
        return equation_string
    else:
        result = [equation_string, N, n, f, m, i, q, r, s]
        return result


def random_equation_full_binary_tree(N, cum_weights, enumerator, 
                                     SRconfig, details=False, i=None):
    """
    Generates a random equation string. Generating the random numbers which
    specify the equation, then passes those as arguments to equation_generator.
    Use `random_equation_binary_tree` instead when there are functions of arity 
    one permitted.

    Parameters
    ----------
    N: int
        Specifies the number of unique binary trees permitted in our search. The
        search considers trees mapping from the integer domain [0 ... `N`-1].

    cum_weights: array like
        Specifies the probability that an integer in [0 ... `N`-1] will be
        randomly selected. Should add to 1. pySRURGS gives preference to larger
        values of `N` because they result in more possible equations and we want
        to give each equation uniform probability of selection.

    enumerator: pySRURGS.EnumeratorFullBinaryTree
        The `EnumeratorFullBinaryTree` object for the symbolic regression 
        problem

    SRconfig: pySRURGS.SymbolicRegressionConfig
        The `SymbolicRegressionConfig` object for the symbolic regression 
        problem

    details: Boolean
        Determines the output type

    i: int (default: None)
        If not none, specifies which full binary tree to use

    Returns
    -------
    if `details` == False:
        equation_string: string
            A randomly generated pySRURGS equation string
    if `details` == True:
        [equation_string, N, n, m, i, q, r, s]
    """
    dataset = SRconfig._dataset
    n = len(SRconfig._n_functions) # number of functions of arity 2
    m = dataset._m_terminals # number of different terminals    
    if i is None:
        i = randgen.choice(range(0, N), p=cum_weights)
    r = enumerator.get_r(n, i) # get the r^th configuration of arity 2 functions
    s = enumerator.get_s(m, i) # get the s^th configuration of terminals
    equation_string = equation_generator_full_binary_tree(i, r, s, enumerator, 
                                                          SRconfig)
    if not details:
        return equation_string
    else:
        result = [equation_string, N, n, m, i, r, s]
        return result


class EnumeratorBinaryTree(object):
    """
    An object housing methods for enumeration of the symbolic regression
    problem. Use `EnumeratorFullBinaryTree` instead when no functions of arity 
    one permitted.

    Returns
    -------
    self
        A pySRURGS.Enumerator object which houses various methods for the
        enumeration of the problem space.

    Example
    --------

    >>> import pySRURGS
    >>> en = pySRURGS.Enumerator()
    >>> en.get_M(1000, 5, 5, 5)
    """

    @memoize
    def get_M(self, N, f, n, m):
        """
        Calculate the total number of equations for this symbolic regression
        problem. Use get_M from `EnumeratorFullBinaryTree` instead if the number 
        of functions of arity one permitted is zero.

        Parameters
        ----------
        N: int
            Specifies the number of unique binary trees permitted in our search.
            We consider trees mapping from the integer domain [0 ... `N`-1].


        f: int
            The number of functions of arity one permitted

        n: int
            The number of functions of arity two permitted

        m: int
            The number of terminals in the problem (includes variables and
            fitting parameters)

        Returns
        -------
        M: int (mpfmath format)
            The number of possible equations in this symbolic regression problem
        """
        def get_count(i):
            l_i = self.get_l_i(i)
            k_i = self.get_k_i(i)
            j_i = self.get_j_i(i)
            count = mempower(n, k_i) * mempower(m, j_i) * mempower(f, l_i)
            return count
        M = mpmath.nsum(get_count, [0, N - 1])
        return M

    @memoize
    def get_G(self, f, i):
        """
        Calculate the total number of configurations of the functions of arity
        one for the binary tree mapped from the integer `i`.  Use get_G from
        `EnumeratorFullBinaryTree` instead if the number of functions of arity 
        one permitted is zero.

        Parameters
        ----------
        f: int
            The number of functions of arity one permitted

        i: int
            The `i`^th binary tree in the integer to tree mapping

        Returns
        -------
        G: int (mpfmath format)
            The number of possible configurations of functions of arity one
        """
        l = self.get_l_i(i)
        G = mempower(f, l)
        return G

    def get_A(self, n, i):
        """
        Calculate the total number of configurations of the functions of arity
        two for the binary tree mapped from the integer `i`.  Use get_A from
        `EnumeratorFullBinaryTree` instead if the number of functions of arity 
        one permitted is zero.

        Parameters
        ----------
        n: int
            The number of functions of arity two permitted

        i: int
            The `i`^th binary tree in the integer to tree mapping

        Returns
        -------
        A: int (mpfmath format)
            The number of possible configurations of functions of arity two
        """
        k = self.get_k_i(i)
        A = mempower(n, k)
        return A

    @memoize
    def get_B(self, m, i):
        """
        Calculate the total number of configurations of the terminals for the
        binary tree mapped from the integer `i`.  Use get_B from
        `EnumeratorFullBinaryTree` instead if the number of functions of arity 
        one permitted is zero.

        Parameters
        ----------
        m: int
            The number of terminals including variables and fitting parameters

        i: int
            The `i`^th binary tree in the integer to tree mapping

        Returns
        -------
        B: int (mpfmath format)
            The number of possible configurations of terminals
        """
        j = self.get_j_i(i)
        B = mempower(m, j)
        return B

    def get_q(self, f, i):
        ''' Generates a random integer between 0 and `G` - 1, inclusive '''
        G = self.get_G(f, i)
        try:
            q = randgen.randint(0, G - 1, dtype=np.int64)
        except ValueError as e:
            if G == 1:
                q = 0
            else:
                print(e)
                raise ValueError
        return q

    def get_r(self, n, i):
        ''' Generates a random integer between 0 and `A` - 1, inclusive '''
        A = self.get_A(n, i)
        try:
            r = randgen.randint(0, A - 1, dtype=np.int64)
        except ValueError as e:
            if A == 1:
                r = 0
            else:
                print(e)
                raise ValueError
        return r

    def get_s(self, m, i):
        ''' Generates a random integer between 0 and `B` - 1, inclusive '''
        B = self.get_B(m, i)
        try:
            s = randgen.randint(0, B - 1, dtype=np.int64)
        except ValueError as e:
            if B == 1:
                s = 0
            else:
                print(e)
                raise ValueError
        return s

    @memoize
    def get_l_i(self, i):
        '''
            from `f` functions of arity one, pick `l_i `
            `l_i` is the number of non-leaf nodes of arity one
            in the tree corresponding to `i`
        '''
        i = int(i)
        if i == 0:
            l_i = 0
        elif i == 1:
            l_i = 0
        elif i == 2:
            l_i = 1
        else:
            left_int, right_int = get_left_right_bits(i)
            left_l_i = self.get_l_i(left_int)
            right_l_i = self.get_l_i(right_int)
            l_i = left_l_i + right_l_i + 1
        return l_i

    @memoize
    def get_k_i(self, i):
        '''
            from `n` functions of arity two, pick `k_i`
            `k_i` is the number of non-leaf nodes
            in the tree corresponding to `i`
        '''
        i = int(i)
        if i == 0:
            k_i = 0
        elif i == 1:
            k_i = 1
        elif i == 2:
            k_i = 0
        else:
            left_int, right_int = get_left_right_bits(i)
            left_k_i = self.get_k_i(left_int)
            right_k_i = self.get_k_i(right_int)
            k_i = left_k_i + right_k_i + 1
        return k_i

    @memoize
    def get_j_i(self, i):
        '''
            from `m` terminals, pick `j_i`
            `j_i` is the number of leafs in the tree corresponding to `i`
        '''
        i = int(i)
        if i == 0:
            j_i = 1
        elif i == 1:
            j_i = 2
        elif i == 2:
            j_i = 1
        else:
            left_int, right_int = get_left_right_bits(i)
            left_j_i = self.get_j_i(left_int)
            right_j_i = self.get_j_i(right_int)
            j_i = left_j_i + right_j_i
        return j_i


class EnumeratorFullBinaryTree(object):
    """
    An object housing methods for enumeration of the symbolic regression
    problem. Use `EnumeratorBinaryTree` instead when functions of arity one are permitted.

    Returns
    -------
    Enumerator
        A pySRURGS.EnumeratorFullBinaryTree object which houses various methods for the
        enumeration of the problem space for case where only functions of arity
        two are permitted.

    Example
    --------

    >>> import pySRURGS
    >>> en = pySRURGS.EnumeratorFullBinaryTree()
    >>> en.get_M(1000, 5, 5)
    """

    @memoize
    def get_M(self, N, n, m):
        """
        Calculate the total number of equations for this symbolic regression
        problem. Use get_M from `EnumeratorBinaryTree` instead if any functions
        of arity one are permitted.

        Parameters
        ----------
        N: int
            Specifies the number of unique binary trees permitted in our search.
            We consider trees mapping from the integer domain [0 ... `N`-1].

        n: int
            The number of functions of arity two permitted

        m: int
            The number of terminals in the problem (includes variables and
            fitting parameters)

        Returns
        -------
        M: int (mpfmath format)
            The number of possible equations in this symbolic regression problem
        """
        def get_count(i):
            k_i = self.get_k_i(i)
            j_i = self.get_j_i(i)
            count = mempower(n, k_i) * mempower(m, j_i)
            return count
        M = mpmath.nsum(get_count, [0, N - 1])
        return M

    @memoize
    def get_A(self, n, i):
        """
        Calculate the total number of configurations of the functions of arity
        two for the binary tree mapped from the integer `i`.  Use get_A from
        `EnumeratorBinaryTree` instead if the number of functions of arity one permitted
        is not zero.

        Parameters
        ----------
        n: int
            The number of functions of arity two permitted

        i: int
            The `i`^th binary tree in the integer to tree mapping

        Returns
        -------
        A: int (mpfmath format)
            The number of possible configurations of functions of arity two
        """
        k = self.get_k_i(i)
        A = mempower(n, k)
        return A

    @memoize
    def get_B(self, m, i):
        """
        Calculate the total number of configurations of the terminals for the
        binary tree mapped from the integer `i`.  Use get_B from
        `EnumeratorBinaryTree` instead if the number of functions of arity one permitted
        is not zero.

        Parameters
        ----------
        m: int
            The number of terminals including variables and fitting parameters

        i: int
            The `i`^th binary tree in the integer to tree mapping

        Returns
        -------
        B: int (mpfmath format)
            The number of possible configurations of terminals
        """
        j = self.get_j_i(i)
        B = mempower(m, j)
        return B

    def get_r(self, n, i):
        ''' Generates a random integer between 0 and `A` - 1, inclusive '''
        A = self.get_A(n, i)
        try:
            r = randgen.randint(0, A - 1, dtype=np.int64)
        except ValueError as e:
            if A == 1:
                r = 0
            else:
                print(e)
                raise ValueError
        return r

    def get_s(self, m, i):
        ''' Generates a random integer between 0 and `B` - 1, inclusive '''        
        B = self.get_B(m, i)
        try:
            s = randgen.randint(0, B - 1, dtype=np.int64)
        except ValueError as e:
            if B == 1:
                s = 0
            else:
                print(e)
                raise ValueError
        return s

    @memoize
    def get_k_i(self, i):
        '''
            from `n` functions of arity two, pick `k_i`
            `k_i` is the number of non-leaf nodes
            in the tree corresponding to `i`
        '''
        i = int(i)
        if i == 0:
            k_i = 0
        elif i == 1:
            k_i = 1
        else:
            left_int, right_int = get_left_right_bits(i - 1)
            left_k_i = self.get_k_i(left_int)
            right_k_i = self.get_k_i(right_int)
            k_i = left_k_i + right_k_i + 1
        return k_i

    @memoize
    def get_j_i(self, i):
        '''
            from `m` terminals, pick `j_i`
            `j_i` is the number of leafs
            in the tree corresponding to `i`
        '''
        i = int(i)
        if i == 0:
            j_i = 1
        elif i == 1:
            j_i = 2
        else:
            left_int, right_int = get_left_right_bits(i - 1)
            left_j_i = self.get_j_i(left_int)
            right_j_i = self.get_j_i(right_int)
            j_i = left_j_i + right_j_i
        return j_i


def create_fitting_parameters(max_params, param_values=None):
    """
    Creates the lmfit.Parameters object based on the number of fitting 
    parameters permitted in this symbolic regression problem.

    Parameters
    ----------
    max_params: int
        The maximum number of fitting parameters. Same as `max_num_fit_params`.

    param_values: None OR (numpy.array of length max_params)
        Specifies the values of the fitting parameters. If none, will default
        to an array of ones, which are to be optimized later.

    Returns
    -------
    params: lmfit.Parameters
        Fitting parameter names specified as ['p' + str(integer) for integer
        in range(0, max_params)]
    """
    params = lmfit.Parameters()
    for int_param in range(0, max_params):
        param_name = 'p' + str(int_param)
        param_init_value = np.float(1)
        params.add(param_name, param_init_value)
    if param_values is not None:
        for int_param in range(0, max_params):
            param_name = 'p' + str(int_param)
            params[param_name].value = param_values[int_param]
    return params


def eval_equation(params, function_string, SRconfig, mode='residual'):
    """
    Evaluates the equation numerically.

    Parameters
    ----------
    params: lmfit.Parameters
        The lmfit.parameters object used to optimize fitting parameters

    function_string: string
        The equation string in dictionary code tags in place.
        Use `clean_funcstring` to put in place the dictionary code tags.

    SRconfig: pySRURGS.SymbolicRegressionConfig
        The symbolic regression configuration object for this problem

    mode: string
        'residual' or 'y_calc'.

    Returns
    -------
    output: array like
        An array of residuals or predicted dependent variable values, depending
        on the value of `mode`.
    """
    my_data = SRconfig._dataset
    len_data = len(my_data._y_data)
    df = my_data._data_dict
    pd = params.valuesdict()
    y_label = my_data._y_label
    independent_var_vector = df[y_label]
    # residual = [BIG_NUM] * len(df[y_label])
    if mode == 'residual':
        eval_string = '(' + function_string + ') -  df["' + y_label + '"]'
        residual = eval(eval_string)
        if my_data._data_weights is not None:
            residual = np.multiply(residual, my_data._data_weights)
        output = residual
    elif mode == 'y_calc':
        y_value = eval(function_string)
        output = y_value
    elif type(mode) == dict:
        df = mode
        y_value = eval(function_string)
        output = y_value    
    if np.size(output) == 1:
        # if model only has parameters and no data variables, we can have a
        # situation where output is a single constant
        output = np.resize(output, np.size(independent_var_vector))
    return output


def clean_funcstring_params(funcstring):
    """
    Replaces the pySRURGS fitting parameter prefix/suffix from an equation
    string with the dictionary codes needed to numerically evaluate the equation
    string.

    Parameters
    ----------
    funcstring : string
        A pySRURGS generated equation string

    Returns
    -------
    funcstring: string
        `funcstring` with the fitting parameters' prefix and suffix replaced
        with the dictionary tags.
    """
    funcstring = funcstring.replace(fitting_param_prefix, 'params["')
    funcstring = funcstring.replace(fitting_param_suffix, '"].value')
    return funcstring


def clean_funcstring_vars(funcstring):
    """
    Replaces the pySRURGS variable prefix/suffix from an equation
    string with the dictionary codes needed to numerically evaluate the equation
    string.

    Parameters
    ----------
    funcstring : string
        A pySRURGS generated equation string

    Returns
    -------
    funcstring: string
        `funcstring` with the variables' prefix and suffix replaced with
        the dictionary tags.
    """
    funcstring = funcstring.replace(variable_prefix, 'df["')
    funcstring = funcstring.replace(variable_suffix, '"]')
    return funcstring


def clean_funcstring(funcstring):
    """
    Replaces the pySRURGS variables' and fitting parameters' prefix/suffix from
    an equation string with the dictionary codes needed to numerically evaluate
    the equation string.

    Parameters
    ----------
    funcstring : string
        A pySRURGS generated equation string

    Returns
    -------
    funcstring: string
        `funcstring` with the variables' and fitting parameters' prefix and
        suffix replaced with the dictionary tags.
    """
    funcstring = clean_funcstring_vars(funcstring)
    funcstring = clean_funcstring_params(funcstring)
    return funcstring


def check_goodness_of_fit(individual, params, SRconfig):
    """
    Calculates metrics to assess the goodness of fit. Does so by first
    optimizing the fitting parameters.

    Parameters
    ----------
    individual : string (or, alternatively, a DEAP individual object)
        A pySRURGS generated equation string

    params: lmfit.Parameters

    SRconfig: pySRURGS.SymbolicRegressionConfig

    Returns
    -------
    result: tuple
        (sum_of_squared_residuals, sum_of_squared_totals, R2,
         params_dict_to_store, residual)
    """
    my_data = SRconfig._dataset
    # If funcstring is a tree, transform to string
    funcstring = str(individual)
    funcstring = clean_funcstring(funcstring)
    # Evaluate the sum of squared difference between the expression
    if len(params) > 0:
        result = lmfit.minimize(eval_equation, params,
                                args=(funcstring, SRconfig),
                                method='leastsq', 
                                nan_policy='propagate')
        residual = result.residual
        y_calc = eval_equation(result.params, 
                               funcstring, 
                               SRconfig,
                               mode='y_calc')
        params_dict_to_store = result.params
    else:
        residual = eval_equation(params, funcstring, SRconfig)
        y_calc = eval_equation(params, funcstring, SRconfig, mode='y_calc')
        params_dict_to_store = params
    avg_y_data = np.average(my_data._y_data)
    sum_of_squared_residuals = sum(pow(residual, 2))
    sum_of_squared_totals = sum(pow(y_calc - avg_y_data, 2))
    R2 = 1 - sum_of_squared_residuals / sum_of_squared_totals
    result = (sum_of_squared_residuals,
              sum_of_squared_totals,
              R2,
              params_dict_to_store,
              residual)
    return result


@memoize
def ith_binary_tree(i):
    """
    Generates the `i`th binary tree. Use ith_full_binary_tree when no functions
    of arity one are permitted.

    Parameters
    ----------
    i: int
        A non-negative integer which will be used to map to a unique binary tree

    Returns
    -------
    tree: string
        The binary tree represented using [.,.] to represent a full binary tree,
        |.| to represent functions of arity one, and '.' to represent terminals.
    """
    if i == 0:
        tree = '.'
    elif i == 1:
        tree = '[., .]'
    elif i == 2:
        tree = '|.|'
    else:
        left_int, right_int = get_left_right_bits(i)
        left = ith_binary_tree(left_int)
        right = ith_binary_tree(right_int)
        tree = '[' + left + ', ' + right + ']'
    return tree


@memoize
def ith_full_binary_tree(i):
    """
    Generates the `i`th binary tree. Use ith_binary_tree when functions
    of arity one are permitted.

    Parameters
    ----------
    i: int
        A non-negative integer which will be used to map to a unique binary tree

    Returns
    -------
    tree: string
        The binary tree represented using [.,.] to represent a full binary tree,
        and '.' to represent terminals.
    """
    if i == 0:
        tree = '.'
    elif i == 1:
        tree = '[., .]'
    else:
        left_int, right_int = get_left_right_bits(i - 1)
        left = ith_full_binary_tree(left_int)
        right = ith_full_binary_tree(right_int)
        tree = '[' + left + ', ' + right + ']'
    return tree


@memoize
def get_cum_weights_binary_tree(N, f, n, m, enumerator):
    """
    Generates the relative probabilities of selecting the `i`th binary tree.
    Sums to 1. Ensures that each equation has equal probability of selection.
    Gives increasing probability with increasing `i`, because larger values
    of `i` correspond to more complex trees which permit more equations.
    Use `get_cum_weights_full_binary_tree` when functions of arity one are not permitted.

    Parameters
    ----------
    N: int
        Specifies the number of unique binary trees permitted in our search.
        We consider trees mapping from the integer domain [0 ... `N`-1].

    f: int
        The number of functions of arity one permitted

    n: int
        The number of functions of arity two permitted

    m: int
        The number of terminals in the problem (includes variables and
        fitting parameters)

    enumerator: pySRURGS.EnumeratorBinaryTree
        The enumerator of the symbolic regression problem

    Returns
    -------
    cum_weights: numpy.array
        The relative probability of selecting `i` in the range(0,N)
        to ensure that the equations from the corresponding binary trees
        have equal probability of selection
    """
    en = enumerator
    weights = [en.get_G(f,i) * en.get_A(n,i) * en.get_B(m, i) for i in 
               range(0, N)]
    cum_weights = np.array(weights) / np.sum(weights)
    cum_weights = cum_weights.astype(np.float64)
    return cum_weights


@memoize
def get_cum_weights_full_binary_tree(N, n, m, enumerator):
    """
    Generates the relative probabilities of selecting the `i`th binary tree.
    Sums to 1. Ensures that each equation has equal probability of selection.
    Gives increasing probability with increasing `i`, because larger values
    of `i` correspond to more complex trees which permit more equations.
    Use `get_cum_weights_binary_tree` when functions of arity one are permitted.

    Parameters
    ----------
    N: int
        Specifies the number of unique binary trees permitted in our search.
        We consider trees mapping from the integer domain [0 ... `N`-1].

    n: int
        The number of functions of arity two permitted

    m: int
        The number of terminals in the problem (includes variables and
        fitting parameters)

    enumerator: pySRURGS.EnumeratorFullBinaryTree
        The enumerator of the symbolic regression problem

    Returns
    -------
    cum_weights: numpy.array
        The relative probability of selecting `i` in the range(0,N)
        to ensure that the equations from the corresponding binary trees
        have equal probability of selection
    """
    en = enumerator
    weights = [en.get_A(n, i) * en.get_B(m, i) for i in range(0, N)]
    cum_weights = np.array(weights) / np.sum(weights)
    cum_weights = cum_weights.astype(np.float64)
    return cum_weights


class ResultList(object):
    """
    Stores multiple results from a pySRURGS run. Typically, is loaded from the
    SqliteDict database file. `self._results` needs to be generated by appending
    `Result` objects to it.

    Returns
    -------
    self: ResultList
    """

    def __init__(self):
        self._results = []

    def sort(self):
        """
        Sorts the results in the result list by decreasing value of mean squared
        error.
        """
        self._results = sorted(self._results, key=lambda x: x._MSE)

    def print(self, y_data, top=5, mode='succinct'):
        """
        Prints the Normalized Mean Squared Error, R^2, Equation (simplified),
        and Parameters values for the top results_dict. Run `self.sort` prior
        to executing `self.print`.

        Parameters
        ----------
        y_data: array like
            The dependent variable's data from the pySRURGS.Dataset object

        top: int
            The number of results to display. Will be the best models if
            `self.sort` has been run prior to printing.

        mode: string
            'succinct' or 'detailed' depending on whether you want to see
            the difficult to read original equation string

        Returns
        -------
        table_string: string
            A table housing the Normalized Mean Squared Error, R^2,
            Equation (simplified), and Parameters values in the
            tabulate package format.
        """
        table = []
        header = ["Normalized Mean Squared Error", "R^2",
                  "Equation, simplified", "Parameters"]
        num_eqn = int(np.min((top, len(self._results))))
        for i in range(0, num_eqn):
            row = self._results[i].summarize(mode)
            row[0] = row[0] / np.std(y_data)
            table.append(row)
        table_string = tabulate.tabulate(table, headers=header)
        print(table_string)


def initialize_db(path_to_db):
    '''
        Initializes the SqliteDict database file with an initial null value
        for the 'best_result' key
    '''
    with SqliteDict(path_to_db, autocommit=True) as results_dict:
        try:
            results_dict['best_result']
        except KeyError:
            results_dict['best_result'] = Result(
                None, None, np.inf, None, None)
    return


def uniform_random_global_search_once(seed, SRconfig):
    """
    Runs pySRURGS once against the CSV file

    Parameters
    ----------
    SRconfig: pySRURGS.SymbolicRegressionConfig
        The symbolic regression configuration object for this problem

    seed: int (or None)
        Sets the seed of the pseudorandom number generator. Will make results
        reproducible if not None.

    Returns
    -------
    y_calc: array like
        The predicted values of the dependent variable

    """    
    (f, n, m, cum_weights, N, dataset, enumerator, _, _) = setup(SRconfig)
    valid = False
    if seed is not None:
        randgen.seed(seed)
    while valid == False:
        if f == 0:
            eqn_str = random_equation_full_binary_tree(N, cum_weights, 
                                                       enumerator, SRconfig)
        else:
            eqn_str = random_equation_binary_tree(N, cum_weights, enumerator, 
                                                  SRconfig)
        try:
            simple_eqn = simplify_equation_string(eqn_str, dataset)
            path_to_db = SRconfig._path_to_db
            initialize_db(path_to_db)
            params = create_fitting_parameters(dataset._int_max_params)
            (sum_of_squared_residuals, sum_of_squared_totals,
             R2, params_fitted,
             residual) = check_goodness_of_fit(eqn_str, params, SRconfig)
            if np.isnan(R2) or np.isnan(sum_of_squared_totals):
                raise FloatingPointError
            valid = True
        except FloatingPointError:
            pass
    MSE = sum_of_squared_residuals
    result = Result(simple_eqn, eqn_str, MSE, R2, params_fitted)
    return result


def uniform_random_global_search_once_to_queue(seed, SRconfig, queue):
    '''
        Runs uniform random global search once, then takes the result and puts 
        it in a queue so that it can be committed in a multiprocessing safe 
        fashion
    '''
    result = uniform_random_global_search_once(seed, SRconfig)
    queue.put(result)    


def uniform_random_global_search_once_to_db(seed, SRconfig):
    '''
        Runs uniform random global search once, then takes the result and saves 
        it immediately to the database
    '''
    result = uniform_random_global_search_once(seed, SRconfig)
    path_to_db = SRconfig._path_to_db
    with SqliteDict(SRconfig._path_to_db, autocommit=False) as results_dict:
        simple_eqn = result._simple_equation
        results_dict[simple_eqn] = result
        results_dict.commit()    

        
def solution_saving_worker(queue, n_items, output_db):
    """
        Takes solutions from the queue of evaluated solutions, 
        then saves them to the database.
        This function was not working - so I dropped multiprocessing.
        I also found it more trouble than it was worth performance wise.
    """
    checkpoint = int(n_items/100) + 1
    with SqliteDict(output_db, autocommit=False) as results_dict:
        for j in range(0, n_items):
            result = queue.get()
            simple_eqn = result._simple_equation
            results_dict[simple_eqn] = result
            if j == checkpoint:
                print('  Saving results to db: ' + str(j/n_items))
                results_dict.commit()
        results_dict.commit()
        print('  Saving results to db.')


def generate_benchmark(benchmark_name, SRconfig):
    """
    Generate a random symbolic regression benchmark problem.

    Parameters
    ----------
    benchmark_name: string
        A string that denotes the name of the problem. Typically, an integer
        set as a string will do.

    SRconfig: pySRURGS.SymbolicRegressionConfig
        The symbolic regression configuration object for this problem.

    Returns
    -------
    None


    Notes
    -----
    Saves a CSV file to: `benchmarks_dir + '/' + benchmark_name + '_train.csv'`
    Saves a CSV file to: `benchmarks_dir + '/' + benchmark_name + '_test.csv'`
    Saves a human readable text file to:
            `benchmarks_dir + '/' + benchmark_name + '_params.txt'`

    `benchmarks_dir` is a global variable typically set to './benchmarks'
    """
    (f, n, m, cum_weights, N, dataset,
     enumerator, n_functions, f_functions) = setup(SRconfig)
    valid = False
    while valid == False:
        print(ERASE_LINE + "Iterating...", end='\r')
        try:
            # specify the equation
            if f == 0:
                eqn_details = random_equation_full_binary_tree(N, cum_weights,
                                             enumerator, SRconfig, details=True)
            else:
                eqn_details = random_equation_binary_tree(N, cum_weights,
                                             enumerator, SRconfig, details=True)
            eqn_original = eqn_details[0]
            eqn_simple = simplify_equation_string(eqn_original, dataset)
            eqn_specifiers = eqn_details[1:]
            # create the fitting parameters values
            fitting_parameters = (randgen.random_sample((5)) - 0.5) * 20
            fit_param_list = create_fitting_parameters(5)
            for i in range(0, 5):
                fit_param_list['p' + str(i)].value = fitting_parameters[i]
            eqn_original_cleaned = clean_funcstring(eqn_original)
            # create the training dataset
            for zz in range(0, 2):  # 1st iteration, train. 2nd iteration, test.
                sample_dataset = np.zeros((100, 6))
                for j in range(0, 5):
                    sample_dataset[:, j] = randgen.random_sample((100,)) * 10
                dataset._dataframe = pandas.DataFrame(sample_dataset)
                dataset._dataframe.columns = dataset._x_labels.tolist() + \
                    [dataset._y_label]
                dataset._data_dict = dataset.get_data_dict()
                # calculate y for the sample problem
                y_calc = eval_equation(fit_param_list, eqn_original_cleaned,
                                       SRconfig, mode='y_calc')
                dataset._dataframe[dataset._y_label] = y_calc
                # save the test and train sets to file
                if zz == 0:
                    path1 = benchmarks_dir + '/' + benchmark_name + '_train.csv'
                else:
                    path1 = benchmarks_dir + '/' + benchmark_name + '_test.csv'
                dataset._dataframe.to_csv(path1, index=False)
                # test to see that the dataset can be loaded
                dataset_test = Dataset(path1, 5)
                path2 = benchmarks_dir + '/' + benchmark_name + '_params.txt'
                # save the problem parameters to a text file
                with open(path2, "w") as text_file:
                    msg = 'Permitted variables: ' + \
                        str(dataset._x_labels.tolist()) + '\n'
                    msg += 'Permitted fitting parameters: ' + \
                        str(list(fit_param_list.keys())) + '\n'
                    msg += 'Fitting parameters: '
                    msg += str(np.array(fit_param_list)) + '\n'
                    msg += 'Permitted functions: ' + \
                        str(f_functions + n_functions) + '\n'
                    msg += 'Simplified equation: ' + str(eqn_simple) + '\n'
                    eqn_original = remove_variable_tags(eqn_original)
                    eqn_original = remove_parameter_tags(eqn_original)
                    msg += 'Raw equation: ' + str(eqn_original) + '\n'
                    text_file.write(msg)
            valid = True
            if eqn_simple == '0':
                valid = False
        except Exception as e:
            print(ERASE_LINE + str(e), end='\r')
            valid = False


def setup(SR_config):
    """
    Returns the values stored in SR_config for use in the algorithm
        
    Parameters
    ----------

    SR_config: pySRURGS.SymbolicRegressionConfig
        The symbolic regression configuration object for this problem

    Returns
    -------
    result: tuple
        (f, n, m, cum_weights, N, dataset, enumerator, n_funcs, f_funcs)


    Notes
    -----
    f : int
        The number of functions of arity one in the problem

    n: int
        The number of functions of arity two in the problem

    cum_weights: array like
        The relative weight of selecting the binary trees from 0 to N-1, 
        calculated to ensure equal probabilty of selecting each equation.

    N: int
        The number of unique binary trees permitted in the search. Same as
        `max_permitted_trees`.

    dataset: pySRURGS.Dataset
        The dataset object for the problem.

    enumerator: pySRURGS.EnumeratorBinaryTree OR 
                pySRURGS.EnumeratorFullBinaryTree (if `f_funcs` == 0)

    n_funcs: list
        A list of strings of the functions of arity two permitted in this search

    f_funcs: list
        A list of strings of the functions of arity one permitted in this search
    """
    # reads the configuration, the csv file, and creates needed objects
    N = SR_config._max_permitted_trees
    n = len(SR_config._n_functions)  # the number of functions of arity 2
    n_funcs = SR_config._n_functions
    f_funcs = SR_config._f_functions
    f = len(SR_config._f_functions)  # the number of functions of arity 1
    num_fit_param = SR_config._max_num_fit_params
    dataset = SR_config._dataset
    m = dataset._m_terminals  # the number of vars + number of fit params
    if f == 0:
        enumerator = EnumeratorFullBinaryTree()
        cum_weights = get_cum_weights_full_binary_tree(N, n, m, enumerator)
    else:
        enumerator = EnumeratorBinaryTree()
        cum_weights = get_cum_weights_binary_tree(N, f, n, m, enumerator)
    return (f, n, m, cum_weights, N, dataset, enumerator, n_funcs, f_funcs)


def create_db_name(path_to_csv, additional_name=None):
    '''
    Generates a name of a SqliteDict file based on the CSV filename

    Parameters
    ----------
    path_to_csv: string
        An absolute or relative path to the CSV for the problem.

    additional_name: string
        An additional specifier used in generating the database file name

    Returns
    -------
    db_name: string
        A filepath starting with './db/' and matching with the CSV file name
        with an optional `additional_name` within the file name. Ends in '.db'.
    '''
    csv_basename = os.path.basename(path_to_csv)
    csv_name = csv_basename[:-4]
    if additional_name is not None:
        db_name = './db/' + csv_name + additional_name + '.db'
    else:
        db_name = './db/' + csv_name + '.db'
    return db_name


def get_resultlist(path_to_db):
    '''
    Loads the ResultList of a previous run of pySRURGS.
    Skips the `best_result` key. 

    Parameters
    ----------
    path_to_db: string
        An absolute or relative path to the SqliteDict database file.

    Returns
    -------
    result_list: pySRURGS.ResultList
        An unsorted ResultList object for the previously run problem
    '''
    result_list = ResultList()
    with SqliteDict(path_to_db) as results_dict:
        keys = results_dict.keys()        
        for eqn in keys:
            if eqn == 'best_result':
                continue
            result = results_dict[eqn]
            result_list._results.append(result)
    return result_list


def compile_results(SRconfig, print_mode=True):
    '''
    Reads from the generated SqliteDict file to determine the best stored models

    Parameters
    ----------
    SRconfig: pySRUGS.SymbolicRegressionConfig
        The symbolic regression problem's configuration object       

    print_mode: Boolean
        If true, prints to screen. If false, does not.

    Returns
    -------
    result_list: pySRURGS.ResultList 
        Compiling all the results found in the `path_to_db` database file
    '''
    dataset = SRconfig._dataset
    path_to_db = SRconfig._path_to_db
    result_list = get_resultlist(path_to_db)
    result_list.sort()
    if print_mode == True:
        result_list.print(dataset._y_data)
    return result_list


def count_results(path_to_db):
    '''
    Reads the generated SqliteDict file to determine the number of generated 
    models

    Parameters
    ----------
    path_to_db: string
        An absolute or relative path to the SqliteDict database file.

    Returns
    -------
    n_results: int
        The number of results in the ResultList
    '''
    result_list = get_resultlist(path_to_db)
    n_results = len(result_list._results)
    return n_results


def plot_results(SRconfig, output_dir='./image/'):
    '''
    Reads the generated SqliteDict file to determine the best model,
    then plots it against the raw data. saves the figure to './image/plot.png'
    and './image/plot.svg'. Only works for univariate data.

    Parameters
    ----------
    SRconfig: pySRUGS.SymbolicRegressionConfig
        The symbolic regression problem's configuration object

    Returns
    -------
    None

    Raises
    ------
    Exception, if data is not univariate.

    '''
    path_to_csv = SRconfig._path_to_csv
    path_to_db = SRconfig._path_to_db
    dataset = SRconfig._dataset
    result_list = get_resultlist(path_to_db)
    result_list.sort()
    best_model = result_list._results[0]
    param_values = best_model._params
    equation_string = best_model._equation
    num_params = len(param_values)
    params_obj = create_fitting_parameters(num_params,
                                           param_values=param_values)
    evaluatable_equation_string = equation_string
    eval_eqn_string = clean_funcstring(equation_string)
    plt.figure(figsize=(3.14, 2))
    if len(dataset._x_labels) == 1:
        data_dict = dict()
        xlabel = dataset._x_labels[0]
        data_dict[xlabel] = np.linspace(np.min(dataset._x_data),
                                        np.max(dataset._x_data))
        y_calc = eval_equation(params_obj, eval_eqn_string, SRconfig, 
                               mode=data_dict)
        plt.plot(data_dict[xlabel], y_calc, 'b-',
                 label=dataset._y_label + ' calculated')
        plt.plot(dataset._x_data, dataset._y_data, 'ro',
                 label=dataset._y_label + ' original data')
        plt.ylabel(dataset._y_label)
        plt.xlabel(dataset._x_labels[0])
    else:
        xlabel = 'y_predicted'
        ylabel = 'y_observed'
        y_pred = eval_equation(params_obj, eval_eqn_string, SRconfig)
        y_calc = y_pred / dataset._y_data
        plt.plot(dataset._y_data, y_calc, 'bo',
                 label="Prediction Error")
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
    plt.legend()
    plt.tight_layout()
    output_files = ['plot.eps', 'plot.svg', 'plot.png']
    for img in output_files:
        output_path = os.path.join(output_dir, img)
        if os.path.isfile(output_path):
            os.remove(output_path)   
        plt.savefig(output_path)


def generate_benchmarks_SRconfigs():
    '''
    Generates two SymbolicRegressionConfig objects for the generation of 100
    randomly generated symbolic regression problems.
    First SRconfig is simpler than the second in that the second permits
    functions of arity one.

    Parameters
    ----------
    None

    Returns
    -------
    result: tuple
      (SR_config1, SR_config2)      
    '''
    SR_config1 = SymbolicRegressionConfig(path_to_toy_csv, 
                                          None,
                                          n_functions=['add', 'sub', 'mul', 
                                                       'div'],
                                          f_functions=[],
                                          max_num_fit_params=5,
                                          max_permitted_trees=200)
    SR_config2 = SymbolicRegressionConfig(path_to_toy_csv, 
                                          None,
                                          n_functions=['add', 'sub', 'mul', 
                                                       'div', 'pow'], 
                                          f_functions=['sin', 'sinh', 'log'], 
                                          max_num_fit_params=5, 
                                          max_permitted_trees=200)
    result = (SR_config1, SR_config2)
    return result


def generate_benchmarks():
    '''
    Generates 100 randomly generated symbolic regression problems. The first
    twentry problems are simpler than the latter 80, in that the latter 80
    permit functions of arity one in the search space.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    ------
    Benchmark problems are saved in the `benchmarks_dir` filepath
    `benchmarks_dir` is a global variable.
    '''
    SR_config1, SR_config2 = generate_benchmarks_SRconfigs()
    # first set is from 0 - 19 inclusive
    for z in range(0, 20):
        print(ERASE_LINE + "Generating benchmark:", z, "out of:", 99, end="\r")
        generate_benchmark(str(z), SR_config1)
    for z in range(20, 100):
        print(ERASE_LINE + "Generating benchmark:", z, "out of:", 99, end="\r")
        generate_benchmark(str(z), SR_config2)
    print("Outputting a summary to ", benchmarks_summary_tsv)
    read_benchmarks()


def read_benchmarks():
    '''
    Reads the benchmark problems and generates a summary tab separated value 
    file.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    ------
    Benchmark problems' summary is saved in `benchmarks_summary_tsv` filepath
    `benchmarks_summary_tsv` is a global variable.
    '''
    with open(benchmarks_summary_tsv, 'w') as benchmarks_file:
        wrtr = csv.writer(benchmarks_file, delimiter='\t', lineterminator='\n')
        for i in range(0, 100):
            param_file = benchmarks_dir + '/' + str(i) + '_params.txt'
            with open(param_file) as pfile:
                param_file_lines = pfile.readlines()
                for line in param_file_lines:
                    if 'Simplified equation:' in line:
                        true_equation = line.replace(
                            'Simplified equation: ', '').strip()
                        true_equation = true_equation.replace(' ', '')
                    if 'Raw equation:' in line:
                        raw_equation = line.replace(
                            'Raw equation: ', '').strip()
                        raw_equation = raw_equation.replace('"', '')
                        raw_equation = raw_equation.replace(' ', '')
                row = [i, true_equation, raw_equation]
                wrtr.writerow(row)


def count_number_equations(SRconfig):
    '''
    Counts the number of possible equations in this problem. A wrapper function
    around EnumeratorBinaryTree.get_M / EnumeratorFullBinaryTree.get_M.

    Parameters
    ----------
    SRconfig: pySRURGS.SymbolicRegressionConfig
        The symbolic regression problem configuration object

    Returns
    -------
    number_possible_equations: int
    '''
    (f, n, m, cum_weights, N, dataset, enumerator, _, _) = setup(SRconfig)
    if f == 0:
        number_possible_equations = enumerator.get_M(N, n, m)
    else:
        number_possible_equations = enumerator.get_M(N, f, n, m)
    return number_possible_equations


def exhaustive_search(SRconfig):
    '''
    Runs a brute-force/exhaustive symbolic regression search against the CSV
    file. WARNING, unless you specify a very simple problem, this computation
    will indefinitely. Consider using the `count_number_equations` function
    to first determine how many equations you will be considering.

    Parameters
    ----------
    SRconfig: pySRURGS.SymbolicRegressionConfig
        The symbolic regression configuration object for this problem

    Returns
    -------
    None
       
    '''
    warning_string = """
    WARNING: you are running an exhaustive search on a large search space.
    Are you sure you want to do this?!
    """
    path_to_db = SRconfig._path_to_db
    path_to_csv = SRconfig._path_to_csv
    path_to_weights = SRconfig._path_to_weights
    num_equations = count_number_equations(SRconfig)
    print("Number of equations: ", num_equations)
    if num_equations > 50000:
        print(warning_string)        
        print("Waiting 10 seconds for user to break with ctrl-c, otherwise will run.")
        time.sleep(10)
    if (('add' not in SRconfig._n_functions and
         'sub' not in SRconfig._n_functions) or
            SRconfig._max_num_fit_params == 0):
        msg = "Exhaustive search needs `add` or `sub` and >=1 fit parameter to be truly exhaustive"
        raise Exception(msg)
    (f, n, m, _, N, dataset, enumerator, _, _) = setup(SRconfig)
    initialize_db(path_to_db)
    # we need to have two streams here, the first for the case where there are
    # functions of arity one permitted, the second for the case where there are
    # no functions of arity one permitted. Within each stream, we need to have
    # a substream for single processing and a substream for multiprocessing
    results = ResultList()
    if f > 0:  # functions of arity one present       
        for i in range(0, N):
            G = int(enumerator.get_G(f, i))
            A = int(enumerator.get_A(n, i))
            B = int(enumerator.get_B(m, i))
            for r in range(0, A):
                for s in range(0, B):
                    for q in range(0, G):
                        index_tuple = (q, r, s)
                        print("i:", i, "q:", q, "r:", r, "s:", s, end='\r')
                        check_equation_at_specified_indices_to_db(
                                                            index_tuple, i,                                                                
                                                            SRconfig)
    elif f == 0:  # functions of arity one absent
        for i in range(0, N):
            A = int(enumerator.get_A(n, i))
            B = int(enumerator.get_B(m, i))
            for r in range(0, A):
                for s in range(0, B):
                    index_tuple = (r, s)
                    print("i:", i, "r:", r, "s:", s, 
                          end='\r')
                    check_equation_at_specified_indices_to_db(
                                                        index_tuple, i,
                                                        SRconfig)
                        

def check_equation_at_specified_indices(index_tuple, i, SRconfig):
    '''
    The indices specified gets mapped into the pySRURGS enumeration scheme and
    the corresponding equation gets evaluated against the dataset.

    Parameters
    ----------
    index_tuple: tuple housing (q,r,s) or (r,s) if there are zero functions
                 of arity one

        q: int
            the index specifying which configuration of functions of arity one
            to use within [0, G-1]

        r: int
            the index specifying which configuration of functions of arity two
            to use within [0, A-1]

        s: int
            the index specifying which configuration of terminals to use within
            [0, B-1]

    i: int
        the index specifying which binary tree to consider within [0, N-1]

    SRconfig: pySRURGS.SymbolicRegressionConfig
        The symbolic regression configuration object for this problem

    Returns
    -------
    result: pySRURGS.Result or None (if floating point error)
    '''
    path_to_csv = SRconfig._path_to_csv
    path_to_db = SRconfig._path_to_db
    (f, n, m, _, N, dataset, enumerator, _, _) = setup(SRconfig)
    if len(index_tuple) == 3:
        q, r, s = index_tuple
    elif len(index_tuple) == 2:
        r, s = index_tuple
    else:
        raise Exception("Invalid length to index_tuple")
    if f > 0:
        eqn_str = equation_generator_binary_tree(i, q, r, s, enumerator, 
                                                 SRconfig)
    else:
        eqn_str = equation_generator_full_binary_tree(i, r, s, enumerator, 
                                                      SRconfig)
    try:
        try:
            simple_eqn = simplify_equation_string(eqn_str, dataset)
        except BaseException:
            pdb.set_trace()
        initialize_db(path_to_db)
        params = create_fitting_parameters(dataset._int_max_params)
        (sum_of_squared_residuals, sum_of_squared_totals,
         R2, params_fitted,
         residual) = check_goodness_of_fit(eqn_str, params, SRconfig)
        if np.isnan(R2) or np.isnan(sum_of_squared_totals):
            raise FloatingPointError
        valid = True
    except FloatingPointError:
        return None
    MSE = sum_of_squared_residuals
    result = Result(simple_eqn, eqn_str, MSE, R2, params_fitted)
    return result


def check_equation_at_specified_indices_to_queue(index_tuple, i, SRconfig, 
                                                 queue):
    '''
        runs check_equation_at_specified_indices then puts the result in a 
        queue for later push to the database
    '''
    result = check_equation_at_specified_indices(index_tuple, i, SRconfig)
    queue.put(result)
    

def check_equation_at_specified_indices_to_db(index_tuple, i, SRconfig):
    '''
        runs check_equation_at_specified_indices then puts the result in a 
        queue for later push to the database
    '''
    result = check_equation_at_specified_indices(index_tuple, i, SRconfig)
    with SqliteDict(SRconfig._path_to_db, autocommit=False) as results_dict:
        simple_eqn = result._simple_equation
        results_dict[simple_eqn] = result
        results_dict.commit()    


if __name__ == '__main__':
    # Read the doc string at the top of this script.
    # Run this script in terminal with '-h' as an argument.
    parser = argparse.ArgumentParser(
        prog='pySRURGS.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "train",
        help="absolute or relative file path to the csv file housing the training data. The rightmost column of the CSV file should be the dependent variable.")
    parser.add_argument(
        "iters",
        help="the number of equations to be attempted in this run",
        type=int)
    # TODO parser.add_argument("-test", help="absolute or relative file path
    # to the csv file housing the testing data")
    parser.add_argument(
        "-memoize_funcs",
        help="memoize the computations. If you are running large `iters` and you do not have massive ram, do not use this option.",
        action="store_true")
    parser.add_argument(
        "-count",
        help="Instead of doing symbolic regression, just count out how many possible equations for this configuration. No other processing performed.",
        action="store_true")
    parser.add_argument(
        "-benchmarks",
        help="Instead of doing symbolic regression, generate the 100 benchmark problems. No other processing performed.",
        action="store_true")
    parser.add_argument(
        "-deterministic",
        help="If set, the pseudorandom number generator will act in a predictable manner and pySRURGS will produce reproducible results.",
        action="store_true")
    parser.add_argument(
        "-plotting",
        help="plot the best model against the data to ./image/plot.png and ./image/plot.svg - note only works for univariate datasets",
        action="store_true")
    parser.add_argument(
        "-exhaustive",
        help="instead of running pure random search, do an exhaustive search. Be careful about running this as it may run forever. `iters` gets ignored.",
        action="store_true")
    parser.add_argument(
        "-funcs_arity_two",
        help="a comma separated string listing the functions of arity two you want to be considered. Permitted:add,sub,mul,div,pow",
        default=defaults_dict['funcs_arity_two'])
    parser.add_argument(
        "-funcs_arity_one",
        help="a comma separated string listing the functions of arity one you want to be considered. Permitted:sin,cos,tan,exp,log,sinh,cosh,tanh")
    parser.add_argument(
        "-max_num_fit_params",
        help="the maximum number of fitting parameters permitted in the generated models",
        default=defaults_dict['max_num_fit_params'],
        type=int)
    parser.add_argument(
        "-max_permitted_trees",
        help="the number of unique binary trees that are permitted in the generated models - binary trees define the form of the equation, increasing this number tends to increase the complexity of generated equations",
        default=defaults_dict['max_permitted_trees'],
        type=int)
    parser.add_argument(
        "-path_to_db",
        help="the absolute or relative path to the database file where we will save results. If not set, will save database file to ./db directory with same name as the csv file.",
        default=defaults_dict['path_to_db'])
    parser.add_argument(
        "-path_to_weights",
        help="the absolute or relative path to the CSV file where we store the weights for each point in the dataset. The CSV file should be a single column of non-negative numerical data without a header. If not set, weights are equal to one for all data points.",
        default=defaults_dict['path_to_weights'])
    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)
    arguments = parser.parse_args()
    single_processing_mode = True
    path_to_csv = arguments.train
    max_attempts = arguments.iters
    count_M = arguments.count
    deterministic = arguments.deterministic
    if deterministic:
        randgen.seed(0)
    benchmarks = arguments.benchmarks
    exhaustive = arguments.exhaustive
    path_to_db = arguments.path_to_db
    path_to_weights = arguments.path_to_weights
    n_funcs = arguments.funcs_arity_two
    n_funcs = n_funcs.split(',')
    n_funcs = check_validity_suggested_functions(n_funcs, 2)
    f_funcs = arguments.funcs_arity_one
    if f_funcs is None or f_funcs == '':
        f_funcs = []
    else:
        f_funcs = f_funcs.split(',')
        f_funcs = check_validity_suggested_functions(f_funcs, 1)
    plotting = arguments.plotting
    memoize_funcs = arguments.memoize_funcs
    max_num_fit_params = arguments.max_num_fit_params
    max_permitted_trees = arguments.max_permitted_trees
    SRconfig = SymbolicRegressionConfig(path_to_csv, 
                                        path_to_db, 
                                        n_funcs, 
                                        f_funcs, 
                                        max_num_fit_params, 
                                        max_permitted_trees, 
                                        path_to_weights)
    num_flags = np.sum([exhaustive, count_M, benchmarks])
    if num_flags > 1:
        msg = "You are only permitted to have one of these options:"
        msg += '[exhaustive, count_M, benchmarks]; '
        msg += 'you have ' + str(num_flags) + " of them"
        raise Exception(msg)
    if benchmarks:
        generate_benchmarks()
        exit(0)
    if count_M:
        count_number_equations(path_to_csv, SRconfig)
        exit(0)
    os.makedirs('./db', exist_ok=True)
    if exhaustive == True:
        if not single_processing_mode:
            raise("Invalid mode")
        elif single_processing_mode:
            print("Running exhaustive search in single processor mode")
            exhaustive_search(SRconfig)
    else:
        if not single_processing_mode:
            raise("Invalid mode")
        elif single_processing_mode:
            print("Running in single processor mode")
            for i in tqdm.tqdm(range(0, max_attempts)):
                uniform_random_global_search_once_to_db(None,
                                                        SRconfig)
    results_list = compile_results(SRconfig)
    if plotting:
        plot_results(SRconfig)
