#!/usr/bin/env python
doc_string = '''
pySRURGS - Symbolic Regression by Uniform Random Global Search
Sohrab Towfighi (C) 2019
License: GPL 3.0

All your data needs to be numeric. 
Your CSV file should have a header.
Inside the csv, the dependent variable should be the rightmost column.
Do not use special characters or spaces in variable names.

The config.py file defines the number of fitting parameters, the number of 
permitted binary trees through which we search, and the types of functions 
permitted in the search space.                
'''
print(doc_string)
import sympy
from sympy import simplify, sympify, Symbol
import mpmath
import sys
import lmfit
import csv 
import pdb
import re
import os
import tqdm
import parmap
import pandas
import argparse
import numpy as np
np.seterr(all='raise')
import random 
random.seed(0)
import datetime
import tabulate
from sqlitedict import SqliteDict
import collections
from itertools import repeat
import multiprocessing as mp
from config import *

def memoize(func):
    cache = dict()
    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return memoized_func

def make_timestamp():
    return '{:%Y-%b-%d.%H.%M.%S}'.format(datetime.datetime.now())

def has_nans(X):
    if np.any(np.isnan(X)) == True:
        return True
    else:
        return False
    
def check_for_nans(X):
    if has_nans(X):
        raise Exception("Has NaNs")

def binary(num, pre='', length=16, spacer=0):
    # formats a number into binary 
    return '{0}{{:{1}>{2}}}'.format(pre, spacer, length).format(bin(num)[2:])      
        
def check_file_exists(path_to_file):
    if os.path.isfile(path_to_file) == False:
        raise Exception("Missing file: " + path_to_file)
    return path_to_file

def check_dir_perms(path_to_file):
    if os.access(os.path.dirname(path_to_file), os.W_OK): 
        pass # write privileges ok 
    else: #can not write there
        raise Exception("Output dir has bad permissions: " + path_to_file)
    return path_to_file

def make_variable_name(label):
    return variable_prefix + str(label) + variable_suffix
    
def make_parameter_name(label):
    return fitting_param_prefix + str(label) + fitting_param_suffix
    
def remove_variable_tags(equation_string):
    equation_string = equation_string.replace(variable_prefix, '')
    equation_string = equation_string.replace(variable_suffix, '')
    return equation_string
    
def remove_parameter_tags(equation_string):
    equation_string = equation_string.replace(fitting_param_prefix, '')
    equation_string = equation_string.replace(fitting_param_suffix, '')
    return equation_string
    
def create_variable_list(m):
    if type(m) == str:
        my_vars = pandas.read_csv(m).keys()[:-1].tolist()
        my_vars = [make_variable_name(x) for x in my_vars]
    if type(m) == int:
        my_vars = []
        for i in range(0,m):
            my_vars.append(make_variable_name('x'+str(i)))
    return my_vars   

def create_parameter_list(m):
    my_pars = []
    for i in range(0,m):
        my_pars.append(make_parameter_name('p'+str(i)))
    return my_pars   

def get_opers_dict():
     opers_dict = {"mul":"*", 
                   "div":"/", 
                   "add":"+",
                   "sub":"-",
                   "pow":"**"} 
     opers_dict = collections.OrderedDict(opers_dict)
     opers_dict.move_to_end("mul") 
     return opers_dict

@memoize
def get_bits(x):
    # Get all even bits of x 
    even_bits = x[::2]
    # Get all odd bits of x 
    odd_bits = x[1::2]
    return odd_bits, even_bits

@memoize
def get_left_right_bits(integer):
    # splits an integer into its odd and even bits - AKA left and right bits 
    int_as_bin = binary(integer)
    left_bin, right_bin = get_bits(int_as_bin)   
    left_int =  int(left_bin, 2)
    right_int =  int(right_bin, 2)
    return left_int, right_int

def get_properties(dataframe, label):
    properties = dict()
    properties[label + '_mean'] = dataframe.mean()
    properties[label + '_std'] = dataframe.std()
    properties[label + '_min'] = dataframe.min()
    properties[label + '_max'] = dataframe.max()
    return properties

def get_scale_type(dataframe, header_labels):
    for label in header_labels:
        row = dataframe[label]
        if has_negatives(row) == True:
            return 'subtract_by_mean_divide_by_std'
    return 'divide_by_mean'

def scale_dataframe(df, scale_type):
    # does scaling by column mean    
    msg  = "This kind of scaling does not make sense when"
    msg += " your data contains negatives"
    if df.min() < 0 and scale_type == 'divide_by_mean':
        raise Exception()
    if len(df.shape) == 1:
        ncolumns = 1
        if scale_type == 'divide_by_mean':
            df = scale(df, scale_type)
    else:
        ncolumns = len(df[0,:])
        for i in range(0, ncolumns):
            if scale_type == 'divide_by_mean':
                df[:,i] = scale(df[:,i], scale_type)
    return df 
    
def scale(X, scale_type):
    if scale_type == 'divide_by_mean':
        return X/np.mean(X)
    elif scale_type == 'subtract_by_mean_divide_by_std':
        return (X-np.mean(X))/np.std(X)
    else:
        raise Exception("invalid scale type")

def str_e(my_number):
    '''
        return a number in consistent scientific notation
    '''
    return "{:.2E}".format(my_number)

def sin(x):
    return np.sin(x)

def exp(x):
    return np.exp(x)

def log(x):
    return np.log(np.abs(x))

def tanh(x):
    return np.tanh(x)

def sinh(x):
    return np.sinh(x)

def cosh(x):
    return np.cosh(x)

def sum(x):
    return np.sum(x)

def add(x, y):
    return np.add(x,y)

def sub(x, y):
    return np.subtract(x,y)

def mul(x, y):
    return np.multiply(x,y)

def div(x, y):
    return np.divide(x,y)

def pow(x, y):
    return np.power(x,y)

@memoize
def mempower(a,b):
    result = mpmath.power(a,b)    
    return result

def get_element_of_cartesian_product(*args, repeat=1, index=0):
    pools = [tuple(pool) for pool in args] * repeat 
    if len(pools) == 0:
        return []
    len_product = len(pools[0])
    len_pools = len(pools)
    for j in range(1,len_pools):
        len_product = len_product * len(pools[j])
    if index >= len_product:
        raise Exception("index + 1 is bigger than the length of the product")
    index_list = []
    for j in range(0, len_pools):
        ith_pool_index = index
        denom = 1
        for k in range(j+1, len_pools):
            denom = denom * len(pools[k])
        ith_pool_index = ith_pool_index//denom
        if j != 0:
            ith_pool_index = ith_pool_index % len(pools[j])
        index_list.append(ith_pool_index)
    ith_item = []
    for index in range(0, len_pools):
        ith_item.append(pools[index][index_list[index]])
    return ith_item

opers_dict = get_opers_dict()

def fix_order_of_fitting_parameters(funcstring):    
    # sometimes equations come out like p1 * x2 - p0
    # would prefer p0 * x2 - p1 because they are equivalent and reduces 
    # redundancy 
    # TODO - if a parameter is not in the equation, reduce the integer value 
    # corresponding with the parameters in the equation, if their integer is 
    # larger than that of the missing parameter
    param_pattern = fitting_param_prefix + '(.+?)' + fitting_param_suffix
    all_params = re.findall(param_pattern, funcstring)
    params_indexing = list()
    if len(all_params) > 0:
        for i in range(0,len(all_params)):
            current_param = all_params[i]
            if current_param not in params_indexing:
                params_indexing.append(current_param)
        for j in range(0,len(params_indexing)):
            old_name = make_parameter_name(params_indexing[j])
            temp_name = make_parameter_name('p'+str(j))
            funcstring = funcstring.replace(old_name, temp_name)
        for j in range(0,len(params_indexing)):
            temp_name = make_parameter_name('p'+str(j))
            new_name = make_parameter_name(str(j))
            funcstring = funcstring.replace(temp_name, new_name)
    replacements = {'[' : '(', 
                    ']' : ')'}                   
    for key,value in replacements.items():
        funcstring = funcstring.replace(key, value)
    return funcstring

def simplify_equation_string(eqn_str, dataset):
    z = True 
    while z == True:
        z = False
        for operator_name, operator_symbol in opers_dict.items():            
            pattern = operator_name + '\(([^,]+?), ([^,]+?)\)'
            match = re.search(pattern, eqn_str)
            if match is not None:
                z = True             
                first_argument = match.groups(0)[0]
                second_argument = match.groups(0)[1]
                replacement = '[' + first_argument + operator_symbol
                replacement += second_argument + ']'
                prefix = eqn_str[:match.start()]
                suffix = eqn_str[match.end():]
                eqn_str = prefix + replacement + suffix
    eqn_str = eqn_str.replace('[', '(')
    eqn_str = eqn_str.replace(']', ')')
    s = sympy.sympify(eqn_str, locals = dataset._sympy_namespace)
    try:
        eqn_str = str(sympy.simplify(s))
    except ValueError:
        pass
    if 'zoo' in eqn_str: # zoo (complex infinity) in sympy
        raise FloatingPointError
    eqn_str = remove_variable_tags(eqn_str)
    eqn_str = remove_parameter_tags(eqn_str)
    return eqn_str

def equation_generator(i, q, r, s, dataset, enumerator, simpler=True):
    en = enumerator
    f = len(f_functions)
    n = len(n_functions)
    m = dataset._m_terminals
    tree = ith_full_binary_tree(i)
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
    f_func_config = get_element_of_cartesian_product(f_functions, repeat=l_i, 
                                                     index=q)
    n_func_config = get_element_of_cartesian_product(n_functions, repeat=k_i, 
                                                     index=r)
    term_config = get_element_of_cartesian_product(dataset._terminals_list, 
                                                   repeat=j_i, index=s)
    orig_tree = tree
    for z in range(0,len(n_func_config)):
        func = n_func_config[z]
        tree = tree.replace('[', func +'(', 1)
        tree = tree.replace(']', ')', 1)
    for z in range(0,len(f_func_config)):
        func = f_func_config[z]
        tree = tree.replace('|', func +'(', 1)
        tree = tree.replace('|', ')', 1)
    func_tree = tree
    for z in range(0,len(term_config)):
        term = term_config[z]
        tree = tree.replace('.', term, 1)
    return tree

def equation_generator2(i, r, s, dataset, enumerator, simpler=True):
    # for the case where there are zero functions of arity one 
    en = enumerator
    n = len(n_functions)
    m = dataset._m_terminals
    tree = ith_full_binary_tree2(i)
    A = en.get_A(n, i)
    if r >= A:
        raise Exception("r is an index that must be smaller than A")
    B = en.get_B(m, i)
    if s >= B:
        raise Exception("s is an index that must be smaller than B")
    k_i = en.get_k_i(i)
    j_i = en.get_j_i(i)
    n_func_config = get_element_of_cartesian_product(n_functions, repeat=k_i, 
                                                     index=r)
    term_config = get_element_of_cartesian_product(dataset._terminals_list, 
                                                   repeat=j_i, index=s)
    orig_tree = tree
    for z in range(0,len(n_func_config)):
        func = n_func_config[z]
        tree = tree.replace('[', func +'(', 1)
        tree = tree.replace(']', ')', 1)
    func_tree = tree
    for z in range(0,len(term_config)):
        term = term_config[z]
        tree = tree.replace('.', term, 1)
    return tree
    
def random_equation(N, cum_weights, dataset, enumerator, details=False):
    n = len(n_functions)
    f = len(f_functions)
    m = dataset._m_terminals
    i = random.choices(range(0, N), cum_weights=cum_weights, k=1)[0]
    q = enumerator.get_q(f, i)
    r = enumerator.get_r(n, i)
    s = enumerator.get_s(m, i)   
    equation_string = equation_generator(i, q, r, s, dataset, enumerator, 
                                         simpler=True)
    if details == False:
        return equation_string
    else:   
        original_equation_string = equation_generator(i, q, r, s, dataset, 
                                                      enumerator, simpler=False)        
        return [original_equation_string, equation_string, N, n, f, m, i, q, r, s]
    
def random_equation2(N, cum_weights, dataset, enumerator, details=False):
    # for the case where there are zero functions of arity one 
    n = len(n_functions)
    m = dataset._m_terminals
    i = random.choices(range(0, N), cum_weights=cum_weights, k=1)[0]
    r = enumerator.get_r(n, i)
    s = enumerator.get_s(m, i)   
    equation_string = equation_generator2(i, r, s, dataset, enumerator, 
                                          simpler=True)
    if details == False:
        return equation_string
    else:   
        original_equation_string = equation_generator2(i, r, s, dataset, 
                                                       enumerator, simpler=False)
        return [original_equation_string, equation_string, N, n, m, i, r, s]

class Dataset(object):    
    def __init__(self, path_to_csv_file, int_max_params, scaled=False):
        # the independent variables will be scaled if self._scaled == True.
        self._scaled = scaled
        (dataframe, header_labels) = self.load_csv_data(path_to_csv_file)
        self._int_max_params = int_max_params
        self._scale_type = ''
        self._dataframe = dataframe
        self._header_labels = header_labels                
        x_data, x_labels, x_properties = self.get_independent_data()
        y_data, y_label, y_properties  = self.get_dependent_data()                                                                  
        self._x_data = x_data
        self._x_labels = x_labels
        self._y_data = y_data
        self._y_label = y_label
        self._data_properties = dict()
        self._data_properties.update(x_properties)
        self._data_properties.update(y_properties)
        self._data_dict = self.get_data_dict()
        self._num_variables = len(self._x_labels)
        self._m_terminals = self._num_variables + int_max_params
        self._terminals_list  = (create_parameter_list(int_max_params) + 
                                 create_variable_list(path_to_csv_file))
        self._sympy_namespace = self.make_sympy_namespace()
    def make_sympy_namespace(self):
        sympy_namespace = {}
        for variable_name in self._x_labels:
            sympy_namespace[variable_name] = sympy.Symbol(variable_name)
        param_names = [make_parameter_name(x) for x in 
                                                  range(0,self._int_max_params)]
        for param_name in param_names:
            sympy_namespace[param_name] = sympy.Symbol(param_name)
        sympy_namespace['add'] = sympy.Add
        sympy_namespace['sub'] = lambda a,b: sympy.Add(a, -b)
        sympy_namespace['mul'] = sympy.Mul
        sympy_namespace['div'] = lambda a,b: a*sympy.Pow(b,-1)
        sympy_namespace['pow'] = sympy.Pow
        return sympy_namespace
    def load_csv_data(self, path_to_csv):
        dataframe = pandas.read_csv(path_to_csv)
        column_labels = dataframe.keys()        
        return (dataframe, column_labels)
    def get_independent_data(self):
        dataframe = self._dataframe
        header_labels = self._header_labels
        features = dataframe.iloc[:,:-1]
        features = np.array(features)
        labels   = header_labels[:-1]
        properties = dict()
        for label in labels:
            properties.update(get_properties(dataframe[label], label))
        if self._scaled == True:
            self._scale_type = get_scale_type(dataframe, header_labels)
            features = scale_dataframe(features, self._scale_type)
        return (features, labels, properties)
    def get_dependent_data(self):
        dataframe = self._dataframe
        header_labels = self._header_labels
        feature = dataframe.iloc[:,-1]
        feature = np.array(feature)
        label   = header_labels[-1]
        properties = get_properties(dataframe[label], label)
        return (feature, label, properties)
    def get_data_dict(self):
        dataframe = self._dataframe
        data_dict = dict()
        for label in self._header_labels:
            data_dict[label] = np.array(dataframe[label].values).astype(float)
            check_for_nans(data_dict[label])
            if self._scaled == True and label in self._x_labels:    
                data_dict[label] = scale(data_dict[label], self._scale_type)                
        return data_dict


class Enumerator(object):
    @memoize
    def get_M(self, N, f, n, m):
        def get_f(i):
            l_i = self.get_l_i(i)
            k_i = self.get_k_i(i)
            j_i = self.get_j_i(i)
            f = mempower(n,k_i)*mempower(m,j_i)*mempower(f,l_i)
            return f
        M = nsum(get_f, [0, N-1])
        return M    
    @memoize
    def get_G(self, f, i):
        # G is the number of ways to pick l_i functions of arity 
        # one from f possible functions of arity one
        l = self.get_l_i(i)
        G = mempower(f,l)
        G = int(G)
        return G
    @memoize
    def get_A(self, n, i):
        # A is the number of ways to pick k_i functions of arity 
        # two from n possible functions of arity two
        k = self.get_k_i(i)
        A = mempower(n,k)
        A = int(A)
        return A
    @memoize
    def get_B(self, m, i):
        # B is the number of ways to pick j_i terminals from m terminals  
        j = self.get_j_i(i)   
        B = mempower(m,j)
        B = int(B)
        return B        
    def get_q(self, f, i):
        G = self.get_G(f, i)
        q = random.randint(0, G-1)
        return q
    def get_r(self, n, i):
        A = self.get_A(n, i)
        r = random.randint(0, A-1)
        return r
    def get_s(self, m, i):
        B = self.get_B(m, i)
        s = random.randint(0, B-1)
        return s
    @memoize
    def get_l_i(self, i):
        i = int(i)
        # from n functions of arity two, pick k_i 
        # k_i is the number of non-leaf nodes in the tree corresponding to i
        if i == 0:
            l_i = 0 
        elif i == 1:
            l_i = 1
        elif i == 2:
            l_i = 0
        else:
            left_int, right_int = get_left_right_bits(i)
            left_l_i = self.get_l_i(left_int)
            right_l_i = self.get_l_i(right_int)
            l_i = left_l_i + right_l_i + 1
        return l_i
    @memoize
    def get_k_i(self, i):
        i = int(i)
        # from n functions of arity two, pick k_i 
        # k_i is the number of non-leaf nodes in the tree corresponding to i
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
        i = int(i)
        # from m m_terminals, pick j_i 
        # j_i is the number of leafs in the tree corresponding to i
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

class Enumerator2(object):
    # for the case where the are zero functions of arity one
    @memoize
    def get_M(self, N, f, n, m):
        def get_f(i):
            k_i = self.get_k_i(i)
            j_i = self.get_j_i(i)
            f = mempower(n,k_i)*mempower(m,j_i)
            return f
        M = nsum(get_f, [0, N-1])
        return M    
    @memoize
    def get_A(self, n, i):
        # A is the number of ways to pick k_i functions of arity 
        # two from n possible functions of arity two
        k = self.get_k_i(i)
        A = mempower(n,k)
        A = int(A)
        return A
    @memoize
    def get_B(self, m, i):
        # B is the number of ways to pick j_i terminals from m terminals  
        j = self.get_j_i(i)
        B = mempower(m,j)
        B = int(B)
        return B        
    def get_r(self, n, i):
        A = self.get_A(n, i)
        r = random.randint(0, A-1)
        return r
    def get_s(self, m, i):
        B = self.get_B(m, i)
        s = random.randint(0, B-1)
        return s
    @memoize
    def get_k_i(self, i):
        i = int(i)
        # from n functions of arity two, pick k_i 
        # k_i is the number of non-leaf nodes in the tree corresponding to i
        if i == 0:
            k_i = 0 
        elif i == 1:
            k_i = 1
        else:
            left_int, right_int = get_left_right_bits(i)
            left_k_i = self.get_k_i(left_int)
            right_k_i = self.get_k_i(right_int)
            k_i = left_k_i + right_k_i + 1
        return k_i
    @memoize
    def get_j_i(self, i):
        i = int(i)
        # from m m_terminals, pick j_i 
        # j_i is the number of leafs in the tree corresponding to i
        if i == 0:
            j_i = 1
        elif i == 1:
            j_i = 2
        else:
            left_int, right_int = get_left_right_bits(i)
            left_j_i = self.get_j_i(left_int)
            right_j_i = self.get_j_i(right_int)
            j_i = left_j_i + right_j_i
        return j_i

def create_fitting_parameters(max_params):
    params = lmfit.Parameters()
    for int_param in range(0, max_params):
        param_name = 'p'+str(int_param)
        param_init_value=np.float(1)
        params.add(param_name, param_init_value)
    return params

def eval_equation(params, function_string, my_data, mode='residual'):
    len_data = len(my_data._y_data)
    df = my_data._data_dict
    pd = params.valuesdict()
    y_label = my_data._y_label
    independent_var_vector = df[y_label]
    residual = [BIG_NUM]*len(df[y_label])
    if mode == 'residual':
        eval_string = '(' + function_string + ') -  df["' + y_label + '"]'            
        residual = eval(eval_string)
        output = residual
    elif mode == 'y_calc':
        y_value = eval(function_string)
        output = y_value
    # if model only has parameters and no data variables, we can have a
    # situation where output is a single constant
    if np.size(output) == 1:
        output = np.resize(output, np.size(independent_var_vector))
    return output

def clean_funcstring_params(funcstring):
    funcstring = funcstring.replace(fitting_param_prefix, 'params["')
    funcstring = funcstring.replace(fitting_param_suffix, '"].value')
    return funcstring 
    
def clean_funcstring_vars(funcstring):
    funcstring = funcstring.replace(variable_prefix, 'df["')
    funcstring = funcstring.replace(variable_suffix, '"]')
    return funcstring 

def clean_funcstring(funcstring):
    funcstring = clean_funcstring_vars(funcstring)
    funcstring = clean_funcstring_params(funcstring)
    return funcstring

def check_goodness_of_fit(individual, params, my_data):
    '''
        Given the individual, the lmfit params object, and the datastructure,
        the function outputs sum_of_squared_residuals, sum_of_squared_totals, 
        params_dict_to_store         
    '''
    # If funcstring is a tree, transform to string
    funcstring = str(individual)
    funcstring = clean_funcstring(funcstring)
    # Evaluate the sum of squared difference between the expression
    result = lmfit.minimize(eval_equation, params, 
                            args=(funcstring, my_data),
                            method='leastsq', nan_policy='propagate')
    if result.success == False:
        result = lmfit.minimize(eval_equation, params, 
                            args=(funcstring, my_data),
                            method='nelder', nan_policy='propagate')
    sum_of_squared_residuals = sum(pow(result.residual, 2))        
    avg_y_data = np.average(my_data._y_data)
    y_calc = eval_equation(result.params, funcstring, my_data, mode='y_calc')        
    sum_of_squared_residuals = sum(pow(my_data._y_data - y_calc,2))
    sum_of_squared_totals = sum(pow(y_calc - avg_y_data,2))
    params_dict_to_store = result.params
    residual = result.residual
    R2 = 1 - sum_of_squared_residuals/sum_of_squared_totals
    return (sum_of_squared_residuals, 
            sum_of_squared_totals, 
            R2,
            params_dict_to_store, 
            residual)

@memoize
def ith_full_binary_tree(i):    
    if i == 0:
        tree = '.'
    elif i == 1:
        tree = '[., .]'
    elif i == 2:
        tree = '|.|'
    else:
        left_int, right_int = get_left_right_bits(i)
        left = ith_full_binary_tree(left_int)
        right = ith_full_binary_tree(right_int)
        tree = '[' + left +', ' +right + ']'
    return tree

@memoize
def ith_full_binary_tree2(i):
    # for the cases where there are zero functions of arity two
    if i == 0:
        tree = '.'
    elif i == 1:
        tree = '[., .]'
    else:
        left_int, right_int = get_left_right_bits(i-1)
        left = ith_full_binary_tree2(left_int)
        right = ith_full_binary_tree2(right_int)
        tree = '[' + left +', ' +right + ']'
    return tree

def print_some_trees(nn):
    trees = []
    en = Enumerator()
    for i in range(0,nn):
        tree = ith_full_binary_tree(i)
        k_i = en.get_k_i(i)
        j_i = en.get_j_i(i)
        print(i, get_left_right_bits(i), j_i, k_i, tree)

def print_some_trees2(nn):
    trees = []
    en = Enumerator2()
    for i in range(0,nn):
        tree = ith_full_binary_tree2(i)
        try:
            [left, right] = get_left_right_bits(i-1)           
        except:
            [left, right] = [np.nan, np.nan]
        k_i = en.get_k_i(i)
        j_i = en.get_j_i(i)
        print(i, [left,right], j_i, k_i, tree)

@memoize    
def get_cum_weights(N, f, n, m, enumerator):
    en = enumerator
    weights = [en.get_G(f, i)*en.get_A(n, i)*en.get_B(m, i) for i in range(0,N)]
    weights = np.array(weights)
    cum_weights = np.array(weights)/np.sum(weights)
    return cum_weights

@memoize
def get_cum_weights2(N, n, m, enumerator):
    # for the case where there are zero functions of arity one
    en = enumerator
    weights = [en.get_A(n, i)*en.get_B(m, i) for i in range(0, N)]
    weights = np.array(weights)
    cum_weights = np.array(weights)/np.sum(weights)
    return cum_weights

class ResultList(object):
    def __init__(self):
        self._results = []
    def sort(self):
        self._results = sorted(self._results, key=lambda x: x._MSE)
    def print(self, y_data, top=5):
        table = []
        header = ["Normalized Mean Squared Error", "R^2", "Equation, simplified", 
                  "Parameters"]
        for i in range(0, top):
            row = self._results[i].summarize()
            row[0] = row[0]/np.std(y_data)
            table.append(row)
        table_string = tabulate.tabulate(table, headers=header)
        print(table_string)

class Result(object):
    def __init__(self, simple_equation, equation, MSE, R2, params):
        self._simple_equation = simple_equation
        self._equation = equation
        self._MSE = MSE 
        self._R2 = R2 
        self._params = np.array(params)
    def print(self):        
        print(str_e(self._MSE), str_e(self._R2), self._simple_equation)
    def summarize(self):
        summary = [self._MSE, self._R2, self._simple_equation]
        parameters = []
        for param in self._params:
            parameters.append(str_e(param))
        parameters_str = ','.join(parameters)
        summary.append(parameters_str)
        return summary
    def predict(self, dataset):
        pass
        #TODO
        #parameters = create_parameter_list(dataset._m_terminals)
        #for i in range(0,dataset._m_terminals):
        #    parameters['p'+str(i)].value = fitting_parameters[i]
        #y_calc = eval_equation(fit_param_list, eqn_original_cleaned, dataset, mode='y_calc')
        

def initialize_db(path_to_db):
    with SqliteDict(path_to_db, autocommit=True) as results_dict:
        try:
            results_dict['best_result']
        except KeyError:
            results_dict['best_result'] = Result(None, None, np.inf, None, None)
    return 

def uniform_random_global_search_once(path_to_db, path_to_csv): 
    (f, n, m, cum_weights, N, dataset, enumerator) = setup(path_to_csv)
    valid = False
    while valid == False:
        if f == 0:
            eqn_str = random_equation2(N, cum_weights, dataset, enumerator)
        else:    
            eqn_str = random_equation(N, cum_weights, dataset, enumerator)
        try:
            simple_eqn = simplify_equation_string(eqn_str, dataset)
        except FloatingPointError:
            pass
        initialize_db(path_to_db)    
        with SqliteDict(path_to_db, autocommit=True) as results_dict:
            try: # if we have already attempted this equation, do not run again
                result = results_dict[simple_eqn]
                return result
            except:
                pass
        params = create_fitting_parameters(dataset._int_max_params)
        try:
            (sum_of_squared_residuals, 
                sum_of_squared_totals, 
                R2,
                params_fitted,
                residual) = check_goodness_of_fit(eqn_str, params, dataset)
            valid = True
        except FloatingPointError:
            pass
    MSE = sum_of_squared_residuals
    result = Result(simple_eqn, eqn_str, MSE, R2, params_fitted)
    with SqliteDict(path_to_db, autocommit=True) as results_dict:
        best_result = results_dict['best_result']             
        results_dict[simple_eqn] = result
        if result._MSE < best_result._MSE:                       
            results_dict['best_result'] = best_result            
    return result

def generate_benchmark(path_to_csv, benchmark_name):
    # x_domain is [lower_bound, upper_bound]
    (f, n, m, cum_weights, N, dataset, enumerator) = setup(path_to_csv)
    valid = False
    while valid == False:
        try:
            # specify the equation
            if f == 0:
                eqn_details = random_equation2(N, cum_weights, 
                                               dataset, enumerator, 
                                               details=True)
            else:    
                eqn_details = random_equation(N, cum_weights, 
                                              dataset, enumerator, 
                                              details=True)
            eqn_original = eqn_details[0]
            eqn_simple = simplify_equation_string(eqn_original, dataset)
            eqn_specifiers = eqn_details[2:]
            # create the fitting parameters values
            fitting_parameters = (np.random.sample((5))-0.5)*20
            fit_param_list = create_fitting_parameters(5)
            for i in range(0,5):
                fit_param_list['p'+str(i)].value = fitting_parameters[i]
            eqn_original_cleaned = clean_funcstring(eqn_original)
            # create the training dataset
            for zz in range(0,2): # 1st iteration, train. 2nd iteration, test.                
                sample_dataset = np.zeros((100,6))
                for j in range(0,5):
                    sample_dataset[:,j] = np.random.sample((100,))*10
                dataset._dataframe = pandas.DataFrame(sample_dataset)
                dataset._dataframe.columns = dataset._x_labels.tolist() + [dataset._y_label]
                dataset._data_dict = dataset.get_data_dict()
                # calculate y for the sample problem 
                y_calc = eval_equation(fit_param_list, eqn_original_cleaned, dataset, mode='y_calc')
                dataset._dataframe[dataset._y_label] = y_calc
                # save the test and train sets to file 
                if zz == 0:
                    path = benchmarks_dir + '/' + benchmark_name + '_train.csv'
                else:
                    path = benchmarks_dir + '/' + benchmark_name + '_test.csv'
                dataset._dataframe.to_csv(path, index=False)
                path = benchmarks_dir + '/' + benchmark_name + '_params.txt'
                # save the problem parameters to a text file 
                with open(path, "w") as text_file:
                    msg = 'Permitted variables: ' + str(dataset._x_labels.tolist()) + '\n'
                    msg += 'Permitted fitting parameters: ' + str(list(fit_param_list.keys())) + '\n'
                    msg += 'Fitting parameters: ' 
                    msg += str(np.array(fit_param_list)) + '\n'                    
                    msg += 'Permitted functions: ' + str(f_functions + n_functions) + '\n'
                    msg += 'Simplified equation: ' + str(eqn_simple) + '\n'
                    eqn_original = remove_variable_tags(eqn_original)
                    eqn_original = remove_parameter_tags(eqn_original)
                    msg += 'Raw equation: ' + str(eqn_original) + '\n'
                    text_file.write(msg)
            valid = True
        except FloatingPointError:
            pass
    
def setup(path_to_csv):
    # reads the configuration, the csv file, and creates needed objects
    N = MAX_NUM_TREES
    m = MAX_NUM_FIT_PARAM
    n = len(n_functions)
    f = len(f_functions)        
    if f == 0:
        enumerator = Enumerator2()
        cum_weights = get_cum_weights2(N, n, m, enumerator)
    else:
        enumerator = Enumerator()
        cum_weights = get_cum_weights(N, f, n, m, enumerator)
    dataset = Dataset(path_to_csv, m, scaled=False)
    return (f, n, m, cum_weights, N, dataset, enumerator)

def create_db(path_to_csv):
    csv_basename = os.path.basename(path_to_csv)
    csv_name = csv_basename[:-4]
    db_name = './db/' + csv_name + '.sqlite'    
    return db_name

def compile_results(path_to_db, path_to_csv):
    # reads the generated .sqlite file to determine the best models, then 
    # prints them to screen!
    (_, _, _, _, _, dataset, _) = setup(path_to_csv)
    result_list = ResultList()
    with SqliteDict(path_to_db, autocommit=True) as results_dict:
        keys = results_dict.keys()
        for eqn in keys:
            result = results_dict[eqn]
            result_list._results.append(result)
    result_list.sort()    
    result_list.print(dataset._y_data)
    return result_list

def generate_benchmarks(path_to_csv, start_num, count):
    # make sure to configure config.py before running this, as your generated 
    # benchmarks will use your configuration
    for z in range(start_num, start_num+count):
        generate_benchmark(path_to_csv, str(z))

def read_benchmarks():
    with open('./benchmarks_summary.tsv', 'w') as benchmarks_file:
        wrtr = csv.writer(benchmarks_file, delimiter='\t', lineterminator='\n')
        for i in range(0,100):
            param_file = benchmarks_dir + '/' + str(i) + '_params.txt'
            with open(param_file) as pfile:
                param_file_lines = pfile.readlines()
                for line in param_file_lines:
                    if 'Simplified equation:' in line:
                        true_equation = line.replace('Simplified equation: ', '').strip()
                        true_equation = true_equation.replace(' ', '')
                    if 'Raw equation:' in line:
                        raw_equation = line.replace('Raw equation: ', '').strip()
                        raw_equation = raw_equation.replace('"', '')
                        raw_equation = raw_equation.replace(' ', '')
                row = [i,true_equation,raw_equation]
                wrtr.writerow(row)

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(prog='pySRURGS.py')
    parser.add_argument("train", help="absolute or relative file path to the csv file housing the training data")
    parser.add_argument("iters", help="the number of equations to be attempted in this run", type=int)
    parser.add_argument("-test", help="absolute or relative file path to the csv file housing the testing data")
    parser.add_argument("-single", help="run in single processing mode", action="store_true")
    #sys.argv includes a list of elements starting with the program
    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)
    parser.parse_args()
    args = parser.parse_args()
    single_processing_mode = args.single
    path_to_csv = args.train
    max_attempts = args.iters
    #
    generate_benchmarks(path_to_toy_csv, 0, 20)
    exit(0)
    path_to_db = create_db(path_to_csv)
    os.makedirs('./db', exist_ok=True) 
    if single_processing_mode == False:
        print("Running in multi processor mode")
        results = parmap.map(uniform_random_global_search_once, 
                                   [path_to_db]*max_attempts, 
                                   path_to_csv, pm_pbar=True)
    elif single_processing_mode == True:
        print("Running in single processor mode")
        for i in tqdm.tqdm(range(0,max_attempts)):
            uniform_random_global_search_once(path_to_db, path_to_csv)            
    else:
        raise("Invalid mode")
    compiled_results = compile_results(path_to_db, path_to_csv)
    best_model = compiled_results._results[0]
    