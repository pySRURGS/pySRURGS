'''
    This configuration file changes the nature of the search space
    If you do not know what you are doing, avoid editing it.
    - Sohrab Towfighi
'''

# you can edit n_functions to change the functions you permit in the search
# functions of arity two:
n_functions = ['add','sub','mul','div','pow'] 
# functions of arity one:
f_functions = []#['sin','exp','tanh'] # it is permissible to set f_functions to an empty list.
# the maximum number of fitting parameters in candidate solutions
MAX_NUM_FIT_PARAM = 4
# the number of unique binary trees that are considered in the search space 
MAX_NUM_TREES = 1000

#### Don't edit after this.
# a very big number!
BIG_NUM = 1.79769313e+300
# prefix and suffix variables and parameters thinking I would do string 
# manipulations, but this may end up being needless.
fitting_param_prefix = 'begin_fitting_param_'
fitting_param_suffix = '_end_fitting_param'
variable_prefix = 'begin_variable_'
variable_suffix = '_end_variable'
