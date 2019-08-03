'''
    This configuration file changes the nature of the search space
    If you do not know what you are doing, avoid editing it.
    - Sohrab Towfighi
'''

# you can edit n_functions to change the functions you permit in the search
# functions of arity two:
n_functions = ['add','sub','mul','div','pow'] 
# functions of arity one:
f_functions = ['sum']#['sin','exp','tanh']
# a very big number!
BIG_NUM = 1.79769313e+300
# prefix and suffix variables and parameters thinking I would do string 
# manipulations, but this may end up being needless.
fitting_param_prefix = 'begin_fitting_param_'
fitting_param_suffix = '_end_fitting_param'
variable_prefix = 'begin_variable_'
variable_suffix = '_end_variable'
