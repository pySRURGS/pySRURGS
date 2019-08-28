'''
    This script is only tested on Linux. Windows is problematic with the 
    multiprocessing module.
    
'''
import warnings
warnings.simplefilter('ignore')
import scipy
import scipy.optimize
import numpy as np
import random
from deap import base, creator, tools, gp, algorithms
import operator
import math
import sympy
import lmfit
from sqlitedict import SqliteDict
import pdb
import pickle
import os
import sys
import argparse
import pandas as pd
sys.path.append('./..')
import pySRURGS
import random
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap.algorithms import *

# making the codes analogous 
from pySRURGS import (add, mul, sub, div, pow, sin, cos, tan,
                      log, exp, sinh, tanh, cosh,
                      Dataset, SymbolicRegressionConfig, eval_equation,
                      simplify_equation_string, create_fitting_parameters,
                      create_parameter_list, create_variable_list,
                      initialize_db, check_goodness_of_fit, Result,
                      clean_funcstring, remove_dict_tags, assign_n_evals,
                      BIG_NUM)

def eaSimple(population, toolbox, cxpb, mutpb, goal_total_evals, stats=None,
             halloffame=None, verbose=__debug__):
    """Custom version with number of evaluations checked.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    total_evals = pySRURGS.assign_n_evals(path_to_db)
    if total_evals > goal_total_evals:
        print("Already exceeded goal_total_evals")
        return
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    gen = 0
    total_evals = pySRURGS.assign_n_evals(path_to_db)
    print(total_evals, goal_total_evals)
    while total_evals < goal_total_evals:        
        gen = gen + 1
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        total_evals = pySRURGS.assign_n_evals(path_to_db)
        print(total_evals, goal_total_evals)
    return population, logbook

def make_pset(dataset, int_max_params, csv_path, n_functions, f_functions):
    #Define the tree elements the EA chooses from
    pset = gp.PrimitiveSet('MATH', len(dataset._x_labels)+int_max_params)
    if 'add' in n_functions:
        pset.addPrimitive(add, 2, name='add')
    if 'sub' in n_functions:
        pset.addPrimitive(sub, 2, name='sub')
    if 'mul' in n_functions:
        pset.addPrimitive(mul, 2, name='mul')
    if 'div' in n_functions:
        pset.addPrimitive(div, 2, name='div')
    if 'pow' in n_functions:
        pset.addPrimitive(pow, 2, name='pow')
    if 'sin' in f_functions:
        pset.addPrimitive(sin, 1, name='sin')
    if 'cos' in f_functions:
        pset.addPrimitive(cos, 1, name='cos')
    if 'tan' in f_functions:
        pset.addPrimitive(tan, 1, name='tan')
    if 'exp' in f_functions:
        pset.addPrimitive(exp, 1, name='exp')
    if 'log' in f_functions:
        pset.addPrimitive(log, 1, name='log')
    if 'sinh' in f_functions:
        pset.addPrimitive(sinh, 1, name='sinh')
    if 'cosh' in f_functions:
        pset.addPrimitive(cosh, 1, name='cosh')
    if 'tanh' in f_functions:
        pset.addPrimitive(tanh, 1, name='tanh')
    # dynamically create DEAP variables based on the var names and parameter names 
    var_names = create_variable_list(csv_path)
    for i in range(0,len(var_names)):     
        var_label = var_names[i]
        deap_var_name = "ARG" + str(i)
        var_assignment_command = 'pset.renameArguments(' 
        var_assignment_command += deap_var_name + '="' + var_label + '")'
        eval(var_assignment_command)
    param_names = create_parameter_list(int_max_params)
    for i in range(0, int_max_params):
        deap_var_name = "ARG" + str(len(var_names) + i)
        var_assignment_command = 'pset.renameArguments(' 
        var_assignment_command += deap_var_name + '="' + param_names[i] + '")'
        eval(var_assignment_command)
    return pset

#Define our evaluation function
def evaluate(individual, dataset):
    # replace invalid equations with a random new equation
    if type(individual) == list:
        individual = individual[0]
    funcstring = str(individual)
    funcstring_dict_form = pySRURGS.clean_funcstring(funcstring)
    funcstring_readable_form = pySRURGS.remove_dict_tags(funcstring_dict_form)        
    try:
        simple_eqn = pySRURGS.simplify_equation_string(funcstring_readable_form, dataset)
        with SqliteDict(path_to_db, autocommit=True) as results_dict:
            try: # if we have already attempted this equation, do not run again
                result = results_dict[simple_eqn]
                MSE = result._MSE
                return (MSE,)
            except KeyError:                    
                (sum_of_squared_residuals, 
                sum_of_squared_totals, 
                R2, fitted_params, 
                residual) = pySRURGS.check_goodness_of_fit(funcstring_dict_form, 
                                                           params, dataset)    
                MSE = sum_of_squared_residuals/len(dataset._y_data)
    except (FloatingPointError):
        MSE = pySRURGS.BIG_NUM
        return (MSE,) # don't save these results to the db
    result = pySRURGS.Result(simple_eqn, funcstring, MSE, R2, fitted_params)
    with SqliteDict(path_to_db, autocommit=True) as results_dict:
        results_dict[simple_eqn] = result 
    return (MSE,)

def mean(x):    
    np.seterr('ignore')
    return np.nan_to_num(np.mean(x))
    
def std(x):
    np.seterr('ignore')
    return np.nan_to_num(np.std(x))

def main():
    pop = toolbox.population(n=popsize)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", mean)
    mstats.register("std", std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = eaSimple(pop, toolbox, 0.7, 0.3, goal_total_evals, stats=mstats,
                                   halloffame=hof, verbose=True)
    return pop, log, hof

if __name__ == "__main__":
    # load the arguments
    parser = argparse.ArgumentParser(prog='pySR.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("csv_path", help="absolute path to the csv")
    parser.add_argument("path_to_db", help='path to the pickle file we will be generating')
    parser.add_argument("goal_total_evals", help='number of individuals evaluated', type=int)
    parser.add_argument("popsize", help='number of individuals in the population', type=int)
    parser.add_argument("int_max_params", help='the maximum number of fitting parameters in the model', type=int)
    parser.add_argument("n_functions", help='comma delimited list of functions of arity 2 to be used eg: add,pow,sub,mul,div')
    parser.add_argument("-f_functions", help='comma delimited list of functions of arity 1 to be used eg: sin,cos,tan,exp,log')
    arguments = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)
    # assign arugments to the namespace
    csv_path = arguments.csv_path
    path_to_db = arguments.path_to_db
    initialize_db(path_to_db)
    goal_total_evals = arguments.goal_total_evals
    popsize = arguments.popsize
    n_functions = arguments.n_functions
    n_functions = n_functions.split(',')
    f_functions = arguments.f_functions
    if f_functions is None:
        f_functions = []
    else:    
        f_functions = f_functions.split(',')
    int_max_params = arguments.int_max_params
    #Let's read some data
    dataset = Dataset(csv_path, int_max_params)
    # max_permitted_trees assigned a placeholder value
    SR_config = SymbolicRegressionConfig(n_functions, f_functions, int_max_params, 1)
    params = create_fitting_parameters(int_max_params)
    pset = make_pset(dataset, int_max_params, csv_path, n_functions, f_functions)    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evaluate, dataset=dataset)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    #pool = mp.Pool()
    #toolbox.register("map", pool.map)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    main()
    pySRURGS.compile_results(path_to_db, csv_path, SR_config)