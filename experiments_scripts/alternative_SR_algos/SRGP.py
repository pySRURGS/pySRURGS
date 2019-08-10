# Adapting https://github.com/usnistgov/pysr
# to the pySRURGS code style and datastructure
# Some improvements to performance through checking previously considered equations
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
import scoop
import lmfit
from sqlitedict import SqliteDict
import pdb
from scoop import futures
import pickle
import os
import sys
import multiprocessing as mp
import argparse
import pandas as pd
sys.path.append('./..')

# making the codes analogous 
from pySRURGS import add, mul, sub, div, pow, sin, cos, tan, log, exp, sinh, tanh, cosh
from pySRURGS import Dataset, SymbolicRegressionConfig, simplify_equation_string, eval_equation, create_fitting_parameters, initialize_db, Result, check_goodness_of_fit

# load the arguments
parser = argparse.ArgumentParser(prog='pySR.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("csv_path", help="absolute path to the csv")
parser.add_argument("path_to_db", help='path to the pickle file we will be generating')
parser.add_argument("numgens", help='number of generations for evolution', type=int)
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
        
numgens = arguments.numgens
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
SR_config = SymbolicRegressionConfig(n_functions, f_functions, int_max_params, None)
params = create_fitting_parameters(int_max_params)
np.seterr(all='raise') # need to implement the error handling

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

# dynamically create DEAP variables based on the 
# number of variables in our raw data
for i in range(0,len(dataset._x_labels)):     
    var_label = "df['" + dataset._x_labels[i] + "']"
    deap_var_name = "ARG" + str(i)
    var_assignment_command = 'pset.renameArguments(' 
    var_assignment_command += deap_var_name + '="' + var_label + '")'
    eval(var_assignment_command)

for int_param in range(0, int_max_params):# dynamically create DEAP 
    var_label = "pd['p" + str(int_param) + "']"
    deap_var_name = "ARG" + str(len(dataset._x_labels)+int_param)
    var_assignment_command = 'pset.renameArguments(' 
    var_assignment_command += deap_var_name + '="' + var_label + '")'
    eval(var_assignment_command)

#Define our 'Individual' class
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual',
               gp.PrimitiveTree,
               pset=pset,
               fitness=creator.FitnessMin,
               params=create_fitting_parameters(int_max_params))               

#Define our evaluation function

def stringify(individual):
    """Return the expression in a human readable string.
    """
    string = ""
    if type(individual) == list:
        individual = individual[0]
    stack = []
    for node in individual:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            string = prim.format(*args)
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(string)

    return string

def evaluate(individual):           
    if type(individual) == list:
        individual = individual[0]
    funcstring = str(individual)
    simple_eqn = simplify_equation_string(funcstring, dataset)
    with SqliteDict(path_to_db, autocommit=True) as results_dict:
        try: # if we have already attempted this equation, do not run again
            result = results_dict[simple_eqn]
            return (result._MSE,)
        except:
            pass
    (sum_of_squared_residuals, sum_of_squared_totals, 
    R2, fitted_params, 
    residual) = check_goodness_of_fit(individual, individual.params, dataset)
    MSE = sum_of_squared_residuals/len(dataset._y_data)
    result = Result(simple_eqn, funcstring, MSE, R2, fitted_params)
    individual.params = fitted_params
    with SqliteDict(path_to_db, autocommit=True) as results_dict:
        results_dict[simple_eqn] = result    
    return (MSE,)
  
#Construct our toolbox
toolbox = base.Toolbox()
toolbox.register('expr', gp.genGrow, pset=pset, min_=1, max_=10)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("lambdify", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=4)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.register("map",  futures.map)
    
def filter_population(population, toolbox):
    for i in range(0, len(population)):
        individual = population[i]
        try:
            evaluate(individual)
        except:
            run_bool = True
            while run_bool:
                new_indiv = toolbox.population(n=1)[0]
                try:
                    evaluate(new_indiv)
                    population[i] = new_indiv
                    run_bool = False
                except FloatingPointError:
                    pass
    return population    

def remove_tags(equation_string, dataset):
    for var in dataset._x_labels:
        var_working = "df['"+var+"']"
        equation_string = equation_string.replace(var_working, var)
    for var in range(0,int_max_params):
        var = 'p'+str(var)        
        var_working = "pd['"+var+"']"
        equation_string = equation_string.replace(var_working, var)
    return equation_string

def check_against_db(individual):
    if type(individual) == list:
        individual = individual[0]
    simple_eqn = simplify_equation_string(str(individual), dataset)
    with SqliteDict(path_to_db, autocommit=True) as results_dict:
        try: # if we have already attempted this equation, do not run again
            result = results_dict[simple_eqn]
            individual.fitness.value = result._MSE
        except:
            pass
    return individual

def display_population(population):
    for ind in population:
        print(remove_tags(str(ind), dataset))

def eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=__debug__):
    population = filter_population(population, toolbox)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals']
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    logbook.record(gen=0, nevals=len(invalid_ind))
    best_ever = None
    # Begin the generational process
    for gen in range(1, ngen + 1):        
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        offspring = filter_population(offspring, toolbox)
        # Evaluate the individuals with an invalid fitness
        offspring = list(toolbox.map(check_against_db, offspring))
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # Replace the current population by the offspring
        population[:] = offspring
        # Append the current generation statistics to the logbook
        nevals = len(invalid_ind)
        logbook.record(gen=gen, nevals=nevals)
        if best_ever is None:
            best_ever = min(population, key=lambda ind: ind.fitness.values[0])
        else:
            best = min(population, key=lambda ind: ind.fitness.values[0])
            if best_ever.fitness.values[0] < best_ever.fitness.values[0]:
                best_ever = best
        MSE = best_ever.fitness.values[0]
        NMSE = MSE / np.std(dataset._y_data)
        print(gen, nevals, NMSE, 
              simplify_equation_string(remove_tags(str(best_ever), dataset),dataset))
        #display_population(population)
    return population, logbook, best_ever

def main():    
    pop = toolbox.population(n=popsize)
    pop = filter_population(pop, toolbox)
    pop, logbook, best = eaSimple(pop,
                            toolbox,                            
                            0.7,
                            0.3,
                            numgens)    

if __name__ == "__main__":
    main()
