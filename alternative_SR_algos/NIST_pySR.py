#https://github.com/usnistgov/pysr
import warnings
warnings.simplefilter('ignore')
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
import random
from deap import base, creator, tools, gp, algorithms
import operator
import math
import sympy
import scoop
import pdb
from scoop import futures
import pickle
import os
import sys

#Check args
if len(sys.argv) < 5:
    print("Usage: python -m scoop pysr.py csvfile numgens popsize picklefile")

    sys.exit()

#Let's read some data
import pandas as pd
df = pd.read_csv(sys.argv[1], delimiter=',')
X_names = [k for k in df.keys()[:-1]]
y_name = df.keys()[-1]
# X_names = ['Tr','rhor']
# y_name = 'bracket'
mask = ~df[y_name].isnull()
df = df[mask].copy()
df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
X = np.r_[[df.loc[:,k] for k in X_names]]
y = df.loc[:, y_name]

#So we don't have to deal with evil NaNs propagating about and whatnot
np.seterr(all='ignore')

#Define the tree elements the EA chooses from
pset = gp.PrimitiveSet('MATH', arity=X.shape[0]) #'MATH' is just a name
pset.addPrimitive(lambda x,y: np.nan_to_num(np.add(x,y)),
                  2, name='add')
pset.addPrimitive(lambda x,y: np.nan_to_num(np.subtract(x,y)),
                  2, name='sub')
pset.addPrimitive(lambda x,y: np.nan_to_num(np.multiply(x,y)),
                  2, name='mul')
pset.addPrimitive(lambda x,y: np.nan_to_num(np.divide(x,y)),
                  2, name='div')
pset.addPrimitive(lambda x,y: np.nan_to_num(np.power(np.abs(x),np.abs(y))/2),
                  2, name='pow')
pset.addEphemeralConstant('C',    #95% of the time falls within [-100, 100]
                          lambda: random.gauss(0, 50))
#pset.renameArguments(**{'ARG'+str(i):'x_'+str(i)})

#Define our 'Individual' class
creator.create('FitnessMin', base.Fitness, weights=(-1.0,-1.0))
creator.create('Individual',
               gp.PrimitiveTree,
               pset=pset,
               fitness=creator.FitnessMin)
               

#Define our evaluation function
def optimizeConstants(individual):
    #Optimize the constants
    constants = np.array(list(map(lambda n: n.value,
                              filter(lambda n: isinstance(n, gp.C),individual))))
    if constants.size > 0:
        def setConstants(individual, constants):
            optIndividual = individual
            c = 0
            for i in range(0, len(optIndividual)):
                if isinstance(optIndividual[i], gp.C):
                    optIndividual[i].value = np.nan_to_num(constants[c])
                    optIndividual[i].name = str(optIndividual[i].value)
                    c += 1
            return optIndividual

        def evaluate(constants, individual):
            individual = setConstants(individual, constants)
            func = toolbox.lambdify(expr=individual)
            diff = (func(*X) - y)
            return np.nan_to_num(diff)

        def evaluateLM(constants):
            return evaluate(constants, individual)

        res = scipy.optimize.leastsq(evaluateLM, constants)
        individual = setConstants(individual, res[0])
    return individual
    
def evaluate(individual):
    func = toolbox.lambdify(expr=individual)
    diff = np.sum((func(*X) - y)**2)/len(y)
    return np.nan_to_num(diff),len(individual)    

#Construct our toolbox
toolbox = base.Toolbox()
toolbox.register('expr', gp.genGrow, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("lambdify", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=4)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
toolbox.register("map",  futures.map)

def evolve(population, toolbox, popSize, cxpb, mutpb, ngen,
           stats=None, halloffame=None, verbose=True, pickleFile=None):
    mu = popSize; lambda_ = popSize

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    if pickleFile is not None: #if file not empty
        try:
            with open(pickleFile, "rb") as cp_file:
                cp = pickle.load(cp_file)
            population = cp["population"]
            start_gen = cp["generation"]            
            logbook = cp["logbook"]
            random.setstate(cp["rndstate"])
            print("RESTARTING FROM PICKLE")
        except IOError:
            print("Please wait, evaluating initial population...")

    def varOr(population, toolbox, lambda_, cxpb, mutpb):
        assert (cxpb + mutpb) <= 1.0, ("The sum of the crossover and mutation "
                                       "probabilities must be smaller or equal to 1.0.")
        offspring = []
        for _ in range(lambda_):
            op_choice = random.random()
            if op_choice < cxpb:            # Apply crossover
                ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
                ind1, ind2 = toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                offspring.append(ind1)
            elif op_choice < cxpb + mutpb:  # Apply mutation
                ind = toolbox.clone(random.choice(population))
                ind, = toolbox.mutate(ind)
                del ind.fitness.values
                offspring.append(ind)
            else:                           # Apply reproduction
                offspring.append(random.choice(population))
        return offspring
        
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    if halloffame is not None:
        halloffame.update(population)
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
        #Optimize the new individuals
        offspring = list(toolbox.map(optimizeConstants, offspring))
        # Evaluate the individuals with an invalid fitness
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
        # Select the next generation population
        population = (toolbox.select(population + offspring, math.floor(.99*mu)) +
                      tools.selBest(population + offspring, math.floor(.01*mu)))
        # Pickle the state
        if pickleFile is not None:
            cp = dict(population=population, generation=gen, halloffame=halloffame,
                      logbook=logbook, rndstate=random.getstate())
            with open(pickleFile, "wb") as cp_file:
                pickle.dump(cp, cp_file)
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(offspring), **record)
        if verbose:
            print( logbook.stream)
    return population, logbook

def main():
    random.seed(317)
    numgens = int(sys.argv[2])
    popsize = int(sys.argv[3])
    pickleFile=sys.argv[4]
    pop = toolbox.population(n=popsize)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("best",
                   lambda pop: min(pop, key=lambda fit: fit[0])[0])
    stats.register("len_of_best",
                   lambda pop: min(pop, key=lambda fit: fit[0])[1])       
    pop, logbook = evolve(pop,
                          toolbox,
                          popsize,
                          0.7,
                          0.3,
                          numgens,
                          stats,
                          halloffame=hof,
                          pickleFile=pickleFile)
    best = min(pop, key=lambda ind: ind.fitness.values[0])
    best_fitness = best.fitness.values[0]
    n_evals = 0    
    for entry in logbook:
        n_evals += entry['nevals']
    with open(pickleFile+'.txt', 'w') as myfile:
        myfile.write(str(['best_fitness: ', best_fitness, 'n_evals', n_evals))
    return n_evals

if __name__ == "__main__":
    main()
