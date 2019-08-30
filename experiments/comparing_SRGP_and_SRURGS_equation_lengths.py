from SRGP import *

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
    
    lengths_1 = []
    lengths_2 = []
    lengths_3 = []
    
    for i in range(0,1000):
        tree_SRURGS_arity_1_2 = pySRURGS.ith_full_binary_tree(i)
        length_tree_SRURGS_arity_1_2 = len(tree_SRURGS_arity_1_2)
        lengths_1.append(length_tree_SRURGS_arity_1_2)
        tree_SRURGS_arity_2 = pySRURGS.ith_full_binary_tree2(i)
        length_tree_SRURGS_arity_2 = len(tree_SRURGS_arity_2)
        lengths_2.append(length_tree_SRURGS_arity_2)
        
    for i in range(0,17):
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
        deap_individual = toolbox.population(n=1)[0]
        length_deap_individual = len(str(deap_individual))
        lengths_3.append(length_deap_individual)