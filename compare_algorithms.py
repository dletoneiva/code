from pymoo.factory import get_problem, get_reference_directions
from pymoo.util.plotting import plot
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.factory import get_performance_indicator
import numpy as np

import json


def compare_algorithms(n_iter, n_gen, n_obj, k=5, save_json=False):
    """[Runs MOEAD, NSGA2 and NSGA3 on a benchmark problem and outputs everything to a dict. Saving to json is optional.]

    Args:
        n_iter ([int]): Number of iterations to run
        n_gen ([int]): Number of generations to evaluate
        n_obj ([int]): Number of objectives for the problem
        k (int, optional): k constant for dtlz1 problem
        save_json (bool, optional): Whether the function is supposed to save to a file or not

    Returns:
        [dict]: [Dict containing performance indicators and solutions provided]
    """    
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=6)

    ref_point = np.ones(n_obj) # objective space is normalized, so [1]n is dominated by all solutions

    pop_size = ref_dirs.shape[0] # population size as a minimum to match ref dirs

    n_var = k + n_obj - 1

    problem = get_problem("dtlz1", n_obj=n_obj, n_var=n_var)

    alg1 = NSGA2(pop_size=pop_size)
    alg2 = NSGA3(ref_dirs=ref_dirs)
    alg3 = MOEAD(ref_dirs,
                n_neighbors=3,
                prob_neighbor_mating=0.7)

    hypervolume1 = []
    hypervolume2 = []
    hypervolume3 = []

    sols1 = []
    sols2 = []
    sols3 = []

    for i in range(n_iter):
        seed = np.random.randint(0, 100)

        sol1 = minimize(problem,
                        alg1,
                        ('n_gen', n_gen),
                        seed=seed,
                        verbose=False)
        sol2 = minimize(problem,
                        alg2,
                        ('n_gen', n_gen),
                        seed=seed,
                        verbose=False)
        sol3 = minimize(problem,
                        alg3,
                        ('n_gen', n_gen),
                        seed=seed,
                        verbose=False)

        # normalize
        for obj_idx in range(problem.n_obj):
            max_value1 = max(sol1.F[:, obj_idx])
            sol1.F[:, obj_idx] = sol1.F[:, obj_idx] / max_value1

            max_value2 = max(sol2.F[:, obj_idx])
            sol2.F[:, obj_idx] = sol2.F[:, obj_idx] / max_value2

            max_value3 = max(sol3.F[:, obj_idx])
            sol3.F[:, obj_idx] = sol3.F[:, obj_idx] / max_value3


        hypervolume1_temp = get_performance_indicator("hv", ref_point = ref_point)
        hypervolume1.append(hypervolume1_temp.do(sol1.F))
        sols1.append(sol1.F.tolist())
            
        hypervolume2_temp = get_performance_indicator("hv", ref_point = ref_point)
        hypervolume2.append(hypervolume2_temp.do(sol2.F))
        sols2.append(sol2.F.tolist())

        hypervolume3_temp = get_performance_indicator("hv", ref_point = ref_point)
        hypervolume3.append(hypervolume3_temp.do(sol3.F))
        sols3.append(sol3.F.tolist())

        print('Iteration ' + str(i) + ' executed.')

    output = {
        'sols1' : sols1,
        'sols2' : sols2,
        'sols3' : sols3,
        'hypervolume1' : hypervolume1,
        'hypervolume2' : hypervolume2,
        'hypervolume3' : hypervolume3
        }

    # save to a file
    if save_json:
        json_string = json.dumps(output)
        with open('output_dtlz1_'+str(n_iter)+'_iterations_'+str(n_obj)+'_objs.json','w') as outfile:
            outfile.write(json_string)

    return output

n_iter = 21
n_gen = 200

n_obj = 3
out = compare_algorithms(n_iter, n_gen, n_obj, save_json=True)

n_obj = 4
out = compare_algorithms(n_iter, n_gen, n_obj, save_json=True)

n_obj = 5
out = compare_algorithms(n_iter, n_gen, n_obj, save_json=True)

