from pymoo.factory import get_problem, get_reference_directions
from pymoo.util.plotting import plot
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_performance_indicator


import json

pop_size = 100
n_gen = 200
n_iter = 200

problem = get_problem("dtlz1") # 3 objs, 7 vars, easy pareto
# problem = get_problem("dtlz7") # 3 objs, 10 vars, easy pareto

ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

alg1 = NSGA2(pop_size=pop_size)
alg2 = NSGA3(pop_size=pop_size,
             ref_dirs=ref_dirs)
alg3 = MOEAD(ref_dirs,
             n_neighbors=15,
             prob_neighbor_mating=0.7)

timing1 = []
timing2 = []
timing3 = []

gd1 = []
gd2 = []
gd3 = []

for i in range(n_iter):
    sol1 = minimize(problem,
                    alg1,
                    ('n_gen', n_gen),
                    seed=42,
                    verbose=False)

    sol2 = minimize(problem,
                    alg2,
                    ('n_gen', n_gen),
                    seed=42,
                    verbose=False)

    sol3 = minimize(problem,
                    alg3,
                    ('n_gen', n_gen),
                    seed=42,
                    verbose=False)

    timing1.append(sol1.exec_time)
    timing2.append(sol2.exec_time)
    timing3.append(sol3.exec_time)

    gd1_temp = get_performance_indicator("gd", problem.pareto_front())
    gd1.append(gd1_temp.do(sol1.F))

    gd2_temp = get_performance_indicator("gd", problem.pareto_front())
    gd2.append(gd2_temp.do(sol2.F))

    gd3_temp = get_performance_indicator("gd", problem.pareto_front())
    gd3.append(gd3_temp.do(sol3.F))

    print('Iteration ' + str(i) + ' executed.')

output = {'timings_nsga_ii': timing1,
          'gd_nsga_ii': gd1,
          'timings_nsga_iii': timing2,
          'gd_nsga_iii': gd2,
          'timings_moead': timing3,
          'gd_moead': gd3}

json_string = json.dumps(output)

with open('output_dtlz1.json','w') as outfile:
    outfile.write(json_string)

problem = get_problem("dtlz7") # 3 objs, 10 vars, easy pareto

alg1 = NSGA2(pop_size=pop_size)
alg2 = NSGA3(pop_size=pop_size,
             ref_dirs=ref_dirs)
alg3 = MOEAD(ref_dirs,
             n_neighbors=15,
             prob_neighbor_mating=0.7)

timing1 = []
timing2 = []
timing3 = []

gd1 = []
gd2 = []
gd3 = []

for i in range(n_iter):
    sol1 = minimize(problem,
                    alg1,
                    ('n_gen', n_gen),
                    seed=42,
                    verbose=False)

    sol2 = minimize(problem,
                    alg2,
                    ('n_gen', n_gen),
                    seed=42,
                    verbose=False)

    sol3 = minimize(problem,
                    alg3,
                    ('n_gen', n_gen),
                    seed=42,
                    verbose=False)

    timing1.append(sol1.exec_time)
    timing2.append(sol2.exec_time)
    timing3.append(sol3.exec_time)

    gd1_temp = get_performance_indicator("gd", problem.pareto_front())
    gd1.append(gd1_temp.do(sol1.F))

    gd2_temp = get_performance_indicator("gd", problem.pareto_front())
    gd2.append(gd2_temp.do(sol2.F))

    gd3_temp = get_performance_indicator("gd", problem.pareto_front())
    gd3.append(gd3_temp.do(sol3.F))

    print('Iteration ' + str(i) + ' executed.')

output = {'timings_nsga_ii': timing1,
          'gd_nsga_ii': gd1,
          'timings_nsga_iii': timing2,
          'gd_nsga_iii': gd2,
          'timings_moead': timing3,
          'gd_moead': gd3}

json_string = json.dumps(output)

with open('output_dtlz7.json','w') as outfile:
    outfile.write(json_string)



# plot = Scatter()
# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(sol1.F, facecolor="none", edgecolor="red")
# plot.show()





# plot2 = Scatter()
# plot2.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot2.add(sol2.F, facecolor="none", edgecolor="red")
# plot2.show()
