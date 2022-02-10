from matplotlib.pyplot import tight_layout
import numpy as np
import json
from pymoo.visualization.scatter import Scatter

def plot_solutions(n_obj):
    """Plots the solutions from the compared algorithms

    Args:
        n_obj ([int]): Number of objectives to be plotted.
    """    

    # read files
    with open('output_dtlz1_21_iterations_'+str(n_obj)+'_objs.json') as file:
        out = json.load(file)

    # scatter accepts only numpy arrays
    obj_nsgaii = np.array([np.array(xi) for xi in out['sols1']])
    obj_nsgaiii = np.array([np.array(xi) for xi in out['sols2']])
    obj_moead = np.array([np.array(xi) for xi in out['sols3']])

    # add all solutions to a plot
    plot = Scatter(tight_layout=True)
    for sol in range(len(obj_nsgaii)):
        plot.add(obj_nsgaii[sol], facecolor="none", edgecolor="green")
    plot.show()

    plot = Scatter(tight_layout=True)
    for sol in range(len(obj_nsgaiii)):
        plot.add(obj_nsgaiii[sol], facecolor="none", edgecolor="blue")
    plot.show()

    plot = Scatter(tight_layout=True)
    for sol in range(len(obj_moead)):
        plot.add(obj_moead[sol], facecolor="none", edgecolor="red")
    plot.show()

plot_solutions(5)