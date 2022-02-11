import json
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

n_obj = 3, 4, 5

for nobj in n_obj:
    # read files
    with open('output_dtlz1_21_iterations_'+str(nobj)+'_objs.json') as file:
        out = json.load(file)

    # run ANOVA to check for differences in hypervolumes
    print('ANOVA test for '+str(nobj)+' objectives:')
    print(f_oneway(out['hypervolume1'], 
                   out['hypervolume2'], 
                   out['hypervolume3']))

    # run tukey test to check which ones are different

    endog = np.concatenate([out['hypervolume1'], 
                            out['hypervolume2'], 
                            out['hypervolume3']])
    
    groups = np.concatenate(
                    [np.repeat('NSGAII', repeats=len(out['hypervolume1'])), 
                     np.repeat('NSGAIII', repeats=len(out['hypervolume2'])), 
                     np.repeat('MOEA/D', repeats=len(out['hypervolume3']))])
    print('Tukey test for '+str(nobj)+' objectives:')
    print(pairwise_tukeyhsd(endog=endog,
                      groups=groups,
                      alpha=0.05))