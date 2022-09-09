# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 20:42:42 2016

@author: Anastasis
"""

import numpy as np
import matplotlib.pyplot as plt

import proppa
import ode_simulator

def sample_dist(dist):
    try:
        return dist.rvs()
    except AttributeError:
        ind = np.random.random_integers(low=0,high=len(dist)-1)
        return dist[ind]
#   alternatively:
#    if 'rvs' in dir(dist):
#        return dist.rvs()
#    else:
#        ind = np.random.random_integers(len(dist)-1)
#        return dist[ind]
    
        
def sample_paths(model,t_final=None,dists=None,n_paths=100):
    # retrieve rate functions, initial state, stop time and updates from model
    abstract_rate_funcs = model.reaction_functions()
    updates = model.updates
    init_state = model.init_state
    if dists is None:
        print("Using priors")
        dists = [p.rhs.to_distribution() for p in model.uncertain]
    if t_final is None: # use the final observation time
        t_final = model.obs[-1][0]
    
    paths = []
    i = 0
    while i < n_paths:
        params = [sample_dist(d) for d in dists]
        rate_funcs = [r(params) for r in abstract_rate_funcs]
        path = ode_simulator.solve_odes(rate_funcs,updates,init_state,t_final,
                                        n_points=1000)
        paths.append(path)
        i = i + 1
    return paths


if __name__ == "__main__":
    # toy model and distribution
    model = proppa.load_model('mumps_uncertainer.proppa')
    model.numerize()
    t_f = 4012
    dists = list(np.loadtxt('mumps_samples_part').T)
    #paths = sample_paths(model,t_f,dists,n_paths=500)
    paths = sample_paths(model,t_f,n_paths=500)
    
#    fixed_values = [5,5,10,10]
#    dists = [spst.rv_discrete(values=(v,1)) for v in fixed_values]
#    paths = sample_paths(model,t_f,n_paths=1000,dists=dists)

    ind_I = model.species_order.index('I')    
    for p in paths:
        plt.plot(p[0],[s[ind_I] for s in p[1]])
    plt.title('Paths of I (%d ODE solutions)' % len(paths))    
    plt.show()
    
    ind_I = model.species_order.index('I')
    norm_I = [[s[ind_I] for s in sol] for (t,sol) in paths] 
    
    plot_times = paths[0][0]
    avg_path = np.average(norm_I,axis=0)
    std_path = np.std(norm_I,axis=0)

    upper_line = avg_path + std_path
#    lower_line = avg_path - std_path
    lower_line = np.maximum(avg_path - std_path,0)
    plt.plot(plot_times,avg_path,lw=2)
    plt.plot(plot_times,upper_line,'k--',lw=2)
    plt.plot(plot_times,lower_line,'k--',lw=2)
    plt.fill_between(plot_times,upper_line,lower_line,color='grey',alpha='0.5')
    plt.title('Average I (+/- 1 std)')
    plt.show()
    