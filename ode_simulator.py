# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 11:32:33 2016

@author: Anastasis
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import proppa

def solve_odes(rate_funcs,updates,init_state,t_final,
               n_points=500):
    
    def _dydt(state,t):
        rates = np.array([rf(state) for rf in rate_funcs])
        return rates.dot(updates) 

    times = np.linspace(0,t_final,n_points)
    sols = odeint(_dydt,np.array(init_state),times)
    return (times,sols)


def solve_odes_inhomog(rate_funcs,updates,init_state,t_final,
               n_points=500):
    """ To be used with inhomogeneous models, which require time as an
    additional argument to the rate function """
    def _dydt(state,t):
        args = np.append(state,t)
        rates = np.array([rf(args) for rf in rate_funcs])
        return rates.dot(updates) 

    times = np.linspace(0,t_final,n_points)
    sols = odeint(_dydt,np.array(init_state),times)
    return (times,sols)    

if __name__ == "__main__":
#    location = "SIR_uncertain.proppa"
#    model = proppa.load_model(location)
#    model.numerize()
#    abstract_rates = model.reaction_functions()
#    params = [0.4,0.5]
#    rate_funcs = [f(params) for f in abstract_rates]
##    model.concretise({'r_i':0.4, 'r_r':0.5})
##    rate_funcs = [rf([]) for rf in model.reaction_functions()]
#    init_state = model.init_state
#    updates = model.updates
#    t_final = 5
#    (t,sol) = solve_odes(rate_funcs,updates,init_state,t_final)
#    
#    obs = []
#    with open(model.obsfile) as obsfile:
#        names = [tok.strip() for tok in obsfile.readline().split(" ")] #names
#        for line in obsfile:
#            toks = [float(tok.strip()) for tok in line.split(" ")]
#            obs.append(toks)    
#    
#    for i in range(len(model.species_order)):
#        plt.plot(t,[s[i] for s in sol])
#    for o in obs:
#        for i in range((len(model.species_order))):
#            plt.plot(o[0],o[i+1],'x')
#    plt.legend(model.species_order,loc="lower right")    
#    plt.show()
    
    location = "mumps.proppa"
    model = proppa.load_model(location)
    model.numerize()
    abstract_rates = model.reaction_functions()
    params = []
    rate_funcs = [f(params) for f in abstract_rates]
    init_state = model.init_state
    updates = model.updates
    t_final = 18000
    #t_final = 4012
    (t,sol) = solve_odes(rate_funcs,updates,init_state,t_final)
    to_plot = ['I']
    plt.hold(True)
    for species in to_plot:
        plt.plot(t,[s[model.species_order.index(species)] for s in sol])
    #plt.legend(to_plot,loc="lower right")
    #obs = model.obs
    #plt.plot([o[0] for o in obs],[o[1] for o in obs],'r')
#    plt.show()
    
    