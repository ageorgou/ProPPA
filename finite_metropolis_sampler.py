# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 10:48:14 2015

@author: Anastasis
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 15:22:41 2015

@author: Anastasis
"""

import numpy as np
from numpy import inf
import scipy.stats as spst

from utilities import parameterise_rates,make_statespace,make_generator2
from utilities import find_states, transient_prob
from model_utilities import load_observations, get_updates
from mh import MetropolisSampler
import proppa

class FiniteMetropolisSampler(MetropolisSampler):
    #required_conf = ['proposal'] # same as superclass
    
    def __init__(self,model,conf):
        self.set_model(model)
        if conf is not None:
            self.apply_configuration(conf)
        self.n_pars = len(self.priors)
        self.state = tuple(d.rvs() for d in self.priors)
        self.samples = []
        self.current_prior = np.prod([p.pdf(v) \
            for (p,v) in zip(self.priors,self.state)])
        self.current_L = self.calculate_likelihood(self.state)
    
    def set_model(self,model):
        self.model = model
        self.obs,_ = load_observations(model.obsfile)
        #self.obs = np.array(self.obs)
        self.updates = model.updates
        #self.space = make_statespace(self.updates,self.obs[:,1:])
        self.space = make_statespace(self.updates,
                                     [tuple(o[1:]) for o in self.obs])
    
    
    def calculate_likelihood(self,pars):
        rfs = parameterise_rates(self.rate_funcs,pars)
        space = self.space
        Q = make_generator2(self.space,rfs,self.updates)
        inds = find_states([tuple(o[1:]) for o in self.obs],space)
        L = 1
        i = 0
        while i < len(self.obs) - 1:
            init_prob = np.zeros(len(space))
            init_prob[inds[i]] = 1
            Dt = self.obs[i+1][0] - self.obs[i][0]
            final_prob = transient_prob(Q,Dt,init_prob)
            L = L * final_prob[inds[i+1]]
            i = i + 1
        return L
    


if __name__ == "__main__":
    # create SIR model
    species_names = ('S','I','R')
    def infect_rate(params):
        return lambda s: params[0]*s[0]*s[1]
    def cure_rate(params):
        return lambda s: params[1]*s[1]
    rate_functions = [infect_rate,cure_rate]
    updates = [(-1,1,0),(0,-1,1)]
    init_state = (10,5,0)
    #space = make_statespace(updates,init_state) # unneeded
    # draw a sample trajectory
    #t_f = 10
    #params = [0.4,0.5]
    #concrete_rate_functions = parameterise_rates(rate_functions,params)
    # load observations
    #observations_file = "obsSIR"
    #obs = load_observations(observations_file)
    # prepare the sampler configuration
    conf = {'obs': [], 'parameters': [], 'rate_funcs' : rate_functions}
    parameter_conf = {}
    parameter_conf['prior'] = spst.uniform(loc=0,scale=1)
    parameter_conf['proposal'] = lambda x: spst.norm(loc=x,scale=0.1)
    parameter_conf['limits'] = (0,inf)
    conf['parameters'].extend([parameter_conf,parameter_conf])
    with open('SIR_uncertain.proppa', 'r') as modelfile:
        model = proppa.parse_biomodel(modelfile.read())
    print('Read file.')
    # run a M-H sampler
    sampler = FiniteMetropolisSampler(model,conf)
    #sampler.set_model(model)
    n_samples = 1000
    samples = sampler.gather_samples(n_samples)
#    sampler.calculate_likelihood = \
#            lambda self,pars: calculate_likelihood(pars,self.obs)

#    def rate_functions2(params):
#        return [lambda s: params[0]*s[0]*s[1],
#                lambda s: params[1]*s[1]]