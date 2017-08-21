# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 18:03:13 2015

@author: Anastasis
"""

import numpy as np

from mh import MetropolisSampler
from utilities import gillespie,parameterise_rates
import utilities
#from model_utilities import load_observations,get_updates

class ABCSampler(MetropolisSampler):
    """A simulation-based sampler which avoids likelihood computations.
    
    This class provides a likelihood-free alternative to the other samplers in
    the framework. It is particularly suited to large systems, where computing
    the likelihood is infeasiable or impractical. Instead, it follows an ABC
    (Approximate Bayesian Computation) approach: for each parameter, the system
    is simulated and a distance is calculated betwween the simulated output and
    the obesrvation. If this distance is small enough (lower than the 'eps'
    configuration parameter), the parameter sample is accepted with a certain
    probability, otherwise it is rejected.
    """
    
    required_conf = MetropolisSampler.required_conf + ['eps']
    
    def __init__(self,model,conf=None):
        if conf is not None:
            self.apply_configuration(conf)
        self.n_pars = len(self.priors)
        self.samples = []
        self.eps = conf['eps']
        if 'dist' in conf:
            self.dist = conf['dist']
        else:
            self.dist = utilities.euclid_trace_dist
        self._set_model(model)
        self.state = tuple(d.rvs() for d in self.priors)
        self.current_prior = np.prod(
                [p.pdf(v) for (p,v) in zip(self.priors,self.state)])
        self.current_dist = self._calculate_distance(self.state)
    
    @staticmethod
    def prepare_conf(model):
        conf = super().prepare_conf(model)
        conf['eps'] = 1
        return conf
    
    def _set_model(self,model):
        self.model = model
        #self.obs = self.fix_obs(model.obs)
        self.obs = model.obs
        self.updates = model.updates
    
    def take_sample(self,append=True):
        proposed = self._propose_state()
        acceptance_prob = self._calculate_accept_prob(proposed)
        if np.random.rand() <= acceptance_prob:
            self.current_prior = self.proposed_prior
            self.current_dist = self.proposed_dist
            self.state = proposed
        if append:
            self.samples.append(self.state)
    
    def _calculate_accept_prob(self,proposed):
        """Overriden to use distance instead of likelihood (following ABC)."""
        self.proposed_dist = self._calculate_distance(proposed)
        if self.proposed_dist < self.eps:
            self.proposed_prior = np.prod(
                    [p.pdf(v) for (p,v) in zip(self.priors,proposed)])
            ratio = self.proposed_prior / self.current_prior
            return ratio
        else: # if the distance is higher than eps, always reject the proposal
            return 0
        
    def _calculate_distance(self,proposed):
        """Compute the distance metric for a proposed parameter value.
        
        Given a proposed parameter set, this simulates the system to obtain a 
        trace. The trace is then compared to the observation using the chosen
        distance metric, and this distance is returned.
        """
        # simulate the system
        rates = parameterise_rates(self.rate_funcs,proposed)
        stop_time = self.obs[-1][0]
        init_state = self.obs[0][1:]
        sample_trace = gillespie(rates,stop_time,init_state,self.updates)
        # get the distance according to the error metric specified
        return self.dist(sample_trace,self.obs)

    def fix_obs(self,obs):
        """Unused?"""
        times = [t[0] for t in obs]
        states = [t[1:] for t in obs]
        return utilities.combine_times_states(times,states)

if __name__ == "__main__":
    import scipy.stats as spst
    from matplotlib.pyplot import figure, hist
    
    import proppa

    
    species_names = ('S','I','R')
    def infect_rate(params):
        return lambda s: params[0]*s[0]*s[1]
    def cure_rate(params):
        return lambda s: params[1]*s[1]
    rate_functions = [infect_rate,cure_rate]
    updates = [(-1,1,0),(0,-1,1)]
    init_state = (10,5,0)
    conf = {'obs': [], 'parameters': [], 'rate_funcs' : rate_functions,
            'eps': 70}
    parameter_conf = {}
    parameter_conf['prior'] = spst.uniform(loc=0,scale=1)
    parameter_conf['proposal'] = lambda x: spst.norm(loc=x,scale=0.01)
    parameter_conf['limits'] = (0,np.inf)
    conf['parameters'].extend([parameter_conf,parameter_conf])
    with open('SIR_uncertain.proppa', 'r') as modelfile:
        model = proppa.parse_biomodel(modelfile.read())
    # run a M-H sampler
    sampler = ABCSampler(model,conf)
    n_samples = 50000
    samples = sampler.gather_samples(n_samples)
    figure(); hist([s[0] for s in samples])
    figure(); hist([s[1] for s in samples])