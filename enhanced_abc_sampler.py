# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 21:17:48 2017

@author: Anastasis
"""

import numpy as np

from abc_sampler import ABCSampler
from utilities import gillespie,parameterise_rates,split_path,combine_times_states
#from model_utilities import load_observations,get_updates

class EnhancedABCSampler(ABCSampler):
    """A variant of the ABC sampler supporting enhanced features.
    
    This class works exactly the same way as the standard ABC sampler, with two
    differences: it can consider multiple observation files, and it supports
    complex observables (specified as functions of the species, rather than
    direct measurements of the species themselves).
    """
    
    supports_enhanced = True
        
    def _set_model(self,model):
        super()._set_model(model)
        self.obs_mapping = model.observation_mapping()
    
    def _calculate_distance(self,proposed):
        distance = 0
        # simulate the system
        rates = parameterise_rates(self.rate_funcs,proposed)
        for ob in self.obs:
            stop_time = ob[-1][0]
            init_state = self.model.init_state
            sample_trace = gillespie(rates,stop_time,init_state,self.updates)
            # get the distance according to the error metric specified
            trans_trace = self._translate(sample_trace,list(proposed))
            trans_ob = [(t[0],tuple(t[1:])) for t in ob]
            distance += self.dist(trans_trace,trans_ob)
        return distance
        
    def _translate(self,trace,params):
        """Map a simulated trace to the corresponding observable quantities."""
        #times,states = [t[0] for t in trace], [t[1:] for t in trace]
        times,states = split_path(trace)
        translated_states = [[m(params)(state) for m in self.obs_mapping]
                                for state in states]
        return combine_times_states(times,translated_states)
        #return [[t] + s for (t,s) in zip(times,translated_states)]

    def _translate2(self,trace,params):
        """Unused?"""
        times,states = [t[0] for t in trace], [tuple(t[1:]) for t in trace]
        #times,states = split_path(trace)
        translated_states = [[m(params)(state) for m in self.obs_mapping]
                                for state in states]
        return combine_times_states(times,translated_states)
        #return [[t] + s for (t,s) in zip(times,translated_states)]

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
    sampler = EnhancedABCSampler(model,conf)
    n_samples = 50000
    samples = sampler.gather_samples(n_samples)
    figure(); hist([s[0] for s in samples])
    figure(); hist([s[1] for s in samples])