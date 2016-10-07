# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:47:14 2015

@author: Anastasis
"""

import numpy as np

from finite_state_gibbs import *
from roulette import Roulette
from utilities import make_statespace, make_generator2

class RouletteGibbsSampler(RaoTehGibbsSampler):
    def __init__():
        
        self.obs_states = [o[1:] for o in self.obs]
        proposed = self.propose_state()
        self.accept(proposed)
        
    
    def propose_state(self):
        # propose parameters
        proposed = _sample_gamma(self.hyper + self.hyper_updates)
        rfs = parameterise_rates(self.rate_funcs,proposed)
        
        roul = Roulette(Roulette.Geometric(0.95))
        roul.run()
        n_terms = roul.n_terms
        limits = np.max(self.obs_states,axis=0) + n_terms - 1
        self.space = make_statespace(self.updates,self.obs_states,limits)
        self.Q = make_generator2(self.space,rfs,self.updates)
        self.times, self.states = _sample_posterior_path(self.Q,rfs,self.obs)
        (self.times,self.states) = _remove_self_loops(self.times,self.states)
        return proposed
    
    def calculate_accept_prob(proposed):
        pass
    
    def accept(self,proposed):
        self.state = proposed
        self.old_Q = self.Q
        self.old_space = self.space
        self.old_times = self.times
        self.old_states = self.states
        rfs = parameterise_rates(self.rate_funcs,proposed)
        self.hyper_updates = _gamma_updates(self.times,self.states,
                                            rfs,self.updates)
        
    def take_sample(self,append=True):
        proposed = self.propose_state()
        acceptance_prob = self.calculate_accept_prob(proposed)
        if np.random.rand() <= acceptance_prob:
            self.accept(proposed)
        if append:
            self.samples.append(self.state)
