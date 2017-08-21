# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:27:55 2015

@author: Anastasis
"""

import numpy as np

from utilities import parameterise_rates,make_statespace,make_generator2
from utilities import find_states, transient_prob,crop_statespace
import model_utilities as mu
from mh import MetropolisSampler
from roulette import Roulette

class RouletteMetropolisSampler(MetropolisSampler):
    """A sampler working with random truncations of the state-space.
    
    This sampler is appropriate for large or infinite-state systems. It
    computes an unbiased estimate of the likelihood by randomly truncating the
    state-space and calculating the exact solution (via matrix exponentiation)
    for the resulting finite spaces. The estimates are used in a pseudomarginal
    Metropolis-Hastings algorithm. The approach is explained in the paper
    'Unbiased Bayesian inference for population Markov jump processes via
    random truncations' by Georgoulas et al. (doi: 10.1007/s11222-016-9667-9).
    """
    
    def _set_model(self,model):
        self.model = model
        self.obs,_ = mu.load_observations(model.obsfile)
        self.updates = mu.get_updates(model)
        self.spaces = [ (tuple(self.obs[i][1:]),tuple(self.obs[i+1][1:]))
                            for i in range(len(self.obs)-1) ]
        self.max_trunc = [0] * (len(self.obs)-1)
    
    def _calculate_likelihood(self,pars):
        """Overriden to compute an unbiased estimate of the likelihood."""
#        # Choose truncation point via Russian Roulette
#        roul = Roulette(Roulette.Geometric(0.95))
#        roul.run()
#        n_terms = roul.n_terms
#        # grow the maximum state-space, if needed
#        if n_terms > self.max_trunc:
#            i = 0
#            while i < len(self.obs):
#                limits = ( np.max([self.obs[i][1:],self.obs[i+1][1:]],axis=0)
#                           + n_terms - 1 )
#                self.spaces[i] = make_statespace(self.model.updates,
#                                                 self.spaces[i],limits)
#                i = i + 1
#            self.max_trunc = n_terms
        # compute likelihood estimate
        rfs = parameterise_rates(self.rate_funcs,pars)
        L = 1
        i = 0
        while i < len(self.obs) - 1:
            n = 0
            prob = old_final_prob = 0
            Dt = self.obs[i+1][0] - self.obs[i][0]
            # Choose truncation point via Russian Roulette
            roul = Roulette(Roulette.Geometric(0.95))
            roul.run()
            n_terms = roul.n_terms
            cumul = np.cumprod(1 - np.array(roul.probs))
            while n < n_terms:
                limits = np.max([self.obs[i][1:],self.obs[i+1][1:]],axis=0) + n
                # grow the maximum state-space, if needed
                if n > self.max_trunc[i]:
                    self.spaces[i] = make_statespace(self.updates,
                                                 self.spaces[i],limits)
                    self.max_trunc[i] = n
                # limit the space if needed
                if n < self.max_trunc[i]:
                     space = list(crop_statespace(self.spaces[i],limits))
                else:
                     space = self.spaces[i]
                Q = make_generator2(space,rfs,self.updates)
                #inds = find_states([o[1:] for o in self.obs],space)
                inds = find_states([tuple(self.obs[i][1:]),
                                    tuple(self.obs[i+1][1:])],space)
                init_prob = np.zeros(len(space))
                init_prob[inds[0]] = 1
                new_final_prob = transient_prob(Q,Dt,init_prob)[inds[1]]
                # compute and add next "Roulette term"
#                print(inds)
#                print(init_prob.shape)
#                print(new_final_prob)
#                print(old_final_prob)
                prob += (new_final_prob - old_final_prob) / cumul[n]
                old_final_prob = new_final_prob
                n = n + 1
            L = L * prob
            i = i + 1
            
        return L