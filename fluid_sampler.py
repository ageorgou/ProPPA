# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 19:11:16 2016

@author: Anastasis
"""

import numpy as np
from scipy.integrate import odeint

from mh import MetropolisSampler

class FluidSampler(MetropolisSampler):
    """A sampler based on a deterministic approximation of the model.
    
    This class implements likelihood computations by constructing a set of
    Ordinary Differential Equations (ODEs) and comparing their solution to the
    observations. It overrieds the default acceptance probability computation
    by using an equivalent formulation in terms of log-likelihoods, for
    improved numerical stability, and to handle multiple observation files.
    """
    
    required_conf = MetropolisSampler.required_conf + ['obs_noise']
    supports_enhanced = True
    supports_partial = True
    
    def __init__(self,model,conf=None):
        np.seterr(divide='raise')
        super().__init__(model,conf)
    
    @staticmethod
    def prepare_conf(model):
        conf = MetropolisSampler.prepare_conf(model)
        conf['rate_funcs'] = model.reaction_functions()
        conf['obs_noise'] = 1
        return conf
    
    def apply_configuration(self,conf):
        super().apply_configuration(conf)
        self.obs_noise = conf['obs_noise']
    
    def _set_model(self,model):
        self.n_species = len(model.species_order)
        self.updates = model.updates
        self.init_state = model.init_state
        self.obs = [np.array(ob) for ob in model.obs]
        self.obs_mapping = model.observation_mapping()
    
    def _calculate_accept_prob(self,proposed):
        """Overriden to work with log-likelihood."""
        self.proposed_prior = np.prod([p.pdf(v)
                for (p,v) in zip(self.priors,proposed)])
        self.proposed_L = self._calculate_likelihood(proposed)
        ratio =  ((self.proposed_prior / self.current_prior) * 
                  np.exp(self.proposed_L - self.current_L))
        return ratio

    def _calculate_likelihood(self,proposed):
        rfs = [f(proposed) for f in self.rate_funcs]
        n_observed = len(self.obs_mapping)
        invV = 1/self.obs_noise * np.eye(n_observed)
        init_cond = self.init_state
        logL = 0
        for ob in self.obs:
            times = ob[:,0]
            sols = odeint(self._dydt,init_cond,times,args = (rfs,))
            # Compute all differences between solutions and observations
            diffs = ob[:,1:] - np.array(
                            [m(proposed)(sols.T) for m in self.obs_mapping]).T
            # Under gaussian noise, the log-likelihood is a quadratic form 
            # (up to an additive constant). For multiple files, the total
            # log-likelihood is obtained by summing over individual data sets.
            logL += -1/2 * np.sum((diffs*diffs).dot(invV))
        return logL
    
    def _dydt(self,y,t,rfs):
        """The right hand side of the ODEs approximating the system."""
        rates = np.array([rf(y) for rf in rfs])
        return rates.dot(self.updates)