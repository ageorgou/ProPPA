# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 19:11:16 2016

@author: Anastasis
"""

import numpy as np
import scipy as sp
from scipy.integrate import odeint

from mh import MetropolisSampler
#from utilities import parameterise_rates

class FluidSampler(MetropolisSampler):
    
    required_conf = MetropolisSampler.required_conf + ['obs_noise']
    supports_enhanced = True
    
    def __init__(self,model,conf):
        self.all_proposed = []
        self.all_like = []
        np.seterr(divide='raise')
        #self.logfile = open('fluid_log','w')
        if conf is not None:
            self.apply_configuration(conf)
        self.set_model(model)
        self.n_pars = len(self.priors)
        #self.obs_noise = conf['obs_noise']
        self.samples = []
        self.state = list(d.rvs() for d in self.priors)
        #self.state = [0.7]
        self.current_prior = np.prod([p.pdf(v) \
            for (p,v) in zip(self.priors,self.state)])
        self.current_L = self.calculate_likelihood(self.state)
    
    @staticmethod
    def supports_partial():
        """ Indicates whether this sampler supports partial observations,
            i.e. only some of the species of the model.
        """
        return True
    
    @staticmethod
    def prepare_conf(model):
        conf = MetropolisSampler.prepare_conf(model)
        conf['rate_funcs'] = model.reaction_functions()
        conf['obs_noise'] = 1
        #conf['observed_species'] = range(len(model.species_order))
        return conf
    
    def apply_configuration(self,conf):
        super(FluidSampler,self).apply_configuration(conf)
        #MetropolisSampler.apply_configuration(conf)
        self.obs_noise = conf['obs_noise']
        #self.observed_species = conf['observed_species']

    
    def set_model(self,model):
        self.n_species = len(model.species_order)
        self.updates = model.updates
        # now set in apply_configuration():
#        self.rate_funcs = model.reaction_functions()
#        self.derivs = model.derivative_functions()
        self.init_state = model.init_state
        self.obs = [np.array(ob) for ob in model.obs]
        self.obs_mapping = model.observation_mapping()
    
    def calculate_accept_prob(self,proposed):
        """Overriden to work with log-likelihood."""
        self.proposed_prior = np.prod([p.pdf(v) \
            for (p,v) in zip(self.priors,proposed)])
        self.proposed_L = self.calculate_likelihood(proposed)

        ratio =  (self.proposed_prior / self.current_prior) * \
            np.exp(self.proposed_L - self.current_L)
        return ratio


    def calculate_likelihood(self,proposed):
        self.all_proposed.append((self.state,proposed))
        #rfs = parameterise_rates(self.rate_funcs,proposed)
        rfs = [f(proposed) for f in self.rate_funcs]
        #n_observed = len(self.observed_species)
        n_observed = len(self.obs_mapping)
        invV = 1/self.obs_noise * np.eye(n_observed)
        init_cond = self.init_state
        logL = 0
        for ob in self.obs:
            times = ob[:,0]            
    #        sols = odeint(self._dydt,init_cond,times,
    #                      args = (rfs,),hmax=0.001)
            sols = odeint(self._dydt,init_cond,times,args = (rfs,))
            #diffs = ob[:,1:] - sols[:,self.observed_species]
            # TODO is this too slow?
            diffs = (ob[:,1:] - 
                    np.array([m(proposed)(sols.T) for m in self.obs_mapping]).T)
            #logL = np.sum(-1/2 * diffs.dot(invV).T.dot(diffs))
            self.all_like.append(logL)
            logL += -1/2 * np.sum((diffs*diffs).dot(invV))
        return logL
    
    def _dydt(self,y,t,rfs):
        rates = np.array([rf(y) for rf in rfs])
        return rates.dot(self.updates)
    
    def log(self,*args):
        #print(*args,file=self.logfile,flush=True)
        pass
    
    def gather_samples(self,n_samples):
        n = 0
        while n < n_samples:
            self.take_sample()
            n = n + 1
            self.log(n,"samples taken.")
            if n % 500 == 0:
                print("Taken",n,"samples")
        return self.samples
    
    def propose_state(self):
        #within_limits = [False] * self.n_pars
        proposed_all = [0] * self.n_pars
        i = 0
        while i < self.n_pars:
            lower, upper = self.limits[i]
            within_limits = False
            while not within_limits:
                proposed = self.proposals[i](self.state[i]).rvs()
                self.log("Tried to propose:",proposed)
                within_limits = proposed < upper and proposed > lower
                if not within_limits:
                    self.log(" ...Rejected")
            proposed_all[i] = proposed
            i = i +1
        return tuple(proposed_all)
     
#    def mask(self,y):
#        return np.array([y[i] for i in self.observed_species])
#    
#    def mask2(self,y):
#        return y[:,self.observed_species]