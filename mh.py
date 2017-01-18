# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:58:47 2015

@author: s1050238
"""

import numpy as np

class MetropolisSampler(object):
    """A class implementing a Metropolis-Hastings sampler.
    """
    
    required_conf = ['proposals']
    supports_enhanced = False
    
#      A compact model needs:
#        ordering of species names
#        ordering of reactions?
#        ordering of parameters
#        stoichiometry (as a dictionary or list of updates)
#        priors on parameters
#        rate functions
    
    def __init__(self,conf=None):
        if conf is not None:
            self.apply_configuration(conf)
        self.n_pars = len(self.priors)
        self.state = tuple(d.rvs() for d in self.priors)
        self.samples = []
        self.current_prior = np.prod([p.pdf(v) \
            for (p,v) in zip(self.priors,self.state)])
        self.current_L = self.calculate_likelihood(self.state)

    @staticmethod
    def prepare_conf(model):
        conf = {'parameters': []}
        for p in model.uncertain:
            par_conf = {'name' : p.lhs}
            prior = p.rhs.to_distribution()
            par_conf['prior'] = prior
            # the below does not work: p.a (p.b) gives the lower (upper) bound
            # of the support of p ONLY IF p has the default parametrisation
            # (loc = 0, scale = 1)
            #par_conf['limits'] = (max(prior.a,0), min(prior.b,np.inf))
            # Instead, use the inverse cdf:
            par_conf['limits'] = (max(prior.ppf(0),0),min(prior.ppf(1),np.inf))
            # Or could just use (0,np.inf) for all distributions?
            conf['parameters'].append(par_conf)
        conf['rate_funcs'] = model.reaction_functions()
        return conf
    
    def apply_configuration(self,conf):
        self.priors = [p['prior'] for p in conf['parameters']]
        self.proposals = [p['proposal'] for p in conf['parameters']]
        self.limits = [p['limits'] for p in conf['parameters']]
        #self.obs = conf['obs']
        self.rate_funcs = conf['rate_funcs']
    
    def calculate_accept_prob(self,proposed):
        self.proposed_prior = np.prod([p.pdf(v) \
            for (p,v) in zip(self.priors,proposed)])
        self.proposed_L = self.calculate_likelihood(proposed)
        
        ratio =  (self.proposed_prior * self.proposed_L)/ \
                    (self.current_prior * self.current_L)
        return ratio
    
    def take_sample(self,append=True):
        #proposed = tuple(p(self.state) for p in self.proposals)
        proposed = self.propose_state()
        acceptance_prob = self.calculate_accept_prob(proposed)
        if np.random.rand() <= acceptance_prob:
            self.current_L = self.proposed_L
            self.current_prior = self.proposed_prior
            self.state = proposed
        if append:
            self.samples.append(self.state)
         
    def gather_samples(self,n_samples):
        n = 0
        while n < n_samples:
            self.take_sample()
            n = n + 1
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
                within_limits = proposed < upper and proposed > lower
            proposed_all[i] = proposed
            i = i +1
        return tuple(proposed_all)
    
    def reset(self):
        self.samples = []
        self.state = tuple(d.rvs() for d in self.priors)

    
    def test_propose(self,n_proposals):
        """ For debugging purposes. """
        proposals = []
        for i in range(n_proposals):
            proposals.append(self.propose_state())
        return proposals

if __name__ == "__main__":
    from numpy import inf
    import scipy.stats as spst
    import matplotlib.pylab
    # make default model
    species = ['S','I','R']
    reactions = ['infect','cure']
    params = ['r1','r2']
    stoich = []
    # make default configuration
    default_conf = {'obs': [], 'parameters': []}
    default_conf['obs'] = []
    parameter_conf = {}
    parameter_conf['prior'] = spst.uniform(loc=0,scale=10)
    parameter_conf['proposal'] = lambda x: spst.norm(loc=x,scale=1)
    parameter_conf['limits'] = (-inf,inf)
    default_conf['parameters'].append(parameter_conf)
    
    class GaussianSampler(MetropolisSampler):
        target_dist = spst.norm(loc=5,scale=1)
        def calculate_likelihood(self,proposed):
            return self.target_dist.pdf(proposed)
    
    s = GaussianSampler(default_conf)
    n_samples = 1000
    samples = s.gather_samples(n_samples)
    matplotlib.pylab.hist(np.array(samples))