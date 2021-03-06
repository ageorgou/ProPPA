# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:30:10 2015

@author: s1050238
"""

import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.integrate import odeint

from mh import MetropolisSampler

class LNASampler(MetropolisSampler):
    """A sampler using the Linear Noise Approximation.
    
    This class computes the likelihood by using a continuous approximation of
    the system. It is similar to the fluid sampler but, instead of being
    deterministic, it considers the state to follow a Gaussian distribution at
    every time point. It constructs and solves a system of ODEs capturing the
    mean and covariance of this distribution, and computes the likelihood under
    it. The calculation implements the method proposed by Fearnhead et al. in
    'Inference for reaction networks using the linear noise approximation'
    (2014, doi: 10.1111/biom.12152).
    """
    
    required_conf = MetropolisSampler.required_conf + ['obs_noise']
    
    @staticmethod
    def prepare_conf(model):
        conf = super().prepare_conf(model)
        conf['rate_funcs'] = model.reaction_functions()
        conf['derivs'] = model.derivative_functions()
        conf['obs_noise'] = 0.1
        return conf
    
    def apply_configuration(self,conf):
        super().apply_configuration(conf)
        self.derivs = conf['derivs']
        self.obs_noise = conf['obs_noise']

    
    def _set_model(self,model):
        self.n_species = len(model.species_order)
        self.updates = model.updates
        self.obs = model.obs
    
    def _calculate_likelihood(self,proposed):
        """Overriden to compute likelihood according to Fearnhead et al."""
        N = self.n_species
        V = self.obs_noise * np.eye(N)
        t = self.obs[0][0]
        b = self.obs[0][1:]
        S = np.zeros(N**2)
        init_cond = np.hstack((b,S))
        
        L = 1
        i = 1
        while i < len(self.obs):
            t_next = self.obs[i][0]
            o_next = self.obs[i][1:]
            # Solve ODE for mean/variance of state given past observations
            sols = odeint(self._dydt,init_cond,[t,t_next],
                                args=([f(proposed) for f in self.rate_funcs],
                                      [[f(proposed) for f in fl]
                                          for fl in self.derivs]
                                     ),
                                hmax = 0.001
                              )
            last_sol = sols[-1,:]
            b = last_sol[:N]
            S = last_sol[N:].reshape(N,N)
            # Update likelihood (using mean and variance of new observation)
            L_update = mvn.pdf(o_next,b,S+V)
            L = L * L_update
            # Compute posterior mean/variance of state, incl. new observation
            factor = S.dot(np.linalg.inv(S+V))
            b = b + factor.dot(o_next-b)
            S = S - factor.dot(S)
            # Set up next iteration
            init_cond = np.hstack((b,S.reshape(N*N)))
            t = t_next
            i = i + 1
        return L
    
    def _dydt(self,y,t,rfs,rds):
        """The right hand side of the ODEs approximating the system."""
        curr_b = y[:self.n_species]
        curr_S = y[self.n_species:].reshape(self.n_species,self.n_species)
        
        jumps = self.updates
        n_react = self.updates.shape[0]
        rates = np.zeros(n_react)
        rate_derivs = np.zeros((n_react,self.n_species))
        for k in range(n_react):
            rates[k] = rfs[k](curr_b)
            for l in range(self.n_species):
                rate_derivs[k,l] = rds[k][l](curr_b)
        A = jumps.T.dot(rate_derivs)
        D = jumps.T.dot(np.diag(rates)).dot(jumps)
        
        new_b = jumps.T.dot(rates)
        new_S = A.dot(curr_S) + curr_S.dot(A.T) + D
        
        return np.hstack((new_b,new_S.reshape(self.n_species**2)))