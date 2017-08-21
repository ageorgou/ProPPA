# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:32:16 2015

@author: s1050238

Solution
following V. Rao & Y. Teh, "Fast MCMC Sampling for Markov Jump Processes and
Extensions" (2013, JMLR)
"""

from operator import sub

import numpy as np
import scipy as sp

from mh import MetropolisSampler
from utilities import find_states, gillespie, parameterise_rates
from utilities import make_statespace, make_generator2
import model_utilities as mu

class RaoTehGibbsSampler(MetropolisSampler):
    
    def __init__(self,model,conf):
        self._set_model(model)
        self.apply_configuration(conf)
        self.n_pars = len(self.hyper[0])
        #TODO: vectorise / broadcast
        #self.state = tuple(_sample_gamma(a,b) for (a,b) in self.hyper)
        self.hyper_updates = np.zeros((self.n_pars,self.n_pars))
        self.samples = []
        self.space = make_statespace(self.updates,
                                     [tuple(o) for o in self.obs[:,1:]])
    def _set_model(self,model):
        self.model = model
        self.obs = np.array(model.obs)
        self.updates = model.updates
    
    def apply_configuration(self,conf):
        self.hyper = np.array(([p['prior_a'] for p in conf['parameters']],
                               [p['prior_b'] for p in conf['parameters']]))
        self.rate_funcs = conf['rate_funcs']
        #self.proposals = [p['proposal'] for p in conf['parameters']]
        #self.limits = [p['limits'] for p in conf['parameters']]
        #self.obs = conf['obs']
        self.rate_funcs = conf['rate_funcs']
    
    def take_sample(self,append=True):
        """Overriden so that proposed samples are always accepted."""
        self.state = self.propose_state()
        if append:
            self.samples.append(self.state)
        pass
    
    def _propose_state(self):
        """Propose parameters by sampling from the conditional posterior."""
        # make generator
        # sample path
        # add self-loops
        # create generator of DTMC
        # run FFBS to draw new trace
        
        #proposed = (0.4,0.5) # Debug
        while True:
            proposed = _sample_gamma(self.hyper + self.hyper_updates)
           # print(proposed)
            rfs = parameterise_rates(self.rate_funcs,proposed)
            A = make_generator2(self.space,rfs,self.updates)
            (times,states) = _sample_posterior_path(A,rfs,self.obs,self.space,
                                                self.updates)
            if states is not None:
                break
        # remove self-loops from new trace
        (times,states) = _remove_self_loops(times,states)
        # (update gamma hyperparameters and) sample parameters
        self.hyper_updates = _gamma_updates(times,states,rfs,self.updates)
        
        return proposed
    
    def _calculate_accept_prob(self,proposed):
        """Overriden to always return 1, although the value is not used."""
        return 1


def _discretise_generator(A):
    exit_rate = 1.1 * max(-np.diag(A))
    B = np.eye(A.shape[0]) + A / exit_rate
    return B, exit_rate


def _add_self_loops(A,states,times,space,exit_rate):
    i = 0
    inds = find_states(states,space)
    all_times = []
    all_states = []
    #print(times)
    while i < len(times) - 1:
        dt = times[i+1] - times[i]
        total_rate = A[inds[i],inds[i]] + exit_rate
        n_jumps = sp.stats.poisson(total_rate*dt).rvs()
        new_times = (times[i] + np.random.random(n_jumps) * dt).tolist()
        new_times.sort()
        new_times.insert(0,times[i])
        all_times.extend(new_times)
        all_states.extend([states[i]]*(n_jumps+1))
        i = i + 1
    return (all_states,all_times)

def _sample_posterior_path(A,rfs,obs,space,updates):
    path = gillespie(rfs,obs[-1,0],obs[0,1:],updates)
    #print(path)
    times,states = zip(*path)
    P,exit_rate = _discretise_generator(A)
    (states,times) = _add_self_loops(A,states,times,space,exit_rate)
    states = _FFBS(P,space,times,obs)
    if states is not None:
        return times,states
    else:
        return times,None


def _remove_self_loops(times,states):
    # len(times) must == len(states)
    to_keep = [0]
    n = 1
    while n < len(times):
        if np.any(states[n] != states[n-1]):
            to_keep.append(n)
        n = n + 1
    #to_keep = [0] + \
    #            [n for n in range(len(times)) if states[n] != states[n-1]]
    # TODO: use numpy indexing, if the results are ndarrays?
    new_times = [times[n] for n in to_keep]
    new_states = [states[n] for n in to_keep]
    return (new_times,new_states)


def _sample_gamma(hypers):
    #print(hypers)
    a, b = hypers #a in top row, b in bottom
    return sp.stats.gamma.rvs(a,scale=1/b)


def _gamma_updates(times,states,rate_funcs,updates):
        #TODO: check whether numpy can make this faster (vectorised/broadcast?)
        a_updates = np.zeros(len(rate_funcs))
        b_updates = np.zeros(len(rate_funcs))
        dt = np.diff(times)
        n = 0
        while n < len(times) - 1:
            #dt = times[n] - times[n-1]
            jump = list(map(sub,states[n+1],states[n]))
            update_ind = updates.tolist().index(jump)
            #props = [r(states[n]) for r in rate_funcs]
            props = np.array([r(states[n]) for r in rate_funcs])
            a_updates[update_ind] += 1
            #b_updates = list(map(add,props*dt,b_updates))
            #print(n)
            #print(props*dt)
            #print(props*dt + b_updates)
            b_updates = props*dt[n] + b_updates
            n = n + 1
        return a_updates,b_updates

 
def _FFBS(P,space,times,obs):
    n_states = P.shape[0]
    dim = len(space[0]) # number of species / dimension of state-space
    #init_ind = find_states([tuple(o) for o in obs[:,1:].tolist()],space)
    # I don't think we actually need the above? just for the first observation
    init_ind = find_states([tuple(obs[0,1:])],space)
    a = np.zeros((len(times),n_states))
    a[0,init_ind] = 1
    probs = np.zeros((len(times),n_states))
    probs[0,init_ind] = 1
    # forward filtering:
    i = 1
    while i < len(times):
        #find which observations occured
        ind = np.logical_and(obs[:,0] >= times[i-1],obs[:,0] < times[i])
        # observation probabilities
        if not np.any(ind):
            probs[i] = np.ones(n_states)
        else:
            probs[i] = np.prod([_obs_probs(obs[ii,1:],space)
                            for ii in np.where(ind)[0]],
                           axis=0)
        # forward recursion
        a[i] = (a[i-1]*probs[i-1]).dot(P)
        if not any(a[i]):
            return None
        i = i + 1
    # last observation
    if obs[-1,0] == times[-1]:
        probs[i] = _obs_probs(obs[-1,1:],space)

    # sample to be drawn
    sample = np.zeros((len(times),dim))
    # initialise backward message
    i = len(times) - 1
    b = a[i] * probs[i]
    if not np.any(b):
        return None
    ind = np.random.choice(n_states,p=b/b.sum())
    sample[i] = space[ind]
    # backward sampling:
    i = i - 1
    while i >= 0:
        # backward recursion
        b = a[i] * probs[i] * P[:,ind] # TODO: check if correct (a / b?)
        ind = np.random.choice(n_states,p=b/b.sum())
        sample[i] = space[ind]
        i = i - 1

    return sample

    
def _obs_probs(obs,space):
    D = 1E-6
    diffs = np.array(space) - obs
    dists = np.sqrt(np.sum(diffs**2,1))
    p = 1 / (2**dists + D)
    p = p / sum(p) # TODO: do we need to add an entry for an absorbing state?
    return p

    
if __name__ == "__main__":
    # set up model
    species_names = ('S','I','R')
    def rf1(params):
        return lambda s: params[0]*s[0]*s[1]
    def rf2(params):
        return lambda s: params[1]*s[1]
    rate_functions = [rf1,rf2]
    updates = [(-1,1,0),(0,-1,1)]
    init_state = (10,5,0)
    space = []
    
    
    