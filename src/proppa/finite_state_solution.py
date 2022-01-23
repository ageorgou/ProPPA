# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 15:22:41 2015

@author: Anastasis
"""

import numpy as np
from numpy import inf
import scipy.stats as spst
from scipy.linalg import expm

from utilities import gillespie,parameterise_rates,make_statespace
from mh import MetropolisSampler

def read_observations(obs_file):
    pass
    return None

def my_likelihood(pars,obs):
    L = 1
    i = 0
    while i < len(obs):
        ind = find_states(space,[obs(i)[1],obs(i+1)[1]])
        init_prob = [0] * len(space)
        init_prob[ind[0]] = 1
        Dt = obs(i+1)[0] - obs(i)[0]
        final_prob = transient_prob(Q,Dt,np.array(init_prob))
        L = L * final_prob[ind[1]]
        i = i + 1
    return L

def transient_prob(Q,t,init_prob):
    prob = init_prob.dot(expm(Q*t))
    return prob

def find_states(target_states,state_list):
    return [state_list.index(s) for s in target_states]

def load_observations(input_name):
    obs = []    
    # see if first line has species names; if not, assign default ordering
    # (from model or alphabetically?)
    with open(input_name) as f:
        first_line = f.readline()
        tokens = first_line.strip().split()
        if not all_numbers(tokens):
            if is_time_header(tokens[0]):
                species_names = tuple(tokens[1:])
            else:
                # TODO: raise something?
                print('Warning: first column should be named "time"')
        else:
            print('Warning: no species names found, assuming default order')
            f.seek(0)
        for line in f:
            obs.append([float(x) for x in line.strip().split()])
    
    return obs

def all_numbers(the_list):
    try:
        [float(n) for n in the_list]
        return True
    except ValueError:
        return False

def is_time_header(the_string):
    return the_string == "t" or the_string == "T" or the_string == "time"

def split_observations(obs):
    times = [o[0] for o in obs]
    measurements = [tuple(o[1:]) for o in obs]
    return (times,measurements)

if __name__ == "__main__":
    # create SIR model
    species_names = ('S','I','R')
    def rf1(params):
        return lambda s: params[0]*s[0]*s[1]
    def rf2(params):
        return lambda s: params[1]*s[1]
    rate_functions = [rf1,rf2]
    updates = [(-1,1,0),(0,-1,1)]
    init_state = (10,5,0)
    space = make_statespace(updates,init_state)
    # draw a sample trajectory
    t_f = 10
    params = [0.4,0.5]
    concrete_rate_functions = parameterise_rates(rate_functions,params)
    sample_trace = gillespie(concrete_rate_functions,t_f,init_state,updates)
    # load observations
    observations_file = ""
    obs = load_observations(observations_file)
    # prepare the sampler configuration
    conf = {'obs': [], 'parameters': []}
    conf['obs'] = obs
    parameter_conf = {}
    parameter_conf['prior'] = spst.uniform(loc=0,scale=1)
    parameter_conf['proposal'] = lambda x: spst.norm(loc=x,scale=1)
    parameter_conf['limits'] = (0,inf)
    conf['parameters'].extend([parameter_conf,parameter_conf])
    # run a M-H sampler
    class MyFiniteSampler(MetropolisSampler):
        def calculate_likelihood(self,pars):
            return my_likelihood(pars,self.obs)
    sampler = MyFiniteSampler()
    n_samples = 1000
    samples = sampler.gather_samples(n_samples)
#    sampler.calculate_likelihood = \
#            lambda self,pars: calculate_likelihood(pars,self.obs)

#    def rate_functions2(params):
#        return [lambda s: params[0]*s[0]*s[1],
#                lambda s: params[1]*s[1]]