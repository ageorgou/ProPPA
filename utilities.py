# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 19:16:33 2015

@author: s1050238
"""

from operator import add, lt

import numpy as np
from scipy.stats import expon
from scipy.spatial.distance import euclidean
from scipy.linalg import expm

# Gillespie's Stochastic Simulation Algorithm
def gillespie(rate_funcs,stop_time,init_state,updates):
    
    n_reacts = len(rate_funcs)
    t = 0
    s = tuple(init_state)
    path = [(t,s)]
    
    while True:
        jump_rates = [f(s) for f in rate_funcs]
        exit_rate = sum(jump_rates)
        if exit_rate == 0:
            break
        probs = [r/exit_rate for r in jump_rates]
        index = np.random.choice(n_reacts,p=probs)
        t = t + expon.rvs(scale=1/jump_rates[index])
        if t >= stop_time:
            break
        s = update_state(s,updates[index])
        #s = tuple(map(add,s,updates[index])) #extra tuple() for Python 3.x
        #s = tuple(x+y for (x,y) in zip(s,updates[index]))
        path = path + [(t,s)]
    path = path + [(stop_time,s)]
    return path

# Functions for paths / trajectories
def extract_times(trace):
    return [time for (time,state) in trace]

def extract_states(trace):
    return [state for (time,state) in trace]

def combine_times_states(times,states):
    if len(times) != len(states):
        # Should probably raise some exception here
        print('Time and state list must have equal length.')
        return None
    else:
        return zip(times,states)

def split_path(trace):
    return extract_times(trace), extract_states(trace)

def normalise_trace(trace,times):
    #new_trace = []
    new_trace = [None] * len(times)
    i = j = 0
    while i < len(times):
        while trace[j][0] < times[i] and j < len(trace) - 1:
            j = j + 1
        #new_trace.append( (times[i],trace[j][1]) )
        new_trace[i] = (times[i],trace[j][1])
        i = i + 1
            
    return new_trace


# Functions for states and state-spaces
def update_state(state,update):
    return tuple(map(add,state,update))

def make_statespace(updates,initial,limits=None):
# check whether limits, updates and initial all have the same dimension
    space = new_states = set(initial)
    while True:
        new_states = set(update_state(s,u) for s in new_states for u in updates)
        #new_states = remove_negative_states(new_states)
        #new_states = {s for s in new_states if is_nonnegative(s)}
        new_states = set(filter(is_nonnegative,new_states))
        if limits is not None:
            new_states = crop_statespace(new_states,limits)
        if new_states.issubset(space):
            break
        space.update(new_states)
    return list(space)

def crop_statespace(space,limits):
    outside_states = set(s for s in space if any(map(lt,limits,s)))
    space_set = set(space)
    space_set.difference_update(outside_states)
    return space_set
#    for s in space:
#        if any(map(lt,limits,s)):
#            space.remove(s)
    
#def remove_negative_states(states):
#    return {s for s in states if all(x >= 0 for x in s)}

def is_nonnegative(state):
    return all(x >= 0 for x in state)

def find_states(target_states,state_list):
    indices = [None] * len(target_states)    
    for i,item in enumerate(state_list):
        try:
            ind = target_states.index(item)
        except ValueError as ve:
            continue    
        indices[ind] = i
        if all(indices):
            break
    return indices

def make_generator(states,rate_funcs,updates):
# TODO: can definitely write this better
        def make_generator_row(s):
            end_states = [update_state(s,u) for u in updates]
            end_indices = find_states(end_states,states)
            rates = [(i,rate_funcs[i](s)) for i in range(len(rate_funcs))
                        if end_indices[i] is not None]
            row = np.zeros(len(states))
            #row = [0] * len(states)
            for (i,r) in rates:
                row[end_indices[i]] = r
            state_index = find_states([s],states)[0]
            row[state_index] = -sum(row)
            return row
        
        return np.array([make_generator_row(s) for s in states])
        #return [make_generator_row(s) for s in states]

def make_generator2(states,rate_funcs,updates):
    states_array = np.array(states)
    n_states = len(states)
    Q = np.zeros((n_states,n_states))
    for rf,u in zip(rate_funcs,updates):
        rates = rf(states_array.T)
        end_states = states_array + u
        end_indices = find_states([tuple(s) for s in end_states.tolist()],
                                   states) #hacky
        start_indices = find_not_none(end_indices)
        end_indices = [end_indices[i] for i in start_indices]
        Q[start_indices,end_indices] = rates[start_indices]
    for i in range(n_states):
        Q[i,i] = -sum(Q[i,:])
    return Q

def find_not_none(the_list):
    return [ind for (ind,obj) in enumerate(the_list) if obj is not None]

def parameterise_rates(rate_funcs,parameters):
        return tuple(r(parameters) for r in rate_funcs)

#def square_diff(x,y):
#    return (x-y)**2

# Convenience functions for common tasks
def transient_prob(Q,t,init_prob):
    prob = init_prob.dot(expm(Q*t))
    return prob

def euclid_trace_dist(trace,points):
    #norm_trace = normalise_trace(trace,extract_times(points))
    norm_trace = normalise_trace(trace,[p[0] for p in points])
    #distances = map(square_diff,extract_states(norm_trace),
    #                extract_states(points))
    #return sqrt(sum(distances))
    #return euclidean(extract_states(norm_trace),[p[1:] for p in points])
    distances = [euclidean(t1,t2) for (t1,t2) in 
                 zip(extract_states(norm_trace),[p[1:] for p in points])]
    return sum(distances)
    

def ess(samples):
    N = len(samples)
    autocorr = np.correlate(samples,samples,mode='full')
    acf = autocorr[N-1:] / autocorr[N-1]
    
    n = 0
    S = 0
    while acf[n] > 0:
        S = S + acf[n]
    E = N / (1 + 2*S)
    
    return E

def ess_all(samples):
    n_cols = samples.shape[1]
    E = [ess(samples[:,i]) for i in range(n_cols)]
    return E

if __name__ == "__main__":
    def rf1(s):
        return 0.4*s[0]*s[1]
    def rf2(s):
        return 0.5*s[1]    
    updates = [[-1,1,0],[0,-1,1]]
    init = [10,5,0]
    path = gillespie([rf1,rf2],5,init,updates)