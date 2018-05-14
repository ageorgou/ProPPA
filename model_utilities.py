# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 22:16:44 2015

@author: Anastasis
"""

from operator import add, sub, mul, truediv
import os.path
import sys
if sys.version_info[0] == 2:
    import ConfigParser as cp
else:
    import configparser as cp

import numpy as np

#from finite_metropolis_sampler import FiniteMetropolisSampler
#from finite_state_gibbs import RaoTehGibbsSampler
#from abc import ABCSampler

class ProPPAException(Exception):
    pass

def get_updates(model,species_names=None):
    reactions = model.get_reactions()
    reaction_names = list(model.get_reactions().keys()) #to have a fixed order
    if species_names is None:
        species_names = [sd.lhs for sd in model.species_defs]
    species_order = dict([(s,i) for (i,s) in enumerate(species_names)])
    n_species = len(species_names)
    stoichiometries = {}
    for (name,reaction) in reactions.items():
        stoich = [0] * n_species
        for reactant in reaction.reactants:
            stoich[species_order[reactant.species]] = -reactant.stoichiometry
        for product in reaction.products:
            stoich[species_order[product.species]] = product.stoichiometry
        stoichiometries[name] = stoich
    updates = [stoichiometries[r_name] for r_name in reaction_names]
    return np.array(updates),reaction_names

# to load and handle observations file:
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
                print('Warning: first column should be named "time".')
                species_names = tuple(tokens[1:])
        else:
            print('Warning: no species names found.')
            species_names = None
            f.seek(0)
        for line in f:
            obs.append([float(x) for x in line.strip().split()])
    
    return obs,species_names

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

def split_indices(l,lookup):
    """Return two lists, within and without. within contains the
    indices of all elements of l that are in lookup, while without
    contains the remaining elements of l."""
    within,without = [],[]
    for (i,v) in enumerate(l):
        try:
            ind = lookup.index(v)
            within.append((i,ind))
        except ValueError: # v not found in lookup
            without.append((i,v))
    return within,without



# to read configuration file
def read_configuration(filename):
    config = cp.ConfigParser()
    try:
        config.read(filename)
    except Exception as e:
        print(e)
    return config

def warning_missing(name,default):
    msg = "Warning: did not find parameter %s. Using default value of %r" % \
            (name,default)
    return msg

#samplers = {'direct' : FiniteMetropolisSampler,
#            'ABC': ABCSampler,
#            'gibbs' : RaoTehGibbsSampler
#            }
samplers = {'direct' : 'FiniteMetropolisSampler',
            'ABC': 'ABCSampler',
            'gibbs' : 'RaoTehGibbsSampler'
            }
def setup_sampler(model):
    sampler_name = model.algorithm
    try:
        sampler_cls = samplers[sampler_name]
    except KeyError:
        print("Cannot recognize sampler name " + sampler_name +
              ". Valid options are: ")
        print(", ".join(samplers))
        return None
    check_requirements(sampler_cls,model)
    return sampler_cls

def check_requirements(sampler,model):
    return

# Converting kinetic laws to Python functions
def as_string2(e):
    name = e.name
    if e.arguments == []:
        if name is not None:
            return '{' + name + '}'
        else:
            return str(e.number)
    elif len(e.arguments) == 1:
        return name + as_string(e.arguments[0])
    else: # here we should probably check whether the expression is an operator
          # or a "proper" function (i.e. it should be printed with parentheses)
        return ( "(" + as_string(e.arguments[0]) +
                 name + as_string(e.arguments[1]) + ")" )

def as_string(e):
    name = e.name
    if e.arguments == []:
        if name is not None:
            return name
        else:
            return str(e.number)
    elif len(e.arguments) == 1:
        if name == 'floor':
            return ( 'floor(' + as_string(e.arguments[0]) + ')')
        elif name == 'H' or name == 'heaviside':
            return ('(1 if ' + as_string(e.arguments[0]) + ' > 0 else 0)' )
        elif name == 'exp':
            return ('exp(' + as_string(e.arguments[0]) + ')')
        else:
            return name + as_string(e.arguments[0])
    else:
        return ( '(' + as_string(e.arguments[0]) +
                         name + as_string(e.arguments[1]) + ')' )

def write_results(results,file):
    """Not pretty, just writes any numpy data (eg sampling results) to the
       specified file."""
    np.savetxt(file,results)

def read_config(model):
    file = os.path.join(model.location,model.conffile)
    with open(file) as contents:
        config = {}
        for line in contents:
            tok = [el.strip() for el in line.split('=')]
            if tok[0].startswith('proposal'):
                prop_tok = tok[0].split()
                if len(prop_tok) == 2:
                    if 'proposals' not in config:
                        config['proposals'] = {} 
                    config['proposals'][prop_tok[1]] = float(tok[1])
                else:
                    print('Could not understand line: ' + line)
            else:
                config[tok[0].lower()] = float(tok[1])
    return config

#def apply_state(state,species_names,expr):
#    env = {}
#    for i in range(len(state)):
#        env[species_names[i]] = proppa.Expression.num_expression(state[i])
#    return expr.get_value(env)


#def get_functions(model):
#    pass
#
#def f_add(f,g):
#    def h(x):
#        return f(x) + g(x)
#    return h
#
#def f_sub(f,g):
#    def h(x):
#        return f(x) - g(x)
#    return h
#
#def f_mult(f,g):
#    def h(x):
#        return f(x) * g(x)
#    return h
#
#def f_div(f,g):
#    def h(x):
#        return f(x) / g(x)
#    return h
#
#def f_floor(f):
#    def h(x):
#        return floor(f(x))
#    return h
#
#ops = { '+':add,
#        '-':sub,
#        '*':mul,
#        '/':truediv}
#def f_op(f,g,op):
#    def h(x):
#        return ops[op](f(x),g(x))
#    return h