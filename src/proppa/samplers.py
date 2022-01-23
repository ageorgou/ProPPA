# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:05:58 2015

@author: s1050238
"""

from model_utilities import ProPPAException
from finite_metropolis_sampler import FiniteMetropolisSampler
from finite_state_gibbs import RaoTehGibbsSampler
from fluid_sampler import FluidSampler
from lna_sampler import LNASampler
from roulette_metropolis_sampler import RouletteMetropolisSampler
from abc_sampler import ABCSampler
from enhanced_abc_sampler import EnhancedABCSampler

sampler_dict = {'direct': FiniteMetropolisSampler,
                'gibbs': RaoTehGibbsSampler,
                'roulette-mh': RouletteMetropolisSampler,
                'ode': FluidSampler,
                'fluid' : FluidSampler,
                'lna': LNASampler,
                'abc': ABCSampler,
                'abc_enhanced': EnhancedABCSampler,
                'enhanced_abc': EnhancedABCSampler}

def get_sampler(name):
    try:
        return sampler_dict[name.lower()]
    except KeyError:
        raise ProPPAException("Unrecognised inference algorithm: " + name)