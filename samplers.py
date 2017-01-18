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

#class InferenceMethod(object):
#    def __init__(self,model,setup):
#        #any (semi-)standard initilisations that must be done for all methods
#        #e.g. number of steps
#        self.model = model # assume already numerized
#        if not self.check_requirements(setup):
#            pass # raise some error
#        self.additional_preparation(setup)
#    
#    def check_requirements(self,setup):
#        return True
#    
#    def infer(self):
#        pass
#        
#    def get_sampler(self):
#        pass
#    
#    def additional_preparation(self,setup):
#        pass
#
#
#class DirectMH(InferenceMethod):
#    def check_requirements(self,model,setup):
#        return True
#    
#    def infer(self):
#        # set up solver
#    
#        pass
#
#class FiniteGibbs(InferenceMethod):
#    def check_requirements(self,setup):
#        # check if all reactions have distinct updates, all parameters have
#        # Gamma priors and the kinetic laws are of the form param * function
#        return True
#    
#    def infer(self):
#        pass
#
#class LNA_MH(InferenceMethod):
#    pass
#
#class TruncationMH(InferenceMethod):
#    pass
#
#class TruncationGibbs(InferenceMethod):
#    pass