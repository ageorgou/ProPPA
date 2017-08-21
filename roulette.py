# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:35:36 2015

@author: s1050238
"""

import numpy as np

class Roulette(object):
    """A class describing a Russian Roulette-style truncation strategy."""

    def __init__(self,scheme=None):
        self.n_terms = 0
        self.probs = []
        if scheme is not None:
            self.scheme = scheme
        else:
            self.scheme = Roulette.Geometric(0.95) # default scheme
    
    def run(self):
        for stop_prob in self.scheme:
            a = np.random.random()
            if a < stop_prob:
                break
            else:
                self.n_terms = self.n_terms + 1
                self.probs.append(stop_prob)
    

    @staticmethod
    def Geometric(reduction_factor):
        """Create a Roulette scheme with geometric acceptance probabilities.
        
        A method for creating roulette schemes with geometrically reducing
        acceptance probability. At every iteration, the generator returns the
        stopping probability, starting with 0.
        """
        prob = 1 # probability to continue
        while True:
            yield 1 - prob
            prob = prob * reduction_factor