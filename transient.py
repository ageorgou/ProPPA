# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 12:33:06 2015

@author: Anastasis
"""

from scipy.linalg import expm
#import numpy as np

def transient_prob(Q,t,init_prob):
    prob = init_prob.dot(expm(Q*t))
    return prob