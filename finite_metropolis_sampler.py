# -*- coding: utf-8 -*-
import numpy as np
from numpy import inf
import scipy.stats as spst

import proppa
from utilities import parameterise_rates, make_statespace, make_generator2
from utilities import find_states, transient_prob
from mh import MetropolisSampler


class FiniteMetropolisSampler(MetropolisSampler):
    """A sampler for finite systems using exact likelihood computation.

    This class computes the likelihood directly via matrix exponentiation. This
    method is exact, but only applicable to finite state-spaces. It may become
    very expensive when the state-space is very large; in these cases,
    approximate methods such as FluidSampler or ABCSampler may be a better
    option.
    """

    def _set_model(self, model):
        self.model = model
        self.obs = model.obs
        self.updates = model.updates
        self.space = make_statespace(self.updates,
                                     [tuple(o[1:]) for o in self.obs])

    def _calculate_likelihood(self, pars):
        rfs = parameterise_rates(self.rate_funcs, pars)
        Q = make_generator2(self.space, rfs, self.updates)
        # inds will hold the indices of the observed states (the rows of the 
        # state-space to which they correspond)
        inds = find_states([tuple(o[1:]) for o in self.obs], self.space)
        L = 1
        i = 0
        while i < len(self.obs) - 1:
            init_prob = np.zeros(len(self.space))
            init_prob[inds[i]] = 1
            Dt = self.obs[i+1][0] - self.obs[i][0]
            final_prob = transient_prob(Q, Dt, init_prob)
            L = L * final_prob[inds[i+1]]
            i = i + 1
        return L

if __name__ == "__main__":
    # create SIR model
    species_names = ('S', 'I', 'R')
    def infect_rate(params):
        return lambda s: params[0]*s[0]*s[1]
    def cure_rate(params):
        return lambda s: params[1]*s[1]
    rate_functions = [infect_rate, cure_rate]
    updates = [(-1, 1, 0), (0, -1, 1)]
    init_state = (10, 5, 0)
    #space = make_statespace(updates,init_state) # unneeded
    # draw a sample trajectory
    #t_f = 10
    #params = [0.4,0.5]
    #concrete_rate_functions = parameterise_rates(rate_functions,params)
    # load observations
    #observations_file = "obsSIR"
    #obs = load_observations(observations_file)
    # prepare the sampler configuration
    conf = {'obs': [], 'parameters': [], 'rate_funcs': rate_functions}
    parameter_conf = {}
    parameter_conf['prior'] = spst.uniform(loc=0, scale=1)
    parameter_conf['proposal'] = lambda x: spst.norm(loc=x, scale=0.1)
    parameter_conf['limits'] = (0, inf)
    conf['parameters'].extend([parameter_conf, parameter_conf])
    with open('SIR_uncertain.proppa', 'r') as modelfile:
        model = proppa.parse_biomodel(modelfile.read())
    print('Read file.')
    # run a M-H sampler
    sampler = FiniteMetropolisSampler(model, conf)
    n_samples = 1000
    samples = sampler.gather_samples(n_samples)