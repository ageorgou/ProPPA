# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:02:51 2017

@author: Anastasis
"""
import numpy as np
import scipy.stats as spst
import matplotlib.pyplot as plt

import proppa

# ProPPA can be called from the command line (python proppa <model>), or can be
# used as a library to perform inference or simulation of continuous-time,
# population-based stochastic systems with uncertain parameters.
# The following runs each of the three experiments reported in the paper:
names = ["SIR.proppa","rumour.proppa","predPrey.proppa"] # example models
n_samples = 100000 # how many samples to take from the posterior
results = []
for model_name in names:
    model = proppa.load_model(model_name)
    samples = model.infer(n_samples)
    # to plot results at this stage:
#    for i in range(len(samples[0])):
#        plt.hist([s[i] for s in samples],bins=50)
    plt.show()
    results.append(samples)

# To plot the results as in the paper:
# all_par_names has the necessary information about how the data was generated
# format: (name,true_value,lower_limit,upper_limit)
all_par_names = {'SIR.proppa' : [('r_i',0.4,0,1),('r_r',0.5,0,1)],
                 'rumour.proppa' : [('k_s',0.5,0,1),('k_r',0.1,0,1)],
                 'predPrey.proppa' : [('a',0.0001,0,0.001),
                                      ('b',0.0005,0,0.001),
                                      ('c',0.0005,0,0.001),
                                      ('d',0.0001,0,0.001)]
                }
for model_name in names:
    for (i,p) in enumerate(all_par_names[model_name]):
        plt.hist([s[i] for s in samples],bins=50)
        plt.xlim([p[2],p[3]])
        par_name = p[0]
        plt.xlabel("$" + par_name + "$",fontsize=14)
        plt.ylabel('# of samples',fontsize=14)
        plt.axvline(p[1],color='r',linewidth=2)
        filename = ("hist_" + model_name.rsplit(sep=".",maxsplit=1)[0] + "_" +
                    par_name + ".pdf")
        plt.savefig(filename)
        plt.show()

# Specifically for the predator-prey example, the following code was used to
# produce the figures (also plotting the prior and using different colours):
model_name = 'predPrey.proppa'
samples = results[2]
pars = all_par_names[model_name]
colors = ['teal','blue','green','black']
line_colors=['red','red','red','red']
xmin, xmax = pars[0][2:4] # all parameters have the same limits
x_range = np.linspace(xmin,xmax,101)
y = spst.gamma.pdf(x_range,a=4,scale=1/10000) # prior pdf
form = plt.ScalarFormatter(useMathText=True)
form.set_powerlimits((0,0))

for (i,p) in enumerate(pars):
    plt.hist([s[i] for s in samples],bins=50,color=colors[i])
    plt.axvline(p[1],color=line_colors[i],linewidth=2)
    plt.xlim([xmin,xmax])
    scaling = plt.ylim()[1] / np.max(y)
    plt.plot(x_range,y * scaling, 'k--')
    plt.gca().xaxis.set_major_formatter(form)
    par_name = p[0]
    plt.xlabel("$" + par_name + "$",fontsize=14)
    plt.ylabel('# of samples',fontsize=14)
    filename = "".join(["hist_",model_name.rsplit(sep=".",maxsplit=1)[0],"_",
                       par_name,".pdf"])
    plt.savefig(filename)
    plt.show()