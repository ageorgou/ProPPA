import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst

import proppa
from utilities import gillespie,normalise_trace,extract_states

def sample_dist(dist):
    try:
        return dist.rvs()
    except AttributeError:
        ind = np.random.random_integers(low=0,high=len(dist)-1)
        return dist[ind]
#   alternatively:
#    if 'rvs' in dir(dist):
#        return dist.rvs()
#    else:
#        ind = np.random.random_integers(len(dist)-1)
#        return dist[ind]
    

def debug_dist(dists,n_samples=5000):
    n_params = len(dists)
    samples = [dists[i].rvs(size=n_samples) for i in range(n_params)]
    for i in range(n_params):
        plt.hist(samples[i])
        plt.show()
        
def sample_paths(model,t_final=None,dists=None,n_paths=100):
    # retrieve rate functions, initial state, stop time and updates from model
    abstract_rate_funcs = model.reaction_functions()
    updates = model.updates
    init_state = model.init_state
    if dists is None:
        print("Using priors")
        dists = [p.rhs.to_distribution() for p in model.uncertain]
    if t_final is None: # use the final observation time
        t_final = model.obs[-1][0]
    
    paths = []
    i = 0
    while i < n_paths:
        params = [sample_dist(d) for d in dists]
        rate_funcs = [r(params) for r in abstract_rate_funcs]
        path = gillespie(rate_funcs,t_final,init_state,updates)
        paths.append(path)
        i = i + 1
    return paths


if __name__ == "__main__":
    # toy model and distribution
    model = proppa.load_model('predPreyTest.proppa')
    model.numerize()
    t_f = 1
    paths = sample_paths(model,t_f,n_paths=1000)
#    fixed_values = [5,5,10,10]
#    dists = [spst.rv_discrete(values=(v,1)) for v in fixed_values]
#    paths = sample_paths(model,t_f,n_paths=1000,dists=dists)

    
    for p in paths:
        p_t, p_y = zip(*[(pp[0],pp[1][0]) for pp in p])
        plt.step(p_t,p_y)
    plt.title('Paths of prey (%d SSA runs)' % len(paths))    
    plt.show()
    
    plot_times = np.linspace(0,t_f,101)
    norm_paths = [normalise_trace(p,plot_times) for p in paths]
    norm_prey = [ [e[0] for e in extract_states(p)] for p in norm_paths]
    avg_path = np.average(norm_prey,axis=0)
    std_path = np.std(norm_prey,axis=0)

    upper_line = avg_path + std_path
#    lower_line = avg_path - std_path
    lower_line = np.maximum(avg_path - std_path,0)
    plt.plot(plot_times,avg_path,lw=2)
    plt.plot(plot_times,upper_line,'k--',lw=2)
    plt.plot(plot_times,lower_line,'k--',lw=2)
    plt.fill_between(plot_times,upper_line,lower_line,color='grey',alpha='0.5')
    plt.title('Average prey (+/- 1 std)')
    plt.show()
    
    min_prey = np.min(norm_prey,axis=0)
    max_prey = np.max(norm_prey,axis=0)
    plt.plot(plot_times,avg_path)
    plt.plot(plot_times,min_prey,'k--')
    plt.plot(plot_times,max_prey,'k--')
    plt.title('Mean, min and max prey (%d SSA runs)' % len(paths))
    plt.show()