import numpy as np
import matplotlib.pyplot as plt
from aeons.endpoint import EndModel
from aeons.utils import *
from aeons.beta import *
from aeons.plotting import plot_quantiles
figsettings()

import os
current_dir = os.path.dirname(os.path.abspath(__file__))



fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.3))

for chain, ax in zip(['gauss_8', 'slab_spike'], axs):
    name, samples = get_samples(chain)
    model = EndModel(samples)
    endpoint_true = model.true_endpoint()
    logXf_true = samples.logX().iloc[endpoint_true]
    samples['beta_logL'] = get_betas_logL(samples)

    Npoints = 25
    iterations = make_iterations(endpoint_true, Npoints)
    logbetas_term = np.zeros(Npoints)
    logbetas_term_half = np.zeros(Npoints)
    logbetas_grad = np.zeros(Npoints)
    logbetas_post_mean = np.zeros(Npoints)
    logbetas_post_std = np.zeros(Npoints)
    logbetas_logL = np.log(samples.beta_logL.iloc[iterations])

    for i, ndead in enumerate(iterations):
        points = points_at_iteration(samples, ndead)
        logbetas_term[i] = np.log(get_beta_end(points, ndead, epsilon=1e-3))
        logbetas_term_half[i] = np.log(get_beta_end(points, ndead, epsilon=0.5))
        logbetas_grad[i] = get_logbeta_grad(points, ndead)
        logbetas_post_mean[i], logbetas_post_std[i] = get_logbeta_post(points, ndead)
        print('\r', f'Iteration {ndead}/{iterations[-1]}', end='')
        
    logXs = samples.logX().iloc[iterations]
    ax.plot(-logXs, logbetas_term, color='C0', label='Termination ($\\epsilon=10^{{-3}}$)')
    ax.plot(-logXs, logbetas_term_half, ls='--', color='C0', label='Termination ($\\epsilon=0.5$)', zorder=100)
    ax.plot(-logXs, logbetas_grad, color='black', label='Microcanonical')
    ax.plot(-logXs, logbetas_logL, color='orange', label='Canonical')
    ax.fill_between(-logXs, logbetas_post_mean - logbetas_post_std, logbetas_post_mean + logbetas_post_std, alpha=.5, color='red', label='Bayesian')
    ax.fill_between(-logXs, logbetas_post_mean - 2*logbetas_post_std, logbetas_post_mean + 2*logbetas_post_std, alpha=.2, color='red')

for ax in axs:
    ax.set_xlabel(r"$-\log X$")

axs[0].set_ylabel(r'$\log \beta$')    
axs[0].set_ylim(-14, 5)
axs[1].set_ylim(-8, 8)

handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles, labels, ncol=2, bbox_to_anchor=(0.1, -0.4), fontsize=7)
axs[0].set_title('Spherical Gaussian', fontsize=7)
axs[1].set_title('Slab-spike Gaussian', fontsize=7)

fig.savefig(f'{current_dir}/beta_plots.pdf', bbox_inches='tight')