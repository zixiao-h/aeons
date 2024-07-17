import numpy as np
import matplotlib.pyplot as plt
from aeons.endpoint import EndModel
from aeons.utils import *
from aeons.beta import *
from aeons.plotting import plot_quantiles
figsettings()

import os
current_dir = os.path.dirname(os.path.abspath(__file__))


name, samples = get_samples('gauss_32')
samples['betas_logL'] = get_betas_logL(samples)

ndead = 15000
points = points_at_iteration(samples, ndead)
logbeta_term = np.log(get_beta_end(points, ndead))
logbeta_term_half = np.log(get_beta_end(points, ndead, epsilon=0.5))
logbeta_grad = get_logbeta_grad(points, ndead, interval=250)
logbeta_logL = np.log(samples['betas_logL'].iloc[ndead])
logbeta_post = get_logbeta_post(points, ndead)
logbetas = [1, logbeta_term, logbeta_grad, logbeta_logL, logbeta_post]
beta_titles = ['$\\beta=1$', 'Termination $\\beta$', 'Microcanonical $\\beta$', 'Canonical $\\beta$', 'Bayesian $\\beta$']

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(7, 3))
gs = gridspec.GridSpec(2, 6)
ax0 = fig.add_subplot(gs[0, :2])
ax1 = fig.add_subplot(gs[0, 2:4])
ax2 = fig.add_subplot(gs[0, 4:])
ax3 = fig.add_subplot(gs[1, 1:3])
ax4 = fig.add_subplot(gs[1, 3:5])
axs = [ax0, ax1, ax2, ax3, ax4]

for i, ax in enumerate(axs):
    if i == 4:
        for j in range(25):
            logbeta_post_samp = np.random.normal(*logbetas[i])
            logLX = np.exp(logbeta_post_samp) * points.logL + points.logX()
            ax.plot(-points.logX(10), np.exp(logLX - logLX.max()), lw=.5, color='coral', alpha=.05)
    else:
        logLX = np.exp(logbetas[i]) * points.logL + points.logX()
        ax.plot(-points.logX(25), np.exp(logLX - logLX.max()), lw=.5, color='coral', alpha=.25)
    ax.set_xlim(0, 40)
    ax.axvline(-points.logX().iloc[ndead], color='k', linestyle='--')
    ax.set_xlabel('$-\\log X$')
    ax.set_yticks([])
    ax.set_ylabel('$\\mathcal{P}(\\log X)$')
    ax.margins(y=0.02)
    ax.set_title(beta_titles[i])

logLX = np.exp(logbeta_term_half) * points.logL + points.logX()
ax1.plot(-points.logX(25), np.exp(logLX - logLX.max()), lw=.5, color='orangered', alpha=.25)
from matplotlib.patches import Patch
custom_patches = [
    Patch(color='coral', label='$\\epsilon=10^{-3}$'),
    Patch(color='orangered', label='$\\epsilon=0.5$')]
ax1.legend(handles=custom_patches, loc='lower left', fontsize=6)
    
fig.tight_layout()
fig.savefig(f'{current_dir}/beta_posteriors.pdf', bbox_inches='tight')