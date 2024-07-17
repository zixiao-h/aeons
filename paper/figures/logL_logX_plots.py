import numpy as np
import matplotlib.pyplot as plt
from aeons.endpoint import EndModel
from aeons.utils import *
from aeons.beta import *
from aeons.plotting import plot_quantiles
from aeons.likelihoods import full
figsettings()

import os
current_dir = os.path.dirname(os.path.abspath(__file__))


chains = ['gauss_8', 'correlated_6d_close', 'rosenbrock_10', 'cauchy_82']
titles = ['Spherical Gaussian', 'Elongated Gaussian', 'Rosenbrock', 'Cauchy']
fig, axs = plt.subplots(1, 4, figsize=(7, 2))

for chain, title, ax in zip(chains, titles, axs):
    name, samples = get_samples(chain)
    model = EndModel(samples)
    endpoint_true = model.true_endpoint()
    logXf_true = samples.logX().iloc[endpoint_true]
    samples = samples.iloc[:endpoint_true]
    
    ndead = int(endpoint_true * 0.3)
    points = points_at_iteration(samples, ndead)
    logL, X_mean, nk, logZdead = data(points)
    d_G, _ = get_d_G_post(points, ndead)
    
    for i in range(25):
        X = generate_Xs(nk)
        d = np.random.choice(d_G)
        theta = params_from_d(logL[ndead:], X[ndead:], d)
        logL_model = full.func(np.exp(samples.logX()), theta)
        ax.plot(-samples.logX().iloc[ndead:], logL_model.iloc[ndead:], lw=.5, color='deepskyblue')
    ax.plot(-samples.logX().iloc[ndead:], samples.logL.iloc[ndead:], lw=2, color='black')
    ax.set_xlabel('$-\\log X$')
    ax.set_title(f'{title}\n ({len(samples):,} total iterations)', fontsize=8)
    ax.set_yticks([])
axs[0].set_ylabel('$\\log \\mathcal{L}$')

from matplotlib.lines import Line2D
black_patch = Line2D([0], [0], color='black', lw=2)
green_patch = Line2D([0], [0], color='lightseagreen', lw=.5)
axs[0].legend([black_patch, green_patch], ['True likelihood', 'Fitted likelihoods'], loc='lower right')
fig.tight_layout()

fig.savefig(f'{current_dir}/logL_logX_plots.pdf', bbox_inches='tight')