import numpy as np
import matplotlib.pyplot as plt
from aeons.endpoint import EndModel
from aeons.utils import *
from aeons.beta import *
from aeons.plotting import plot_quantiles
figsettings()

import os
current_dir = os.path.dirname(os.path.abspath(__file__))



num_chain = 1
ylim = (0, 1.5)

chains = ['gauss_32', 'correlated_6d', 'correlated_6d_close']
chains_titles = ['Gaussian', 'Elongated Gaussian, two well-separated length scales', 'Elongated Gaussian, gradually changing length scales']
chain, chain_title = chains[num_chain], chains_titles[num_chain]
methods = ['term_3', 'grad', 'logL', 'post']
beta_types = ['Termination, $\\epsilon=10^{-3}$', 'Microcanonical', 'Canonical', 'Bayesian']

fig, axs = plt.subplots(1, 4, figsize=(7, 1.8))
name, samples = get_samples(chain)
model = EndModel(samples)
endpoint_true = model.true_endpoint()
for i, ax in enumerate(axs):
    logXf_true = samples.logX().iloc[endpoint_true]
    iterations, *logXfs = read_from_txt(f'{data_dir}/logXfs/{methods[i]}/{name}.txt')
    logXfs = np.array(logXfs)
    logXs = samples.logX().iloc[iterations]
    plot_quantiles(-logXs, -logXfs, -logXf_true, ylim=ylim, color='deepskyblue', ax=ax)
    ax.set_xlim(0, -logXs.iloc[-1] * 1.1)
    if num_chain == 0:
        ax.set_title(beta_types[i], fontsize=6)

for j in range(1, 4):
    axs[j].set_yticks([])

fig.supxlabel('$-\\log X$', y=0.05)
fig.supylabel('$-\\log \\hat{X}_f$')
fig.suptitle(f'{chain_title} ({endpoint_true:,} total iterations)', y=0.95)
fig.tight_layout()
fig.savefig(f'{current_dir}/beta_endpoint_comparison_{chain}.pdf', bbox_inches='tight')