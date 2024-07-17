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


fig, axs = plt.subplots(1, 2, figsize=(4, 1.5))
chains = ['rosenbrock_10', 'cauchy_84']
titles = ['Rosenbrock', 'Cauchy']

for ax, chain, title in zip(axs, chains, titles):
    name, samples = get_samples(chain)
    model = EndModel(samples)
    true_endpoint = model.true_endpoint()
    logXf_true = samples.logX().iloc[true_endpoint]

    iterations, *logXfs = read_from_txt(f'{data_dir}/logXfs/post/{name}.txt')
    logXfs = np.array(logXfs)
    logXs = samples.logX().iloc[iterations]
    plot_quantiles(-logXs, -logXfs, -logXf_true, (0, 1.5), ax=ax)
    ax.set_title(f'{title}\n{true_endpoint:,} total iterations', fontsize=7)
    ax.margins(x=0.01)

axs[1].set_ylim(0, 150)
fig.supxlabel('$-\\log X$', y=0.05)
fig.supylabel('$-\\log \\hat{{X}}_\\mathrm{{f}}$', x=0)
fig.tight_layout()

fig.savefig(f'{current_dir}/logXfs_non_gaussian.pdf', pad_inches=0, bbox_inches='tight')
