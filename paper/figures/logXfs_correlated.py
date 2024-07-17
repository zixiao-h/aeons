import numpy as np
import matplotlib.pyplot as plt
from aeons.endpoint import EndModel
from aeons.utils import *
from aeons.beta import *
from aeons.plotting import plot_quantiles
figsettings()

import os
current_dir = os.path.dirname(os.path.abspath(__file__))


fig, axs = plt.subplots(1, 4, figsize=(7, 1.6))
chains = ['correlated_6d', 'correlated_3d', 'correlated_6d_close', 'correlated_6d_rotated']
titles = ['$\\Sigma_\\mathrm{{separated}}$', '$\\Sigma_{{3d}}$', '$\\Sigma_\\mathrm{{close}}$', '$\\Sigma_\\mathrm{{separated, rotated}}$']


for chain, title, ax in zip(chains, titles, axs):
    name, samples = get_samples(chain)
    model = EndModel(samples)
    true_endpoint = model.true_endpoint()
    logXf_true = samples.logX().iloc[true_endpoint]
    
    iterations, *logXfs = read_from_txt(f'{data_dir}/logXfs/post/{name}.txt')
    logXfs = np.array(logXfs)
    logXs = samples.logX().iloc[iterations]
    plot_quantiles(-logXs, -logXfs, -logXf_true, (0, 1.3), ax=ax)
    ax.set_title(f'{title}\n{true_endpoint:,} total iterations', fontsize=7)    
    ax.set_ylim(0, 45)
    ax.margins(x=0.01)

for ax in axs.flat[1:]:
    ax.set_yticks([])
    
fig.supxlabel('$-\\log X$', y=0.05)
fig.supylabel('$-\\log \\hat{{X}}_\\mathrm{{f}}$', x=0)
fig.tight_layout()

fig.savefig(f'{current_dir}/logXfs_correlated.pdf', pad_inches=0, bbox_inches='tight')