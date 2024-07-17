import numpy as np
import matplotlib.pyplot as plt
from aeons.endpoint import EndModel
from aeons.utils import *
from aeons.beta import *
from aeons.plotting import plot_quantiles
figsettings()

import os
current_dir = os.path.dirname(os.path.abspath(__file__))

chains = ['correlated_6d', 'correlated_6d_close']
titles = ['$\\Sigma_\\mathrm{{separated}}$', '$\\Sigma_\\mathrm{{close}}$']
fig, axs = plt.subplots(1, 2, figsize=(3.3, 1.5))

for chain, title, ax in zip(chains, titles, axs):
    name, samples = get_samples(chain, reduced=False)
    iterations, *d_Gs = read_from_txt(f'{data_dir}/d_Gs/post/{name}.txt')
    logXs = samples.logX().iloc[iterations]
    plot_quantiles(-logXs, d_Gs, 6, (0, 1.3), ax=ax, color='lightseagreen')
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('$-\\log X$')
    ax.set_yticks([0, 3, 6], [0, 3, 6])

axs[0].set_ylabel('$d_\\mathrm{G}$')
fig.tight_layout()
fig.savefig(f'{current_dir}/d_G_elongated_comparison.pdf', bbox_inches='tight')