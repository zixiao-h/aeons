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

chains = ['SH0ES', 'BAO', 'lensing', 'planck']
titles = ['SH0ES', 'BAO', 'LENSING', 'PLANCK']
fig, axs = plt.subplots(1, 4, figsize=(7, 1.6))
for ax, chain, title in zip(axs, chains, titles):
    name, samples = get_samples(chain)
    model = EndModel(samples)
    true_endpoint = model.true_endpoint()
    logXf_true = samples.logX().iloc[true_endpoint]
    
    iterations, *logXfs = read_from_txt(f'{data_dir}/logXfs/post/{name}.txt')
    logXfs = np.array(logXfs)
    logXs = samples.logX().iloc[iterations]
    logXf_true = samples.logX().iloc[true_endpoint]
    plot_quantiles(-logXs, -logXfs, -logXf_true, ylim=(0, 1.5), color='deepskyblue', ax=ax)
    ax.set_title(f'{title}\n{true_endpoint:,} total iterations', fontsize=7)
    ax.set_yticks([0, -logXf_true], [0, f"{-logXf_true:.1f}"])
    ax.margins(x=0)

fig.supxlabel('$-\\log X$', y=0.02)
fig.supylabel('$-\\log \\hat{X_\\mathrm{f}}$', x=0.01)
fig.tight_layout()

fig.savefig(f'{current_dir}/logXfs_lcdm.pdf', pad_inches=0, bbox_inches='tight')