import numpy as np
import matplotlib.pyplot as plt
from aeons.endpoint import EndModel
from aeons.utils import *
from aeons.beta import *
from aeons.plotting import plot_quantiles
figsettings()

import os
current_dir = os.path.dirname(os.path.abspath(__file__))

d = [8, 16, '16_2000', 32]
titles = [r'$d = 8$', r'$d = 16$', r'$d = 16, n_\mathrm{live} = 2000$', r'$d = 32$']
fig, axs = plt.subplots(1, 4, figsize=(7, 1.6))
axs = axs.flatten()
for i, nDims in enumerate(d):
    name, samples = get_samples(f'gauss_{nDims}')
    model = EndModel(samples)
    true_endpoint = model.true_endpoint()
    logXf_true = samples.logX().iloc[true_endpoint]
    
    iterations, *logXfs = read_from_txt(f'{data_dir}/logXfs/post/{name}.txt')
    logXfs = np.array(logXfs)
    logXs = samples.logX().iloc[iterations]
    plot_quantiles(-logXs, -logXfs, -logXf_true, ylim=(0, 1.5), color='deepskyblue', ax=axs[i])
    axs[i].set_title(f'{titles[i]}\n{true_endpoint:,} total iterations', fontsize=7)
    axs[i].set_yticks([0, -logXf_true], [0, f"{-logXf_true:.1f}"])
    axs[i].margins(x=0)

fig.supxlabel('$-\\log X$', y=0.02, fontsize=8)
fig.supylabel('$-\\log \\hat{X}_\\mathrm{f}$', fontsize=8, x=0.01)
fig.tight_layout()
fig.savefig(f'{current_dir}/logXfs_gaussian.pdf', bbox_inches='tight')
