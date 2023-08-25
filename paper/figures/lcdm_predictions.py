import numpy as np
import matplotlib.pyplot as plt

from aeons.utils import *
from aeons.plotting import *

figsettings()
fig, axs = plt.subplots(3, 4, figsize=(7, 4))
axs = axs.flatten()
fig.delaxes(axs[-1])
for i, chain in enumerate(lcdm_chains):
    ax = axs[i]
    axd = ax.twinx()
    name, samples = get_samples('lcdm', chain)
    iterations, logXfs, logXfs_std, true_endpoint = read_from_txt(f"{aeons_dir}/data/predictions/lcdm/nos/{chain}_bt_25_nos.txt")
    iterations_dG, d_Gs, d_Gs_std = read_from_txt(f'{aeons_dir}/data/predictions/lcdm/dG/{chain}_dG.txt')
    logXf_true = samples.logX().iloc[int(true_endpoint[0])]
    logXs = samples.logX().iloc[iterations]
    plot_std(-logXs, -logXfs, logXfs_std, -logXf_true, ylim=(0, 1.3), ax=ax)
    plot_std(-logXs, d_Gs, d_Gs_std, d_Gs[-1], ylim=(0, 2), ax=axd, color='lightsalmon')
    axd.set_yticks([d_Gs[-1]], [f'{d_Gs[-1]:.1f}'])
    axs[i].set_title(name, fontsize=8)
# Make legend with custom patches
from matplotlib.patches import Patch
lines = [Patch(color='deepskyblue', label='Endpoints'), Patch(color='lightsalmon', label='Effective dimensionality')]
fig.legend(handles=lines, ncol=1, fontsize=6, bbox_to_anchor=(0.97, 0.2))
fig.tight_layout()
fig.supxlabel('$-\\log X$')
fig.supylabel('$-\\log \hat{X}_\mathrm{f}$', x=0.02)
# Another ylabel on the right
fig.text(1, 0.5, '$\\tilde{d}$', va='center')
fig.tight_layout()
fig.savefig('lcdm_predictions.pdf', pad_inches=0, bbox_inches='tight')
