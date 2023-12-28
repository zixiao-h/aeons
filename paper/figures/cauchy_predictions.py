import numpy as np
import matplotlib.pyplot as plt
from util import read_from_txt, figsettings, plot_quantiles, endpoints, DKLs
figsettings()

fig, axs = plt.subplots(1, 4, figsize=(7, 1.6))
for i, exp in enumerate([2, 3, 4, 5]):
    ax = axs[i]
    name = f'cauchy_8{exp}'
    true_endpoint = endpoints[name]
    iterations, *logXfs = read_from_txt(f'data/logXfs/{name}.txt')
    logXfs = np.array(logXfs)
    logX_all = read_from_txt(f'data/logXs/{name}.txt')[0]
    logXs = logX_all[iterations.astype(int)]
    logXf_true = logX_all[true_endpoint]
    plot_quantiles(-logXs, -logXfs, -logXf_true, ylim=(0, 2), color='deepskyblue', ax=ax)
    ax.set_title(f'$\\gamma=10^{{{-exp}}}$, $\\mathcal{{D}}_\\mathrm{{KL}} = {DKLs[name]:.1f}$', fontsize=8)
    ax.set_yticks([0, -logXf_true], [0, f"{-logXf_true:.1f}"])
    ax.margins(x=0)

fig.supxlabel('$-\\log X$', y=0.02)
fig.supylabel('$-\\log \\hat{X_\\mathrm{f}}$', x=0.01)
fig.tight_layout()
fig.savefig('cauchy_predictions.pdf', pad_inches=0, bbox_inches='tight')
