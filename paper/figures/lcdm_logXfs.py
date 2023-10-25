import numpy as np
import matplotlib.pyplot as plt
from aeons.utils import *
from aeons.regress import *
from aeons.endpoint import *
from aeons.plotting import *
from aeons.beta import *
figsettings()

chains = ['SH0ES', 'BAO', 'lensing', 'planck']
fig, axs = plt.subplots(1, 4, figsize=(7, 1.6))
for i, chain in enumerate(chains):
    ax = axs[i]
    name, samples = get_samples(chain)
    true_endpoint = endpoints[name]
    iterations, *logXfs = read_from_txt(f'{data_dir}/logXfs/post/{name}.txt')
    logXfs = np.array(logXfs)
    logXs = samples.logX().iloc[iterations]
    logXf_true = samples.logX().iloc[true_endpoint]
    plot_quantiles(-logXs, -logXfs, -logXf_true, ylim=(0, 1.5), color='deepskyblue', ax=ax)
    ax.set_title(f'{name.upper()}, $\\mathcal{{D}}_\\mathrm{{KL}} = {samples.D_KL():.1f}$', fontsize=8)
    ax.set_yticks([0, -logXf_true], [0, f"{-logXf_true:.1f}"])
    ax.margins(x=0)

fig.supxlabel('$-\\log X$', y=0.02)
fig.supylabel('$-\\log \\hat{X_\\mathrm{f}}$', x=0.01)
fig.tight_layout()
fig.savefig('lcdm_logXfs.pdf', pad_inches=0, bbox_inches='tight')
