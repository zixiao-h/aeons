import matplotlib.pyplot as plt

from aeons.utils import *
from aeons.plotting import *
figsettings()

samples = get_samples('toy', 'gauss_8')[1]
iterations, logLmaxs, logLmaxs_std = read_from_txt(f'{aeons_dir}/data/predictions/gauss/gauss_8_logLmaxs.txt')
_, ds, ds_std = read_from_txt(f'{aeons_dir}/data/predictions/gauss/gauss_8_ds.txt')
_, sigmas, sigmas_std = read_from_txt(f'{aeons_dir}/data/predictions/gauss/gauss_8_sigmas.txt')
logXs = samples.logX().iloc[iterations]

fig, axs = plt.subplots(1, 3, figsize=(7, 1.6)) 
plot_std(-logXs, logLmaxs, logLmaxs_std, true=0, ax=axs[0], color='salmon')
plot_std(-logXs, ds, ds_std, true=8, ylim=(0, 2), ax=axs[1], color='salmon')
plot_std(-logXs, sigmas, sigmas_std, true=0.01, ylim=(0, 3), ax=axs[2], color='salmon')
axs[0].set_title("$\\log \\mathcal{L}_\\mathrm{max}$", fontsize=8)
axs[1].set_title("$d$", fontsize=8)
axs[2].set_title("$\\sigma$", fontsize=8)
# fig.suptitle('Inferred parameters', y=.9, fontsize=8)
fig.supxlabel(r"$-\log X$", y=.05, fontsize=8)
fig.tight_layout()
fig.savefig('gauss_params.pdf', pad_inches=0, bbox_inches='tight')
