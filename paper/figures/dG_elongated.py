import numpy as np
import matplotlib.pyplot as plt
from aeons.utils import *
from aeons.plotting import plot_std
from matplotlib.gridspec import GridSpec
from anesthetic import make_2d_axes
figsettings()

name, samples = get_samples('correlated_6d')
iterations, d_Gs, d_Gs_std = read_from_txt(f'{aeons_dir}/data/predictions/gauss/gauss_36_dG.txt')

fig = plt.figure(figsize=(3.5, 4))
gs = GridSpec(3, 3, wspace=0.1, hspace=.7)
_, ax1 = make_2d_axes([0, 1], upper=False, diagonal=False, fig=fig, subplot_spec=gs[0, 0])
_, ax2 = make_2d_axes([0, 1], upper=False, diagonal=False, fig=fig, subplot_spec=gs[0, 1])
_, ax3 = make_2d_axes([0, 1], upper=False, diagonal=False, fig=fig, subplot_spec=gs[0, 2])
ax4 = fig.add_subplot(gs[1, :])
ax5 = fig.add_subplot(gs[2, :])

samples.set_beta(1e-6).plot_2d(ax1)
samples.set_beta(1e-4).plot_2d(ax2)
samples.set_beta(1).plot_2d(ax3)

betas = np.logspace(-6, 0, 100)
# samples.d_G(beta=betas).plot(ax=ax4, logx=True, lw=1, color='k')
d_Gs_beta = samples.d_G(nsamples=10, beta=betas).unstack()
plot_std(np.log10(betas), d_Gs_beta.mean(axis=1), d_Gs_beta.std(axis=1), ax=ax4)
plot_std(-samples.logX().iloc[iterations], d_Gs, d_Gs_std, ax=ax5)

ax4.axvline(x=-4.3, lw=.5, ls='--', color='gray')
ax4.axvline(x=-2, lw=.5, ls='--', color='gray')
ax5.axvline(x=.5, lw=.5, ls='--', color='gray')
ax5.axvline(x=2.8, lw=.5, ls='--', color='gray')

for ax in [ax1, ax2, ax3]:
    ax = ax.iloc[0,0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')

ax2.iloc[0,0].set_xlabel(r'$\theta_1$')
ax1.iloc[0,0].set_ylabel(r'$\theta_2$', rotation=0, labelpad=8)
# ax4.set_xticks(np.arange(-6, 1, dtype=float), [f"{beta:.0f}" for beta in np.arange(-6, 1)])
ax4.set_xlabel('')
ax4.set_title('$\\tilde{d}(\\log \\beta)$')
ax4.minorticks_off()
ax5.set_title('$\\tilde{d}(-\\log X)$')
fig.suptitle('Gaussian with $\\Sigma = \\mathrm{diag}(10^{-3}, 10^{-6})$', y=.95)

fig.savefig('dG_beta.pdf', pad_inches=0, bbox_inches='tight')
