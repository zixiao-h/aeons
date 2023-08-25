import matplotlib.pyplot as plt
from aeons.utils import *
import matplotlib.patches as patches
figsettings()

def get_beta(points, ndead):
    logX = points.logX().iloc[ndead]
    if logX < -points.D_KL():
        return 1
    def func(beta):
        return logX + points.set_beta(beta).D_KL()
    from scipy import optimize
    res = optimize.root_scalar(func, bracket=[0, 1])
    return res.root

name, samples = get_samples('toy', 'planck_gaussian')
ndead = 8000
points = points_at_iteration(samples, ndead).recompute()
betas = [1, get_beta(points, ndead) * 1.5, get_beta(points, ndead) * 0.5, get_beta(points, ndead) * 0.05]
labels = ['$\\beta=1$', '$\\beta^* < \\beta < 1$', None, '$\\beta = \\beta^*$']
colors = ['black', 'gray', 'darkgray', 'orange']

fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.3))
for i, beta in enumerate(betas):
    axs[0].plot(-points.logX(), np.exp(beta*(points.logL - points.logL.max())), 'x', ms=1, color=colors[i], label=labels[i])
    axs[0].plot(-points.logX(), np.exp(beta*(points.logL - points.logL.max())), lw=.5, alpha=.2, color=colors[i])
    axs[1].plot(-points.logX(), weights(points.logL, points.logX(), beta=beta), lw=.8, color=colors[i])
axs[0].plot(-points.logX().iloc[-1], np.exp(points.logL.iloc[-1] - points.logL.max()), 'x', ms=2, color='k')

l = .2
arrow_start = (6, l)
arrow_end = (2, l)
arrow = patches.FancyArrowPatch(arrow_start, arrow_end, arrowstyle='->', mutation_scale=5)
axs[0].add_patch(arrow)
axs[0].annotate(r'Lower $\beta$', xy=(4, l), xytext=(4, l+.1), ha='center', va='bottom', fontsize=5)

axs[1].axvline(x=-points.logX().iloc[ndead], ls='--', color='k', lw=.5)
for ax in axs:
    ax.margins(x=0.02)
    ax.set_ylim(-0.01, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks([-points.logX().iloc[ndead]], [r'$-\log X^*$'])
axs[0].set_title(r'$\mathcal{L}^\beta$', fontsize=8)
axs[1].set_title('Posterior mass', fontsize=8)
axs[0].legend(loc='upper left', fontsize=5)

fig.subplots_adjust(left=.05, right=.95, wspace=.1, hspace=.1)
fig.savefig('last_live_point.pdf', pad_inches=0.02, bbox_inches='tight')
