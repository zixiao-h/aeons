import matplotlib.pyplot as plt
from aeons.utils import *
from aeons.beta import *
import matplotlib.patches as patches
figsettings()

name, samples = get_samples('planck_gaussian')
ndead = 8000
points = points_at_iteration(samples, ndead)
beta_grad = np.exp(get_logbeta_grad(points, ndead))
betas = [0, beta_grad, 1]
labels = ['$\\beta=1$', '$\\beta = \\beta^*$', '$\\beta = 0$']
colors = ['darkgray', 'orange', 'black']

fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.3))
for i, beta in enumerate(betas):
    axs[0].plot(-points.logX(), np.exp(beta*(points.logL - points.logL.max())), 'x', ms=1, color=colors[i], label=labels[i])
    axs[0].plot(-points.logX(), np.exp(beta*(points.logL - points.logL.max())), lw=.5, alpha=.2, color=colors[i])
    logw = beta * points.logL + points.logX()
    logw -= logw.max()
    axs[1].fill_between(-points.logX(), 0, np.exp(logw), lw=.8, color=colors[i])
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
    ax.set_yticks([])
    ax.set_xticks([0, -points.logX().iloc[ndead]], [0, r'$-\log X^*$'])
axs[0].set_title(r'$\mathcal{L}^\beta$', fontsize=8)
axs[1].set_title('$\\mathcal{P}(\\log X) = \\mathcal{L}^\\beta X$', fontsize=8)
axs[0].legend(loc='upper left', fontsize=5)
# axs[1].margins(x=0)

fig.subplots_adjust(left=.01, right=.99, wspace=.05)
fig.savefig('last_live_point.pdf', pad_inches=0, bbox_inches='tight')
