import numpy as np
import matplotlib.pyplot as plt
from aeons.utils import *
from aeons.plotting import *
from matplotlib.gridspec import GridSpec
figsettings()

name, samples = get_samples('correlated_6d', reduced=False)
iterations, *d_Gs = read_from_txt(f'{data_dir}/d_Gs/post/{name}.txt')
logXs = samples.logX().iloc[iterations]
d_Gs = np.array(d_Gs)

fig = plt.figure(figsize=(3.5, 3))
gs = GridSpec(2, 3, wspace=0.1, hspace=0.35)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, :])
axs = [ax1, ax2, ax3, ax4]

logXstars = [-1, -8, -30]
logX = samples.logX()
# Find index of logX closest to logXstar
ndeads = [np.argmin(np.abs(logX - logXstar)) for logXstar in logXstars]

for i, ndead in enumerate(ndeads):
    ax = axs[i]
    live_points = points_at_iteration(samples, ndead).iloc[ndead:]
    live_points.drop_weights().plot.scatter(0, 4, ax=ax, s=5)
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax4.axvline(-logXstars[i], lw=1)
ax2.set_xlabel(r"$\sigma = 10^{-3}$", fontsize=8)
ax1.set_ylabel(r"$\sigma = 10^{-6}$", fontsize=8)
plot_quantiles(-logXs, d_Gs, 6, (0, 1.2), color='lightseagreen', ax=ax4)
ax4.set_title('Inferred dimensionality as run proceeds', fontsize=8)
ax4.set_xlabel(r'$-\log X$', fontsize=8)
ax4.set_ylabel(r'$\hat{d}_G$', fontsize=8, rotation=0, labelpad=10)
ax4.margins(x=0)
fig.suptitle('Set of live points as run proceeds', fontsize=8, y=0.92)
fig.savefig('d_G_elongated.pdf', pad_inches=0.02, bbox_inches='tight')
