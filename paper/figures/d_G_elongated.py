import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from aeons.utils import points_at_iteration
from util import plot_quantiles, read_from_txt, figsettings
figsettings()

iterations, *d_Gs = read_from_txt(f'data/d_Gs/correlated_6d.txt')
logX_all = read_from_txt(f'data/logXs/correlated_6d.txt')[0]
logXs = logX_all[iterations.astype(int)]
d_Gs = np.array(d_Gs)

fig = plt.figure(figsize=(3.5, 3))
gs = GridSpec(2, 3, wspace=0.1, hspace=0.1)
ax1 = fig.add_subplot(gs[0, 0], zorder=1)
ax2 = fig.add_subplot(gs[0, 1], zorder=1)
ax3 = fig.add_subplot(gs[0, 2], zorder=1)
ax4 = fig.add_subplot(gs[1, :], zorder=0)
axs = [ax1, ax2, ax3, ax4]

logXstars = [-0.5, -8, -30]
# Find index of logX closest to logXstar
ndeads = [np.argmin(np.abs(logX_all - logXstar)) for logXstar in logXstars]
logL, logLbirth = read_from_txt(f'data/logLs/correlated_6d.txt')

for i, ndead in enumerate(ndeads):
    ax = axs[i]
    live_points = points_at_iteration(samples, ndead).iloc[ndead:]
    live_points.drop_weights().plot.scatter(0, 4, ax=ax, s=5, color='lightseagreen')
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax4.axvline(-logXstars[i], lw=1)
    
ax2.set_xlabel(r"$\sigma = 10^{-3}$", fontsize=8)
ax2.xaxis.set_label_position('top') 
ax1.set_ylabel(r"$\sigma = 10^{-6}$", fontsize=8)
plot_quantiles(-logXs, d_Gs, 6, (0, 1.2), color='lightseagreen', ax=ax4)
ax4.set_xlabel(r'$-\log X$', fontsize=8)
ax4.set_ylabel(r'$\hat{d}_G$', fontsize=8, rotation=0, labelpad=10)
ax4.margins(x=0)

dims = [1.5, 3.2, 6]
from matplotlib.patches import ConnectionPatch
for i, logXstar in enumerate(logXstars):
    con = ConnectionPatch(xyA=(0.5, 0), xyB=(-logXstar, dims[i]), coordsA="data", coordsB="data",
                      axesA=axs[i], axesB=ax4, color="C0", zorder=0.5)
    fig.add_artist(con)
    con.set_in_layout(False)

fig.suptitle('Live points and dimensionality as run proceeds', fontsize=8, y=0.95)
fig.savefig('d_G_elongated.pdf', pad_inches=0.02, bbox_inches='tight')
