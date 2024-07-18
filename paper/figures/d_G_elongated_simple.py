import numpy as np
import matplotlib.pyplot as plt
from aeons.endpoint import EndModel
from aeons.utils import *
from aeons.beta import *
from aeons.plotting import plot_quantiles
figsettings()

import os
current_dir = os.path.dirname(os.path.abspath(__file__))

chain = 'elongated_1D_simple'
name, samples = get_samples(chain, reduced=False)
iterations, *d_Gs = read_from_txt(f'{data_dir}/d_Gs/post/{name}.txt')
logXs = samples.logX().iloc[iterations]
if chain == 'elongated_2D_simple':
    dims = [0.5, 1.02, 2.05]
    d_G_true = 2
    suptitle = '2D posterior'
    ylabel = '$\\hat{d}_G$'
else:
    dims = [0.5, 1, 1]
    d_G_true = 1
    suptitle = '1D posterior'
    ylabel = ''


import matplotlib.gridspec as gridspec
from anesthetic import make_2d_axes

fig = plt.figure(figsize=(3.5, 3))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
fig, ax0 = make_2d_axes([0, 1], diagonal=False, upper=False, subplot_spec=gs[0, 0], fig=fig, zorder=1)
fig, ax1 = make_2d_axes([0, 1], diagonal=False, upper=False, subplot_spec=gs[0, 1], fig=fig, zorder=1)
fig, ax2 = make_2d_axes([0, 1], diagonal=False, upper=False, subplot_spec=gs[0, 2], fig=fig, zorder=1)

for ax in [ax0, ax1, ax2]:
    samples.plot_2d(ax)
    ax = ax.iloc[0, 0]
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)

ax0.iloc[0, 0].set_ylabel('$\\theta_1$', fontsize=8, rotation=0, labelpad=10)
ax1.iloc[0, 0].set_title('$\\theta_2$', fontsize=8)
ax0.iloc[0, 0].text(-0.4, 0.3, 'A')
ax1.iloc[0, 0].text(-0.4, 0.3, 'B')
ax2.iloc[0, 0].text(-0.4, 0.3, 'C')


logXstars = [-0.12, -1.6, -6]
# Find index of logX closest to logXstar
logX_all = samples.logX()
ndeads = [np.argmin(np.abs(logX_all - logXstar)) for logXstar in logXstars]
for ndead, ax in zip(ndeads, [ax0, ax1, ax2]):
    points = points_at_iteration(samples, ndead)
    live_points = points.iloc[-500:]
    ax.iloc[0, 0].scatter(live_points[0], live_points[1], s=0.1, color='red')


ax3 = fig.add_subplot(gs[1, 0:3], zorder=0)
plot_quantiles(-logXs, d_Gs, d_G_true, ylim=(0, 2.6/d_G_true), ax=ax3, color='lightseagreen')
ax3.margins(x=0.02)
ax3.set_xlabel('$-\\log X$')
ax3.set_ylabel(ylabel, rotation=0, labelpad=15)


from matplotlib.patches import ConnectionPatch
for ax, logXstar, dim in zip([ax0, ax1, ax2], logXstars, dims):
    con = ConnectionPatch(xyA=(0, -0.5), xyB=(-logXstar, dim), coordsA="data", coordsB="data",
                      axesA=ax.iloc[0, 0], axesB=ax3, color="C0", zorder=0.5)
    fig.add_artist(con)
    con.set_in_layout(False)

fig.suptitle(suptitle, x=0.55)
fig.tight_layout()
fig.savefig(f'{current_dir}/d_G_{name}.pdf', bbox_inches='tight')
