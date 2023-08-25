import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from aeons.utils import *
from aeons.likelihoods import full

theta = [0, 10, 0.01]
logX = np.linspace(-60, 0, 1000)
X = np.exp(logX)
logL = full.func(X, theta)
L = np.exp(logL - logL.max())
LX = L * X
Ztrue = np.trapz(L, X)
LlogL = np.nan_to_num(L/Ztrue * np.log(L/Ztrue))
DKL = np.trapz(LlogL, X)

ndead = 7000
nlive = 1000

def gaussian(x, mu, sig):
    return 1/(np.sqrt(2*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2)/2)
ElogXlivemin = -ndead/nlive - np.log(nlive) - 0.577
StdlogXlivemin = np.sqrt(ndead/nlive**2 + (np.pi**2)/6)
live_logX = logX[(logX > ElogXlivemin) * (logX < -ndead/nlive)]

figsettings()
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(logX, L/L.max(), color='black', label='Likelihood', lw=.5)
ax.fill(logX, LX/LX.max(), alpha=0.4, color='gray', label='Posterior mass')
ax.fill_between(live_logX, np.zeros_like(live_logX), 0.5*np.exp(live_logX + ndead/nlive), alpha=0.5, color='deepskyblue', label='Live points') 
ax.plot(logX, gaussian(logX, ElogXlivemin, StdlogXlivemin), lw=.5, color='red', label='Min live point')
ax.axvline(-ndead/nlive, color='deepskyblue', lw=.5, ls='--')

# Arrows showing log n + gamma
l = 0.32
pad_text = 0.05
ax.annotate(r'$\log n+\gamma$', ((-ndead/nlive+ ElogXlivemin)/2 - 3,l+pad_text), ha='center', va='bottom', fontsize=8)
arrow_start = (-ndead/nlive, l)
arrow_end = (ElogXlivemin, l)
arrow = patches.FancyArrowPatch(arrow_start, arrow_end, arrowstyle='->', mutation_scale=5)
ax.add_patch(arrow)

# Arrow showing direction of iterations
l1 = 0.7
arrow_start = (-22, l1)
arrow_end = (-28, l1)
arrow = patches.FancyArrowPatch(arrow_start, arrow_end, arrowstyle='->', mutation_scale=5)
ax.add_patch(arrow)
ax.annotate("Iterations", (-25, l1+pad_text), ha='center', va='bottom', fontsize=8)

ax.set_xticks([-DKL, ElogXlivemin, -ndead/nlive, 0], [r'$-\mathcal{D}_\mathrm{KL}$', r'$\langle\log X_\mathrm{min}^\mathrm{live}\rangle$', r'$\langle\log X_*\rangle$', '$0$'], fontsize=8)
ax.set_yticks([])
ax.set_ylim(0, 1.05)
ax.margins(x=0)
ax.legend(loc='upper right', fontsize=8)
fig.supxlabel(r'$\log X$', y=0, fontsize=8)

fig.savefig('live_point_distribution.pdf', pad_inches=0, bbox_inches='tight')
