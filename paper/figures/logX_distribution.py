import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from aeons.utils import *
from aeons.likelihoods import full
figsettings()

theta = [0, 10, 0.01]
logX = np.linspace(-60, 0, 1000)
X = np.exp(logX)
logL = full.func(X, theta)
L = np.exp(logL - logL.max())
LX = L * X
Ztrue = np.trapz(L, X)
LlogL = np.nan_to_num(L/Ztrue * np.log(L/Ztrue))
DKL = np.trapz(LlogL, X)

logXf = -44.4

ndead = 7000
nlive = 1000

def gaussian(x, mu, sig):
    return 1/(np.sqrt(2*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2)/2)
ElogXlivemin = -ndead/nlive - np.log(nlive) - 0.577
StdlogXlivemin = np.sqrt(ndead/nlive**2 + (np.pi**2)/6)
live_logX = logX[(logX > ElogXlivemin) * (logX < -ndead/nlive)]
dead_logX = logX[(logX > -ndead/nlive)]

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(logX, L/L.max(), color='black', label='Likelihood', lw=.5)
ax.fill(logX, LX/LX.max(), alpha=0.4, color='gray', label='Posterior mass')
lpoints = 0.4
# Live distribution
ax.fill_between(live_logX, np.zeros_like(live_logX), lpoints*np.exp(live_logX + ndead/nlive), alpha=0.5, color='deepskyblue', label='Dead/live distribution') 
# Dead distribution
ax.fill_between(dead_logX, 0, lpoints, alpha=0.5, color='deepskyblue')
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

# Arrows showing \Delta\log X
pad_text = 0.05
ax.annotate(r'$\Delta\log X = -\frac{1}{n_{i}}$', (-ndead/nlive - 1, lpoints+pad_text), ha='center', va='bottom', fontsize=8)
arrow_start = (-ndead/nlive, lpoints)
arrow_end = (-ndead/nlive - 1.5, lpoints)
arrow = patches.FancyArrowPatch(arrow_start, arrow_end, arrowstyle='->', mutation_scale=5)
ax.add_patch(arrow)

# Arrow showing direction of iterations
l1 = 0.7
arrow_start = (-22, l1)
arrow_end = (-28, l1)
arrow = patches.FancyArrowPatch(arrow_start, arrow_end, arrowstyle='->', mutation_scale=5)
ax.add_patch(arrow)
ax.annotate("Iterations", (-25, l1+pad_text), ha='center', va='bottom', fontsize=8)

ax.set_xticks([logXf, -DKL, ElogXlivemin, -ndead/nlive, 0], [r"$\log X_\mathrm{f}$", r'$-\mathcal{D}_\mathrm{KL}$', r'$\langle\log X_\mathrm{min}^\mathrm{live}\rangle$', r'$\langle\log X_i\rangle$', '$0$'], fontsize=8)
ax.set_yticks([])
ax.set_ylim(0, 1.05)
ax.margins(x=0)
ax.legend(loc='upper right', fontsize=8)
fig.supxlabel(r'$\log X$', y=0, fontsize=8)

fig.savefig('logX_distribution.pdf', bbox_inches='tight', pad_inches=0)
