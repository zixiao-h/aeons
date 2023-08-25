import numpy as np
import matplotlib.pyplot as plt
from aeons.utils import *
figsettings()

def weights(logL, logX, beta):
    logw = beta*logL + logX
    return np.exp(logw - np.max(logw))

name, samples = get_samples('toy', 'planck_gaussian')
samples = samples.iloc[:int(len(samples)*0.7)]
logL = samples.logL
logX = samples.logX()
betastar = 1e-3
DKL = samples.D_KL()
DKLstar = samples.set_beta(betastar).D_KL()


fig, axl = plt.subplots(figsize=(3.5, 1.4))
axw = axl.twinx()
axw.fill(-logX, weights(logL, logX, beta=1), label='$\\beta=1$', alpha=.5, color='gray')
axw.fill(-logX, weights(logL, logX, beta=betastar), label='$\\beta=10^{-2}$', alpha=.5, color='orange')
axl.plot(-logX, np.exp(logL - np.max(logL)), lw=1, color='k')
axl.plot(-logX, np.exp(betastar*(logL - np.max(logL))), lw=1, color='darkorange')
axl.axvline(x=DKL, ls='--', color='k', lw=.5)
axl.axvline(x=DKLstar, ls='--', color='darkorange', lw=.5)

for ax in [axw, axl]:
    ax.set_yticks([])
    ax.margins(x=0)
    ax.set_ylim(0, 1.1)
axl.set_xticks([DKL, DKLstar], [r'$\mathcal{D}_{\mathrm{KL}}$', r'$-\log X^*$'])
axl.set_xlabel(r'$-\log X$')
axw.legend(loc='upper left', fontsize=6)

fig.tight_layout()
fig.savefig('beta_move.pdf', pad_inches=0, bbox_inches='tight')
