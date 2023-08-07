import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['text.usetex'] = True 
rcParams['font.family'] = 'serif'


fig, ax = plt.subplots()

logLmax = 0
d = 30
n = 1000
avlogL = logLmax-d/2

logL = np.linspace(logLmax-d/2-np.sqrt(d/2)*3.5,0,1000)
logpdf = logL - logLmax + (d/2-1)*np.log(logLmax-logL)
logpdf -= np.max(logpdf)
ax.plot(logL, np.exp(logpdf), label='posterior: $\mathcal{P}(\log\mathcal{L})$')

logLs = logLmax-d/2/np.e
logL = np.linspace(logLs,0,1000)
logpdf = (d/2-1) * np.log(logLmax - logL)
logL = np.concatenate([[logLs], logL])
logpdf = np.concatenate([[-np.inf], logpdf])
logpdf -= np.max(logpdf)
ax.plot(logL, np.exp(logpdf), label='live points: $\mathcal{\pi}(\log\mathcal{L}|\mathcal{L}>\mathcal{L}_*)$')

from scipy.special import logsumexp

logL = np.linspace(logLs,0,1000)
logpdf = (d/2-1)*np.log(logLmax - logL) + (n-1)* logsumexp([ d/2*np.log(logLmax-logLs)*np.ones_like(logL), d/2*np.log(logLmax-logL)] , b=[np.ones_like(logL),-np.ones_like(logL)], axis=0)
logpdf -= np.max(logpdf)
ax.plot(logL, np.exp(logpdf), label='maximum live point: $P(\log\mathcal{L}_\mathrm{max}^\mathrm{live})$')


logLmaxlive = logLmax - d/2/np.e + np.log(n)/np.e

l1 = 1.3
ax.axvline(avlogL, color='k', linestyle=':')
ax.axvline(logLmax, color='k', linestyle=':')
ax.plot([avlogL, logLmax], [l1,l1], 'k-')
ax.annotate(r'$\frac{d}{2}$', ((avlogL+logLmax)/2,l1), ha='center', va='bottom')

l2 = 0.7
ax.plot([avlogL-np.sqrt(d/2), avlogL+np.sqrt(d/2)], [l2, l2], 'k-')
ax.annotate(r'$\pm\sqrt{\frac{d}{2}}$', (avlogL,l2), ha='left', va='bottom')

l3 = 1.05
ax.plot([avlogL, avlogL+1], [l3, l3], 'k-')
ax.annotate(r'$+1$', (avlogL+1,l3), ha='right', va='bottom')

l4 = 1.15
ax.plot([logLs, logLmax], [l4, l4], 'k-')
ax.annotate(r'$\frac{d}{2e}$', ((logLmax+logLs)/2,l4), ha='center', va='bottom')

l5 = 0.95
ax.plot([logLs, logLmaxlive], [l5, l5], 'k-')
ax.annotate(r'$\frac{\log n}{e}$', ((logLs+logLmaxlive)/2,l5), ha='center', va='bottom')

ax.set_xticks([avlogL, 
               logLs, 
               logLmaxlive,
               logLmax])
ax.set_xticklabels([r'$\langle\log\mathcal{L}\rangle_\mathcal{P}$',
                    r'$\log\mathcal{L}_*$',
                    r'$\log\mathcal{L}_\mathrm{max}^\mathrm{live}$',
                    r'$\log\mathcal{L}_\mathrm{max}$'])

ax.set_yticks([])
ax.legend(loc='upper left')
ax.set_ylim([0,1.4])

fig.set_size_inches(6,3)
fig.tight_layout()
fig.savefig('anatomy.pdf')
