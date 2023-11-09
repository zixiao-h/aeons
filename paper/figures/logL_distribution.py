import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from aeons.utils import figsettings

figsettings()
fig, ax = plt.subplots()

logLmax = 0
d = 30
n = 1000
avlogL = logLmax-d/2

logLs = logLmax-d/2/np.e
logLdead = np.linspace(logLmax-d/2-np.sqrt(d/2)*4, logLs, 1000)
logLlive = np.linspace(logLs,0,1000)
logLall = np.concatenate([logLdead, logLlive])

logpdf = logLall - logLmax + (d/2-1)*np.log(logLmax-logLall)
logpdf -= np.max(logpdf)
ax.fill(logLall, np.exp(logpdf), color='gray', alpha=.4, label='Posterior')

ldist = .5
pdf_dead = d/2 / (-logLdead) 
pdf_dead /= np.max(pdf_dead)
ax.fill_between(logLdead, 0, ldist*pdf_dead, color='deepskyblue', alpha=0.5, label='Dead/live distribution')

logpdf_live = (d/2-1) * np.log(logLmax - logLlive)
logpdf_live -= np.max(logpdf_live)
ax.fill_between(logLlive, 0, ldist*np.exp(logpdf_live), color='deepskyblue', alpha=0.5)


from scipy.special import logsumexp
logpdf_maxlive = (d/2-1)*np.log(logLmax - logLlive) + (n-1)* logsumexp([ d/2*np.log(logLmax-logLs)*np.ones_like(logLlive), d/2*np.log(logLmax-logLlive)] , b=[np.ones_like(logLlive),-np.ones_like(logLlive)], axis=0)
logpdf_maxlive -= np.max(logpdf_maxlive)
ax.plot(logLlive, .7*np.exp(logpdf_maxlive), lw=.5, color='red', label='Max live point')


logLmaxlive = logLmax - d/2/np.e + np.log(n)/np.e

pad_text = 0.02
l1 = 1.2
ax.axvline(avlogL, color='k', linestyle=':', lw=1)
ax.axvline(logLmax, color='k', linestyle=':', lw=1)
ax.plot([avlogL, logLmax], [l1,l1], 'k-', lw=.5)
ax.annotate(r'$\frac{d}{2}$', ((avlogL+logLmax)/2,l1+pad_text), ha='center', va='bottom', fontsize=8)

l2 = 0.7
ax.plot([avlogL-np.sqrt(d/2), avlogL+np.sqrt(d/2)], [l2, l2], 'k-', lw=.5)
ax.annotate(r'$\pm\sqrt{\frac{d}{2}}$', (avlogL+0.3,l2+pad_text), ha='left', va='bottom', fontsize=8)

l3 = 1.05
ax.plot([avlogL, avlogL+1], [l3, l3], 'k-', lw=.5)
ax.annotate(r'$+1$', (avlogL+2.5,l3), ha='right', va='bottom', fontsize=8)

l4 = 1 
ax.plot([logLs, logLmax], [l4, l4], 'k-', lw=.5)
ax.annotate(r'$\frac{d}{2e}$', ((logLmax+logLs)/2,l4+pad_text), ha='center', va='bottom', fontsize=8)

l5 = 0.75
ax.plot([logLs, logLmaxlive], [l5, l5], 'k-', lw=.5)
ax.annotate(r'$\frac{\log n}{e}$', ((logLs+logLmaxlive)/2,l5+pad_text), ha='center', va='bottom', fontsize=8)

ax.annotate("Iterations", xytext=(-28, 0.5), xy=(-23, 0.52), arrowprops=dict(arrowstyle="->"), fontsize=8)

ax.set_xticks([avlogL, 
               logLs, 
               logLmaxlive,
               logLmax])
ax.set_xticklabels([r'$\langle\log\mathcal{L}\rangle_\mathcal{P}$',
                    r'$\log\mathcal{L}_i$',
                    r'$\log\mathcal{L}_\mathrm{max}^\mathrm{live}$',
                    r'$\log\mathcal{L}_\mathrm{max}$'],
                    fontsize=8)

ax.set_yticks([])
ax.legend(loc='upper left', fontsize=8)
ax.set_ylim([0,1.4])
ax.margins(x=0)

fig.set_size_inches(8, 3)
# fig.tight_layout()
# fig.autofmt_xdate(rotation=25)
fig.savefig('logL_distribution.pdf', pad_inches=0, bbox_inches='tight')
